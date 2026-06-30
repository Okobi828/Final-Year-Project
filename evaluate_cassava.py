#!/usr/bin/env python3
"""
evaluate_cassava.py
====================
Standalone evaluation script for the saved Cassava EfficientNetB4 classifier
produced by train_cassava_co.py.

Supports two saved-model formats:
  1. Full SavedModel  → cassava_fyp_cyclegan/model/final_classifier_savedmodel/
  2. Weights-only     → cassava_fyp_cyclegan/model/final_classifier.weights.h5
     (requires rebuilding the model architecture before loading weights)

Usage
-----
# Evaluate on a held-out directory (walks subfolders named after class labels):
    python evaluate_cassava.py --data_dir /path/to/data

# Point at a different project dir (default: ./cassava_fyp_cyclegan):
    python evaluate_cassava.py --data_dir /path/to/data --project_dir /path/to/cassava_fyp_cyclegan

# If you saved weights only, the script detects that and rebuilds the architecture.
# Use --weights_only to force that path:
    python evaluate_cassava.py --data_dir /path/to/data --weights_only

# Evaluate on specific split CSVs that were saved during training:
    python evaluate_cassava.py --use_saved_splits

Output
------
All results are written to <project_dir>/reports/eval_standalone/:
  - eval_metrics.json
  - eval_classification_report.csv
  - eval_confusion_matrix.csv
  - eval_confusion_matrix.png
"""

import os
import glob
import json
import argparse
import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG  (must match the values used during training)
# =============================================================================

@dataclass
class CFG:
    seed: int = 42
    img_size: int = 224
    num_classes: int = 5
    class_names: tuple = ("Healthy", "CMD", "CBB", "CGM", "CBSD")
    batch_size: int = 32

cfg = CFG()
AUTOTUNE = tf.data.AUTOTUNE


# =============================================================================
# ARCHITECTURE  (identical to train_cassava_co.py – needed for weights-only load)
# =============================================================================

def build_classifier():
    """Rebuild the EfficientNetB4 + attention head used in training."""
    inputs = tf.keras.Input(shape=(cfg.img_size, cfg.img_size, 3))
    backbone = EfficientNetB4(include_top=False, weights=None, input_tensor=inputs)
    backbone.trainable = False

    x = backbone.output
    # Channel attention (SE-style)
    gap = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(128, activation="relu")(gap)
    se = layers.Dense(tf.keras.backend.int_shape(x)[-1], activation="sigmoid")(se)
    x = layers.Multiply()([x, se[:, None, None, :]])
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(cfg.num_classes, dtype="float32")(x)

    model = Model(inputs, outputs)
    return model, backbone


# =============================================================================
# DATA LOADING
# =============================================================================

def decode_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [cfg.img_size, cfg.img_size], method="bicubic")
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def build_dataset(df, batch_size):
    paths = df["image_path"].astype(str).values
    labels = df["label_idx"].astype(np.int32).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (decode_image(p), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def load_images_from_dir(data_dir):
    """
    Walk <data_dir>/<class_folder>/*.jpg and build a DataFrame.
    Accepts both Cassava___<name> folder names and short names (Healthy, CMD, …).
    """
    folder_map = {
        "Cassava___healthy": "Healthy",
        "Cassava___mosaic_disease": "CMD",
        "Cassava___bacterial_blight": "CBB",
        "Cassava___green_mottle": "CGM",
        "Cassava___brown_streak_disease": "CBSD",
    }
    label_to_idx = {name: idx for idx, name in enumerate(cfg.class_names)}

    rows = []
    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        label = folder_map.get(folder, folder)   # try map, else use raw folder name
        if label not in label_to_idx:
            print(f"  ⚠️  Skipping unrecognized folder: {folder}")
            continue
        for p in sorted(glob.glob(os.path.join(folder_path, "*"))):
            if p.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({"image_path": os.path.abspath(p), "label": label,
                             "label_idx": label_to_idx[label]})

    if not rows:
        raise RuntimeError(f"No images found under {data_dir}")

    df = pd.DataFrame(rows)
    print(f"\nLoaded {len(df):,} images across {df['label'].nunique()} classes")
    print("Class distribution:", dict(Counter(df["label"])))
    return df


def load_split_csv(csv_path):
    """Load a split CSV saved by the training script."""
    df = pd.read_csv(csv_path)
    required = {"image_path", "label", "label_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} is missing columns: {missing}")
    # Keep only real images to match the training evaluation convention
    if "is_synthetic" in df.columns:
        df = df[df["is_synthetic"] == 0].reset_index(drop=True)
    return df


# =============================================================================
# EVALUATION + REPORTING
# =============================================================================

def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix plot saved → {save_path}")


def run_evaluation(model, df, report_dir, split_name="eval"):
    os.makedirs(report_dir, exist_ok=True)
    ds = build_dataset(df, cfg.batch_size)
    y_true = df["label_idx"].values.astype(int)

    print(f"\n🔍 Running inference on {len(df):,} images …")
    probs = model.predict(ds, verbose=1)
    pred = probs.argmax(axis=1)

    metrics = {
        "split": split_name,
        "n_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "mean_confidence": float(np.max(probs, axis=1).mean()),
        "low_confidence_rate_lt_0_60": float((np.max(probs, axis=1) < 0.60).mean()),
    }
    try:
        metrics["macro_auc_ovr"] = float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        )
    except Exception as e:
        metrics["macro_auc_ovr"] = None
        print(f"  AUC skipped: {e}")

    report = classification_report(
        y_true, pred, target_names=cfg.class_names, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, pred)

    # ── save artefacts ──
    metrics_path = os.path.join(report_dir, f"{split_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    report_csv = os.path.join(report_dir, f"{split_name}_classification_report.csv")
    pd.DataFrame(report).T.to_csv(report_csv)

    cm_csv = os.path.join(report_dir, f"{split_name}_confusion_matrix.csv")
    pd.DataFrame(cm, index=list(cfg.class_names), columns=list(cfg.class_names)).to_csv(cm_csv)

    cm_png = os.path.join(report_dir, f"{split_name}_confusion_matrix.png")
    plot_confusion_matrix(cm, list(cfg.class_names), cm_png)

    # ── print summary ──
    print(f"\n{'=' * 50}")
    print(f"  EVALUATION RESULTS  —  {split_name.upper()}")
    print(f"{'=' * 50}")
    print(json.dumps(metrics, indent=2))
    print()
    print(classification_report(y_true, pred, target_names=cfg.class_names, zero_division=0))
    print(f"Reports saved → {report_dir}")

    return metrics


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(project_dir, weights_only=False):
    savedmodel_path = os.path.join(project_dir, "model", "final_classifier_savedmodel")
    weights_path = os.path.join(project_dir, "model", "final_classifier.weights.h5")

    if not weights_only and os.path.isdir(savedmodel_path):
        print(f"\n📦 Loading TF SavedModel from:\n  {savedmodel_path}")
        loaded = tf.saved_model.load(savedmodel_path)
        infer = loaded.signatures["serving_default"]
        print("✅ SavedModel loaded.")
        print("  Output keys:", list(infer.structured_outputs.keys()))

        # Wrap the raw TF SavedModel into a callable that returns a numpy prob array,
        # matching the interface expected by run_evaluation / model.predict().
        class WrappedModel:
            def __init__(self, infer_fn):
                self.infer_fn = infer_fn
                # Grab the single output key (e.g. "dense_1", "output_0", etc.)
                self.out_key = list(infer_fn.structured_outputs.keys())[0]

            def predict(self, dataset, verbose=1):
                in_key = list(infer.structured_input_signature[1].keys())[0]
                print(f"  Using input key: '{in_key}', output key: '{self.out_key}'")
                all_probs = []
                for batch in dataset:
                    imgs = batch[0]  # (images, labels) tuple
                    out = self.infer_fn(**{in_key: imgs})
                    logits = out[self.out_key].numpy()
                    # Convert logits → probabilities
                    probs = tf.nn.softmax(logits).numpy()
                    all_probs.append(probs)
                return np.concatenate(all_probs, axis=0)

        return WrappedModel(infer)

    if os.path.isfile(weights_path):
        print(f"\n📦 Loading weights from:\n  {weights_path}")
        print("  (Rebuilding architecture first …)")
        model, _ = build_classifier()
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        model.load_weights(weights_path)
        print("✅ Weights loaded into rebuilt model.")
        return model

    raise FileNotFoundError(
        f"No saved model found.\n"
        f"  Checked SavedModel dir : {savedmodel_path}\n"
        f"  Checked weights file   : {weights_path}\n"
        f"Make sure training completed and the project_dir is correct."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate saved Cassava classifier")
    parser.add_argument("--data_dir", default=None,
                        help="Root folder containing class sub-folders of images.")
    parser.add_argument("--project_dir",
                        default=os.path.join(os.getcwd(), "cassava_fyp_cyclegan"),
                        help="Project directory created by training (default: ./cassava_fyp_cyclegan)")
    parser.add_argument("--weights_only", action="store_true",
                        help="Force loading weights file instead of full SavedModel.")
    parser.add_argument("--use_saved_splits", action="store_true",
                        help="Evaluate using the val/test CSVs saved during training "
                             "(located in <project_dir>). Requires --data_dir images to be "
                             "at the same absolute paths.")
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    args = parser.parse_args()

    cfg.batch_size = args.batch_size
    report_dir = os.path.join(args.project_dir, "reports", "eval_standalone")

    # ── GPU setup ──
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        print(f"✅ GPU detected: {len(gpus)}")
    else:
        print("⚠️  No GPU — running on CPU.")

    # ── Load model ──
    model = load_model(args.project_dir, weights_only=args.weights_only)
    if hasattr(model, "summary"):
        model.summary(line_length=100, print_fn=lambda x: print(x))
    else:
        print("  (Model summary not available for raw TF SavedModel wrapper)")

    # ── Determine which data to evaluate on ──
    if args.use_saved_splits:
        # Evaluate on the original val and test splits used during training
        for split in ("val", "test"):
            # Training saves real_clean_metadata + synthetic; the split info lives
            # in epoch_val_metrics.csv and the original DataFrames are not persisted
            # directly. As a convenience we re-derive from real_clean_metadata.csv.
            meta_csv = os.path.join(args.project_dir, "real_clean_metadata.csv")
            if not os.path.isfile(meta_csv):
                print(f"⚠️  {meta_csv} not found; skipping saved-split evaluation.")
                break

            real_df = pd.read_csv(meta_csv)
            from sklearn.model_selection import train_test_split as tts

            val_size = 0.15
            test_size = 0.15
            train_df, temp_df = tts(real_df, test_size=val_size + test_size,
                                    random_state=cfg.seed, stratify=real_df["label_idx"])
            relative_test = test_size / (val_size + test_size)
            val_df, test_df = tts(temp_df, test_size=relative_test,
                                  random_state=cfg.seed, stratify=temp_df["label_idx"])

            split_df = val_df.reset_index(drop=True) if split == "val" else test_df.reset_index(drop=True)
            # Verify images still exist
            missing = [p for p in split_df["image_path"] if not os.path.isfile(p)]
            if missing:
                print(f"⚠️  {len(missing)} image paths not found on disk (paths may have moved).")

            run_evaluation(model, split_df, report_dir, split_name=f"{split}_real_only")

    elif args.data_dir:
        if not os.path.isdir(args.data_dir):
            raise FileNotFoundError(f"data_dir not found: {args.data_dir}")
        df = load_images_from_dir(args.data_dir)
        run_evaluation(model, df, report_dir, split_name="eval")

    else:
        # Try to find data automatically
        print("\nNo --data_dir provided. Trying to find data automatically …")
        candidates = [
            os.path.join(os.getcwd(), "data"),
            os.path.join(os.getcwd(), "train"),
            os.path.join(os.getcwd(), "..", "data"),
        ]
        found = None
        for c in candidates:
            if os.path.isdir(c):
                imgs = glob.glob(os.path.join(c, "**", "*"), recursive=True)
                if any(p.lower().endswith((".jpg", ".jpeg", ".png")) for p in imgs):
                    found = c
                    break
        if found:
            print(f"  Using: {found}")
            df = load_images_from_dir(found)
            run_evaluation(model, df, report_dir, split_name="eval")
        else:
            print("❌ No data found. Please pass --data_dir or --use_saved_splits.")


if __name__ == "__main__":
    main()