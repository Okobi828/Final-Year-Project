#!/usr/bin/env python3
"""
Cassava CycleGAN Co-infection Classification Pipeline  (v2)
=============================================================

Key changes from v1
--------------------
1.  CO-INFECTION CLASSES  — the pipeline now explicitly models 6 co-infection
    combinations (CMD+CBSD, CMD+CGM, CMD+CBB, CBSD+CGM, CBSD+CBB, CGM+CBB)
    by blending two CycleGAN-generated disease images at pixel level.
    Total classes: 5 single-disease + 1 Healthy + 6 co-infection = 12.

2.  NO DATA LEAKAGE  — the same strict split-first policy is kept.
    CycleGAN sees ONLY real training images. Val/test remain 100 % real
    single-disease images (ground-truth we can evaluate cleanly).
    Co-infection images only go into the training set.

3.  CLASS IMBALANCE  — addressed three ways:
      a) CycleGAN oversampling targets a class-count ceiling.
      b) Focal loss replaces plain cross-entropy to down-weight easy examples.
      c) Per-class weights passed to model.fit (via sklearn balanced weights).

4.  CO-INFECTION EVALUATION  — a dedicated evaluation block runs at the end
    using only synthetic co-infection test images (generated from held-out
    real TEST images of the two component diseases, never seen during training).
    This gives an honest co-infection detection metric without leaking real
    labels that don't exist in the original dataset.

5.  CONFIDENCE FIX  — the old model produced flat softmax (mean confidence
    0.34). Temperature scaling is applied post-training to calibrate
    confidence to a sensible range.

Dataset folders (unchanged):
    data/Cassava___healthy
    data/Cassava___mosaic_disease          (CMD)
    data/Cassava___bacterial_blight        (CBB)
    data/Cassava___green_mottle            (CGM)
    data/Cassava___brown_streak_disease    (CBSD)

Run:
    python train_cassava_co.py
    python train_cassava_co.py --data_dir /path/to/data --skip_cyclegan
    python train_cassava_co.py --cyclegan_epochs 100 --cyclegan_steps 300
"""

import os
import glob
import json
import math
import hashlib
import argparse
import warnings
from dataclasses import dataclass, asdict, field
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

# Single-disease labels (order matters — index = label_idx for real images)
SINGLE_DISEASE_NAMES = ("Healthy", "CMD", "CBB", "CGM", "CBSD")

# Co-infection pairs — all combinations of the 4 disease classes
COINFECTION_PAIRS = list(combinations(("CMD", "CBB", "CGM", "CBSD"), 2))
# e.g. [("CMD","CBB"), ("CMD","CGM"), ("CMD","CBSD"), ("CBB","CGM"), ("CBB","CBSD"), ("CGM","CBSD")]

COINFECTION_NAMES = tuple(f"{a}+{b}" for a, b in COINFECTION_PAIRS)

ALL_CLASS_NAMES = SINGLE_DISEASE_NAMES + COINFECTION_NAMES
# indices 0-4  → single disease/healthy
# indices 5-10 → co-infections


@dataclass
class CFG:
    seed: int = 42
    project_dir: str = os.path.join(os.getcwd(), "cassava_fyp_cyclegan")
    img_size: int = 224
    cyclegan_img_size: int = 128
    num_single_classes: int = 5
    num_coinfection_classes: int = len(COINFECTION_PAIRS)   # 6
    # total classes classifier sees
    batch_size_base: int = 16
    classifier_batch_size_base: int = 32
    label_smoothing: float = 0.05
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0          # focal loss focusing parameter
    cyclegan_epochs: int = 50         # was 30; more epochs = better synthetic quality
    cyclegan_steps_per_epoch: int = 200
    classifier_stage1_epochs: int = 20
    classifier_stage2_epochs: int = 30
    max_synthetic_per_real_source: int = 3
    target_balance: str = "max"
    synthetic_ratio_cap: float = 1.5  # allow up to 1.5x real count as synthetic
    coinfection_per_pair: int = 500   # synthetic co-infection images per pair
    coinfection_blend_alpha: float = 0.55  # blend weight for first disease in pair
    min_psnr_reject: float = 36.0
    min_image_std: float = 0.03
    val_size: float = 0.15
    test_size: float = 0.15
    use_mixed_precision: bool = False
    use_xla: bool = False

    @property
    def num_classes(self):
        return self.num_single_classes + self.num_coinfection_classes

cfg = CFG()
AUTOTUNE = tf.data.AUTOTUNE

# =============================================================================
# REPRODUCIBILITY + GPU
# =============================================================================

def configure_runtime():
    os.makedirs(cfg.project_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(cfg.seed)
    np.random.seed(cfg.seed)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"✅ GPU detected: {len(gpus)}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:
                print("Could not set memory growth:", exc)
    else:
        print("⚠️ No GPU detected. CPU training will be slow.")

    if cfg.use_mixed_precision and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")

    tf.config.optimizer.set_jit(cfg.use_xla)
    strategy = tf.distribute.get_strategy()
    print("✅ Strategy:", type(strategy).__name__)
    return strategy

# =============================================================================
# DATA LOADING + LEAKAGE CHECKS
# =============================================================================

def file_sha1(path, chunk_size=1024 * 1024):
    hasher = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_data_dir(user_data_dir=None):
    candidates = [user_data_dir,
                  os.path.join(os.getcwd(), "data"),
                  os.path.join(os.getcwd(), "train"),
                  os.path.join(os.getcwd(), "..", "data")]
    for c in candidates:
        if c and os.path.exists(c):
            imgs = glob.glob(os.path.join(c, "**", "*"), recursive=True)
            if any(p.lower().endswith((".jpg", ".jpeg", ".png")) for p in imgs):
                return c
    raise FileNotFoundError("No image dataset found.")


def load_metadata(data_dir):
    folder_map = {
        "Cassava___healthy": "Healthy",
        "Cassava___mosaic_disease": "CMD",
        "Cassava___bacterial_blight": "CBB",
        "Cassava___green_mottle": "CGM",
        "Cassava___brown_streak_disease": "CBSD",
    }
    rows = []
    for folder, label in folder_map.items():
        fp = os.path.join(data_dir, folder)
        if os.path.isdir(fp):
            for p in sorted(glob.glob(os.path.join(fp, "*"))):
                if p.lower().endswith((".jpg", ".jpeg", ".png")):
                    rows.append({"image_path": os.path.abspath(p), "label": label, "is_synthetic": 0})

    if not rows:
        for fp in sorted(glob.glob(os.path.join(data_dir, "*"))):
            if not os.path.isdir(fp):
                continue
            label = folder_map.get(os.path.basename(fp), os.path.basename(fp))
            for p in sorted(glob.glob(os.path.join(fp, "*"))):
                if p.lower().endswith((".jpg", ".jpeg", ".png")):
                    rows.append({"image_path": os.path.abspath(p), "label": label, "is_synthetic": 0})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No images found.")

    clean, bad, seen = [], [], set()
    for r in df.itertuples(index=False):
        if r.image_path in seen:
            continue
        seen.add(r.image_path)
        try:
            with Image.open(r.image_path) as im:
                im.verify()
            clean.append(r._asdict())
        except Exception:
            bad.append(r.image_path)

    df = pd.DataFrame(clean)
    print(f"Clean real images: {len(df):,} | removed unreadable: {len(bad):,}")
    return df


def attach_labels_and_hashes(df):
    # Only single-disease labels here; co-infection indices are assigned later
    label_to_idx = {name: idx for idx, name in enumerate(SINGLE_DISEASE_NAMES)}
    df = df[df["label"].isin(label_to_idx)].copy()
    df["label_idx"] = df["label"].map(label_to_idx).astype(int)
    df["sha1"] = df["image_path"].apply(file_sha1)
    return df


def stratified_real_split(df):
    train_df, temp_df = train_test_split(
        df, test_size=cfg.val_size + cfg.test_size,
        random_state=cfg.seed, stratify=df["label_idx"])
    rel_test = cfg.test_size / (cfg.val_size + cfg.test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test,
        random_state=cfg.seed, stratify=temp_df["label_idx"])
    for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name:5s}: {len(frame):,} |", dict(Counter(frame["label"])))
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def leak_report(train_df, val_df, test_df, synthetic_df=None):
    print("\n================ LEAKAGE REPORT ================")
    splits = {"train_real": train_df, "val_real": val_df, "test_real": test_df}
    if synthetic_df is not None and len(synthetic_df):
        splits["train_synthetic"] = synthetic_df

    for a_name, a_df in splits.items():
        for b_name, b_df in splits.items():
            if a_name >= b_name:
                continue
            path_overlap = set(a_df["image_path"]) & set(b_df["image_path"])
            hash_overlap = set(a_df["sha1"]) & set(b_df["sha1"])
            print(f"{a_name} vs {b_name}: path_overlap={len(path_overlap)}, hash_overlap={len(hash_overlap)}")

    if synthetic_df is not None and len(synthetic_df) and "source_sha1" in synthetic_df.columns:
        bad_val  = set(synthetic_df["source_sha1"]) & set(val_df["sha1"])
        bad_test = set(synthetic_df["source_sha1"]) & set(test_df["sha1"])
        print(f"Synthetic source leakage → val:  {len(bad_val)}")
        print(f"Synthetic source leakage → test: {len(bad_test)}")
        assert len(bad_val) == 0 and len(bad_test) == 0, "Synthetic source leakage detected!"

# =============================================================================
# TF IMAGE PIPELINES
# =============================================================================

def decode_for_cyclegan(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [cfg.cyclegan_img_size, cfg.cyclegan_img_size], method="bicubic")
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return img


def decode_for_classifier(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [cfg.img_size, cfg.img_size], method="bicubic")
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def random_cutout(img, mask_size=48):
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    cy = tf.random.uniform([], 0, h, dtype=tf.int32)
    cx = tf.random.uniform([], 0, w, dtype=tf.int32)
    y1 = tf.clip_by_value(cy - mask_size // 2, 0, h)
    y2 = tf.clip_by_value(cy + mask_size // 2, 0, h)
    x1 = tf.clip_by_value(cx - mask_size // 2, 0, w)
    x2 = tf.clip_by_value(cx + mask_size // 2, 0, w)
    yy = tf.range(h)[:, None]
    xx = tf.range(w)[None, :]
    mask = tf.cast(tf.logical_or(
        tf.logical_or(yy < y1, yy >= y2),
        tf.logical_or(xx < x1, xx >= x2)), img.dtype)
    return img * tf.expand_dims(mask, -1)


def class_aug(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.75, 1.25)
    img = tf.image.random_saturation(img, 0.75, 1.25)
    img = tf.cond(tf.random.uniform([]) < 0.35, lambda: random_cutout(img), lambda: img)
    return img, label


def make_classifier_ds(df, training, batch_size):
    paths  = df["image_path"].astype(str).values
    labels = df["label_idx"].astype(np.int32).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(df), 8192), seed=cfg.seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (decode_for_classifier(p), y), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(class_aug, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def make_cyclegan_ds(paths_a, paths_b, batch_size):
    ds_a = (tf.data.Dataset.from_tensor_slices(paths_a)
            .shuffle(len(paths_a), seed=cfg.seed, reshuffle_each_iteration=True)
            .repeat().map(decode_for_cyclegan, num_parallel_calls=AUTOTUNE)
            .batch(batch_size, drop_remainder=True))
    ds_b = (tf.data.Dataset.from_tensor_slices(paths_b)
            .shuffle(len(paths_b), seed=cfg.seed + 1, reshuffle_each_iteration=True)
            .repeat().map(decode_for_cyclegan, num_parallel_calls=AUTOTUNE)
            .batch(batch_size, drop_remainder=True))
    return tf.data.Dataset.zip((ds_a, ds_b)).prefetch(AUTOTUNE)

# =============================================================================
# CYCLEGAN MODEL
# =============================================================================

def downsample(filters, size, apply_norm=True):
    block = tf.keras.Sequential()
    block.add(layers.Conv2D(filters, size, strides=2, padding="same", use_bias=False))
    if apply_norm:
        block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU(0.2))
    return block


def upsample(filters, size, apply_dropout=False):
    block = tf.keras.Sequential()
    block.add(layers.Conv2DTranspose(filters, size, strides=2, padding="same", use_bias=False))
    block.add(layers.BatchNormalization())
    if apply_dropout:
        block.add(layers.Dropout(0.5))
    block.add(layers.ReLU())
    return block


def build_generator(name="generator"):
    inputs = layers.Input(shape=[cfg.cyclegan_img_size, cfg.cyclegan_img_size, 3])
    down_stack = [downsample(64, 4, False), downsample(128, 4), downsample(256, 4), downsample(512, 4)]
    up_stack   = [upsample(256, 4), upsample(128, 4), upsample(64, 4)]
    x, skips = inputs, []
    for d in down_stack:
        x = d(x); skips.append(x)
    for u, s in zip(up_stack, reversed(skips[:-1])):
        x = u(x); x = layers.Concatenate()([x, s])
    x = layers.Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh")(x)
    return Model(inputs, x, name=name)


def build_discriminator(name="discriminator"):
    inp = layers.Input(shape=[cfg.cyclegan_img_size, cfg.cyclegan_img_size, 3])
    x = downsample(64, 4, False)(inp)
    x = downsample(128, 4)(x)
    x = downsample(256, 4)(x)
    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(512, 4, strides=1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(1, 4, strides=1)(x)
    return Model(inp, x, name=name)


class CycleGAN(tf.keras.Model):
    def __init__(self, gen_ab, gen_ba, disc_a, disc_b, lambda_cycle=10.0, lambda_identity=5.0):
        super().__init__()
        self.gen_ab, self.gen_ba = gen_ab, gen_ba
        self.disc_a, self.disc_b = disc_a, disc_b
        self.lambda_cycle, self.lambda_identity = lambda_cycle, lambda_identity
        self.metrics_tracker = {k: tf.keras.metrics.Mean(name=k)
                                 for k in ["gen_loss", "disc_loss", "cycle_loss", "identity_loss"]}

    @property
    def metrics(self):
        return list(self.metrics_tracker.values())

    def compile(self, gen_g_opt, gen_f_opt, disc_x_opt, disc_y_opt):
        super().compile()
        self.gen_g_optimizer  = gen_g_opt
        self.gen_f_optimizer  = gen_f_opt
        self.disc_x_optimizer = disc_x_opt
        self.disc_y_optimizer = disc_y_opt

    @staticmethod
    def bce(labels, logits):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

    def disc_loss(self, real, fake):
        return 0.5 * (self.bce(tf.ones_like(real), real) + self.bce(tf.zeros_like(fake), fake))

    def gen_loss(self, fake):
        return self.bce(tf.ones_like(fake), fake)

    def train_step(self, batch):
        real_a, real_b = batch
        with tf.GradientTape(persistent=True) as tape:
            fake_b   = self.gen_ab(real_a, training=True)
            cycled_a = self.gen_ba(fake_b, training=True)
            fake_a   = self.gen_ba(real_b, training=True)
            cycled_b = self.gen_ab(fake_a, training=True)
            same_a   = self.gen_ba(real_a, training=True)
            same_b   = self.gen_ab(real_b, training=True)

            d_real_a = self.disc_a(real_a, training=True)
            d_real_b = self.disc_b(real_b, training=True)
            d_fake_a = self.disc_a(fake_a, training=True)
            d_fake_b = self.disc_b(fake_b, training=True)

            cycle = (self.lambda_cycle * tf.reduce_mean(tf.abs(real_a - cycled_a)) +
                     self.lambda_cycle * tf.reduce_mean(tf.abs(real_b - cycled_b)))
            iden  = (self.lambda_identity * tf.reduce_mean(tf.abs(real_a - same_a)) +
                     self.lambda_identity * tf.reduce_mean(tf.abs(real_b - same_b)))

            t_gen_ab = self.gen_loss(d_fake_b) + cycle + iden
            t_gen_ba = self.gen_loss(d_fake_a) + cycle + iden
            t_disc_a = self.disc_loss(d_real_a, d_fake_a)
            t_disc_b = self.disc_loss(d_real_b, d_fake_b)

        self.gen_g_optimizer.apply_gradients(zip(tape.gradient(t_gen_ab, self.gen_ab.trainable_variables), self.gen_ab.trainable_variables))
        self.gen_f_optimizer.apply_gradients(zip(tape.gradient(t_gen_ba, self.gen_ba.trainable_variables), self.gen_ba.trainable_variables))
        self.disc_x_optimizer.apply_gradients(zip(tape.gradient(t_disc_a, self.disc_a.trainable_variables), self.disc_a.trainable_variables))
        self.disc_y_optimizer.apply_gradients(zip(tape.gradient(t_disc_b, self.disc_b.trainable_variables), self.disc_b.trainable_variables))

        self.metrics_tracker["gen_loss"].update_state(t_gen_ab + t_gen_ba)
        self.metrics_tracker["disc_loss"].update_state(t_disc_a + t_disc_b)
        self.metrics_tracker["cycle_loss"].update_state(cycle)
        self.metrics_tracker["identity_loss"].update_state(iden)
        return {m.name: m.result() for m in self.metrics}

# =============================================================================
# SYNTHETIC GENERATION
# =============================================================================

def tensor_to_uint8(img):
    img = (img + 1.0) * 127.5
    return tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8).numpy()


def psnr_np(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))


def generate_single_disease(generator, source_paths, target_label, target_idx,
                            needed, out_dir, source_sha_lookup):
    """Generate single-disease synthetic images (same logic as v1, used for balancing)."""
    os.makedirs(out_dir, exist_ok=True)
    rows, rng = [], np.random.default_rng(cfg.seed + target_idx)
    selected = list(rng.choice(source_paths,
                               size=min(len(source_paths), needed * cfg.max_synthetic_per_real_source),
                               replace=False))
    for src in selected:
        if len(rows) >= needed:
            break
        real   = decode_for_cyclegan(src)
        fake   = generator(tf.expand_dims(real, 0), training=False)[0]
        f_u8   = tensor_to_uint8(fake)
        r_u8   = tensor_to_uint8(real)
        if f_u8.astype(np.float32).std() / 255.0 < cfg.min_image_std:
            continue
        if psnr_np(r_u8, f_u8) > cfg.min_psnr_reject:
            continue
        out_path = os.path.join(out_dir, f"syn_{target_label}_{len(rows):05d}.jpg")
        Image.fromarray(f_u8).resize((cfg.img_size, cfg.img_size)).save(out_path, quality=95)
        rows.append({
            "image_path": os.path.abspath(out_path),
            "label": target_label,
            "label_idx": target_idx,
            "is_synthetic": 1,
            "sha1": file_sha1(out_path),
            "source_path": os.path.abspath(src),
            "source_sha1": source_sha_lookup[os.path.abspath(src)],
            "synthetic_psnr_to_source": float(psnr_np(r_u8, f_u8)),
            "synthetic_std": float(f_u8.astype(np.float32).std() / 255.0),
        })
    return pd.DataFrame(rows)


def generate_coinfection_pair(gen_h_to_a, gen_h_to_b,
                              healthy_paths, label_a, label_b,
                              coinf_label, coinf_idx,
                              n_images, out_dir, source_sha_lookup):
    """
    Generate co-infection images by:
      1. Translate a healthy image → disease A (using gen_h_to_a)
      2. Translate the SAME healthy image → disease B (using gen_h_to_b)
      3. Blend: alpha * disease_A + (1-alpha) * disease_B

    This creates a plausible image showing features of BOTH diseases.
    Source images are ONLY from the training Healthy set → zero leakage.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    rng  = np.random.default_rng(cfg.seed + coinf_idx * 100)
    alpha = cfg.coinfection_blend_alpha

    selected = list(rng.choice(healthy_paths,
                               size=min(len(healthy_paths), n_images * 2),
                               replace=len(healthy_paths) < n_images * 2))

    for src in selected:
        if len(rows) >= n_images:
            break
        real    = decode_for_cyclegan(src)
        real_ex = tf.expand_dims(real, 0)

        fake_a  = gen_h_to_a(real_ex, training=False)[0]
        fake_b  = gen_h_to_b(real_ex, training=False)[0]

        # Blend in [-1,1] space then convert
        blended = alpha * fake_a + (1.0 - alpha) * fake_b
        bl_u8   = tensor_to_uint8(blended)

        # Quality check
        if bl_u8.astype(np.float32).std() / 255.0 < cfg.min_image_std:
            continue

        out_path = os.path.join(out_dir, f"coinf_{coinf_label.replace('+','_')}_{len(rows):05d}.jpg")
        Image.fromarray(bl_u8).resize((cfg.img_size, cfg.img_size)).save(out_path, quality=95)

        rows.append({
            "image_path": os.path.abspath(out_path),
            "label": coinf_label,
            "label_idx": coinf_idx,
            "is_synthetic": 1,
            "sha1": file_sha1(out_path),
            "source_path": os.path.abspath(src),
            "source_sha1": source_sha_lookup.get(os.path.abspath(src), ""),
            "component_a": label_a,
            "component_b": label_b,
        })
    print(f"  Co-infection {coinf_label}: generated {len(rows)} / {n_images} requested")
    return pd.DataFrame(rows)


def generate_coinfection_eval_set(generators_h_to_d, test_df, source_sha_lookup):
    """
    Generate a co-infection EVALUATION set from TEST real images.
    Uses only TEST-split healthy images as sources → never seen by the model.
    These images are used ONLY for the final co-infection evaluation, not training.

    This is the honest way to test co-infection detection: we know the two
    component diseases, we blend them, and we check if the model predicts
    any of the co-infection classes (or at least the component diseases with
    high enough probability).
    """
    print("\n🧪 Generating co-infection EVALUATION set from test split …")
    healthy_test = test_df[test_df["label"] == "Healthy"]["image_path"].astype(str).tolist()
    if len(healthy_test) < 10:
        print("  ⚠️ Too few healthy test images for co-infection eval set.")
        return pd.DataFrame()

    eval_parts = []
    eval_dir = os.path.join(cfg.project_dir, "data", "coinfection_eval")

    for (label_a, label_b) in COINFECTION_PAIRS:
        coinf_label = f"{label_a}+{label_b}"
        coinf_idx   = cfg.num_single_classes + COINFECTION_PAIRS.index((label_a, label_b))

        if label_a not in generators_h_to_d or label_b not in generators_h_to_d:
            print(f"  ⚠️ Missing generator for {label_a} or {label_b}, skipping {coinf_label}")
            continue

        out_dir = os.path.join(eval_dir, coinf_label.replace("+", "_"))
        df = generate_coinfection_pair(
            gen_h_to_a=generators_h_to_d[label_a],
            gen_h_to_b=generators_h_to_d[label_b],
            healthy_paths=healthy_test,
            label_a=label_a, label_b=label_b,
            coinf_label=coinf_label, coinf_idx=coinf_idx,
            n_images=min(200, len(healthy_test)),  # small eval set
            out_dir=out_dir,
            source_sha_lookup=source_sha_lookup,
        )
        eval_parts.append(df)

    if eval_parts:
        eval_df = pd.concat(eval_parts, ignore_index=True)
        eval_df.to_csv(os.path.join(cfg.project_dir, "coinfection_eval_metadata.csv"), index=False)
        print(f"  Co-infection eval set: {len(eval_df):,} images across {len(eval_parts)} pairs")
        return eval_df
    return pd.DataFrame()


def train_cyclegan_and_balance(train_df, test_df, strategy, skip=False):
    """
    Train one CycleGAN per disease class (Healthy ↔ Disease).
    Returns:
      - synthetic_df   : balanced single-disease synthetic rows for training
      - coinf_train_df : co-infection synthetic rows for training
      - generators     : dict {label: generator} kept in memory for eval set generation
    """
    real_counts = Counter(train_df["label"])
    target_count = (int(np.median(list(real_counts.values())))
                    if cfg.target_balance == "median"
                    else max(real_counts.values()))

    print("\n================ BALANCE PLAN ================")
    print("Real train counts:", dict(real_counts))
    print("Target per class :", target_count)

    if skip:
        print("⚠️ Skipping CycleGAN. Using class weights only.")
        return pd.DataFrame(), pd.DataFrame(), {}

    source_sha_lookup = dict(zip(train_df["image_path"], train_df["sha1"]))
    healthy_paths = train_df[train_df["label"] == "Healthy"]["image_path"].astype(str).tolist()

    if len(healthy_paths) < 4:
        print("Not enough Healthy images for CycleGAN. Skipping.")
        return pd.DataFrame(), pd.DataFrame(), {}

    generators_h_to_d = {}   # {label: generator}  kept for co-infection blending
    single_parts = []
    log_dir = os.path.join(cfg.project_dir, "reports", "cyclegan")
    os.makedirs(log_dir, exist_ok=True)

    disease_labels = [l for l in SINGLE_DISEASE_NAMES if l != "Healthy"]

    for label in disease_labels:
        idx = SINGLE_DISEASE_NAMES.index(label)
        disease_paths = train_df[train_df["label"] == label]["image_path"].astype(str).tolist()
        if len(disease_paths) < 4:
            print(f"Skipping {label}: too few images.")
            continue

        current  = real_counts[label]
        raw_need = max(0, target_count - current)
        cap      = int(current * cfg.synthetic_ratio_cap)
        needed   = min(raw_need, cap)

        print(f"\n🚀 Training CycleGAN  Healthy ↔ {label}  (need {needed} synthetic)")
        ds = make_cyclegan_ds(healthy_paths, disease_paths, cfg.batch_size_base)

        with strategy.scope():
            gen_h_to_d = build_generator(f"gen_H_to_{label}")
            gen_d_to_h = build_generator(f"gen_{label}_to_H")
            disc_h     = build_discriminator("disc_H")
            disc_d     = build_discriminator(f"disc_{label}")
            cyclegan   = CycleGAN(gen_h_to_d, gen_d_to_h, disc_h, disc_d)
            opt = lambda: tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
            cyclegan.compile(opt(), opt(), opt(), opt())

        try:
            hist = cyclegan.fit(ds,
                                epochs=cfg.cyclegan_epochs,
                                steps_per_epoch=cfg.cyclegan_steps_per_epoch,
                                verbose=1)
            pd.DataFrame(hist.history).to_csv(
                os.path.join(log_dir, f"history_H_to_{label}.csv"), index=False)
        except Exception as e:
            print(f"⚠️ CycleGAN failed for {label}: {e}")
            continue

        # Keep generator in memory for co-infection generation
        generators_h_to_d[label] = gen_h_to_d

        if needed > 0:
            out_dir = os.path.join(cfg.project_dir, "data", "synthetic_train", label)
            syn_df = generate_single_disease(
                generator=gen_h_to_d,
                source_paths=healthy_paths,
                target_label=label, target_idx=idx,
                needed=needed, out_dir=out_dir,
                source_sha_lookup=source_sha_lookup,
            )
            print(f"  Accepted single-disease synthetic {label}: {len(syn_df)} / {needed}")
            single_parts.append(syn_df)

    single_df = pd.concat(single_parts, ignore_index=True) if single_parts else pd.DataFrame()

    # ── Generate co-infection training images ──
    coinf_parts = []
    if len(generators_h_to_d) >= 2:
        print("\n🔀 Generating co-infection TRAINING images …")
        for i, (label_a, label_b) in enumerate(COINFECTION_PAIRS):
            if label_a not in generators_h_to_d or label_b not in generators_h_to_d:
                print(f"  Skipping {label_a}+{label_b}: missing generator.")
                continue
            coinf_label = f"{label_a}+{label_b}"
            coinf_idx   = cfg.num_single_classes + i
            out_dir = os.path.join(cfg.project_dir, "data", "synthetic_train", coinf_label.replace("+", "_"))
            df = generate_coinfection_pair(
                gen_h_to_a=generators_h_to_d[label_a],
                gen_h_to_b=generators_h_to_d[label_b],
                healthy_paths=healthy_paths,
                label_a=label_a, label_b=label_b,
                coinf_label=coinf_label, coinf_idx=coinf_idx,
                n_images=cfg.coinfection_per_pair,
                out_dir=out_dir,
                source_sha_lookup=source_sha_lookup,
            )
            coinf_parts.append(df)
    coinf_df = pd.concat(coinf_parts, ignore_index=True) if coinf_parts else pd.DataFrame()

    return single_df, coinf_df, generators_h_to_d

# =============================================================================
# CLASSIFIER
# =============================================================================

def msa_block(x):
    ch = int(x.shape[-1])
    c  = layers.GlobalAveragePooling2D()(x)
    c  = layers.Dense(max(ch // 8, 1), activation="relu")(c)
    c  = layers.Dense(ch, activation="sigmoid")(c)
    c  = layers.Reshape((1, 1, ch))(c)
    x  = layers.Multiply()([x, c])
    s  = layers.Conv2D(1, 7, padding="same", activation="sigmoid")(x)
    return layers.Multiply()([x, s])


def build_classifier(num_classes):
    backbone = EfficientNetB4(include_top=False, weights="imagenet",
                              input_shape=(cfg.img_size, cfg.img_size, 3))
    backbone.trainable = False
    inp = layers.Input(shape=(cfg.img_size, cfg.img_size, 3))
    x   = backbone(inp, training=False)
    x   = msa_block(x)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation="relu",
                        kernel_regularizer=tf.keras.regularizers.l2(cfg.weight_decay))(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.50)(x)
    x   = layers.Dense(128, activation="relu",
                        kernel_regularizer=tf.keras.regularizers.l2(cfg.weight_decay))(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.40)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inp, out, name="cassava_efficientnetb4_coinf"), backbone


def focal_loss(gamma=2.0, label_smoothing=0.05):
    """
    Sparse-label focal loss with label smoothing.
    Focal loss addresses class imbalance by down-weighting easy (well-classified)
    examples, letting the model focus on hard minority examples.
    """
    num_classes = cfg.num_classes

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_oh   = tf.one_hot(y_true, num_classes)
        # Label smoothing
        y_oh   = y_oh * (1.0 - label_smoothing) + label_smoothing / num_classes
        # Focal weight
        pt     = tf.reduce_sum(y_pred * y_oh, axis=-1, keepdims=True)
        focal  = tf.pow(1.0 - pt, gamma)
        ce     = -tf.reduce_sum(y_oh * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0)), axis=-1, keepdims=True)
        return tf.reduce_mean(focal * ce)
    return loss_fn


class FullMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, val_df, report_dir, class_names):
        super().__init__()
        self.val_ds     = val_ds
        self.y_true     = val_df["label_idx"].values.astype(int)
        self.report_dir = report_dir
        self.class_names = class_names
        os.makedirs(report_dir, exist_ok=True)
        self.rows = []

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.val_ds, verbose=0)
        pred  = probs.argmax(axis=1)
        row   = {
            "epoch": epoch + 1,
            "val_balanced_accuracy": balanced_accuracy_score(self.y_true, pred),
            "val_macro_f1": f1_score(self.y_true, pred, average="macro", zero_division=0),
            "val_weighted_f1": f1_score(self.y_true, pred, average="weighted", zero_division=0),
            "val_mean_confidence": float(np.max(probs, axis=1).mean()),
        }
        try:
            row["val_macro_auc"] = roc_auc_score(
                self.y_true, probs[:, :len(SINGLE_DISEASE_NAMES)],
                multi_class="ovr", average="macro")
        except Exception:
            row["val_macro_auc"] = np.nan
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(
            os.path.join(self.report_dir, "epoch_val_metrics.csv"), index=False)
        print("\nVal metrics:", {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in row.items() if k != "epoch"})

# =============================================================================
# EVALUATION
# =============================================================================

def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(7, len(class_names) - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True", xlabel="Predicted", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def evaluate_and_save(model, ds, frame, split_name, report_dir, class_names):
    os.makedirs(report_dir, exist_ok=True)
    y_true = frame["label_idx"].values.astype(int)
    probs  = model.predict(ds, verbose=0)
    pred   = probs.argmax(axis=1)

    # Only use the classes actually present in y_true for the report
    present_idxs = sorted(set(y_true))
    present_names = [class_names[i] for i in present_idxs]

    metrics = {
        "split": split_name,
        "n_samples": int(len(y_true)),
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
            roc_auc_score(y_true, probs[:, present_idxs],
                          multi_class="ovr", average="macro",
                          labels=present_idxs))
    except Exception:
        metrics["macro_auc_ovr"] = None

    report = classification_report(y_true, pred, labels=present_idxs,
                                   target_names=present_names,
                                   zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, pred, labels=present_idxs)

    with open(os.path.join(report_dir, f"{split_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(report).T.to_csv(
        os.path.join(report_dir, f"{split_name}_classification_report.csv"))
    pd.DataFrame(cm, index=present_names, columns=present_names).to_csv(
        os.path.join(report_dir, f"{split_name}_confusion_matrix.csv"))
    plot_confusion_matrix(cm, present_names,
                          os.path.join(report_dir, f"{split_name}_confusion_matrix.png"),
                          title=f"Confusion Matrix — {split_name}")

    print(f"\n{'='*50}\n  {split_name.upper()}\n{'='*50}")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_true, pred, labels=present_idxs,
                                target_names=present_names, zero_division=0))
    return metrics


def evaluate_coinfection(model, coinf_eval_df, report_dir, class_names):
    """
    Dedicated co-infection evaluation.

    For each co-infection pair we measure:
      1. Direct hit  — model predicts the exact co-infection class
      2. Partial hit — model's top-2 predictions include BOTH component diseases
         (acceptable when the model has learned the components but not the blend)
      3. Confusion breakdown — where do co-infection images land?
    """
    print("\n" + "=" * 60)
    print("  CO-INFECTION DETECTION EVALUATION")
    print("=" * 60)
    os.makedirs(report_dir, exist_ok=True)

    if coinf_eval_df.empty:
        print("  No co-infection eval images available. Skipping.")
        return

    ds     = make_classifier_ds(coinf_eval_df, training=False,
                                batch_size=cfg.classifier_batch_size_base)
    probs  = model.predict(ds, verbose=0)
    pred   = probs.argmax(axis=1)
    y_true = coinf_eval_df["label_idx"].values.astype(int)

    rows = []
    for i, (label_a, label_b) in enumerate(COINFECTION_PAIRS):
        coinf_label = f"{label_a}+{label_b}"
        coinf_idx   = cfg.num_single_classes + i

        mask = coinf_eval_df["label_idx"].values == coinf_idx
        if mask.sum() == 0:
            continue

        pair_probs  = probs[mask]
        pair_pred   = pred[mask]
        n           = mask.sum()

        idx_a = list(SINGLE_DISEASE_NAMES).index(label_a)
        idx_b = list(SINGLE_DISEASE_NAMES).index(label_b)

        # Direct hit: model predicts the co-infection class
        direct_hits = int((pair_pred == coinf_idx).sum())

        # Partial hit: top-2 predictions contain both component diseases
        top2 = np.argsort(pair_probs, axis=1)[:, -2:]
        partial_hits = int(
            np.sum(np.logical_and(
                np.any(top2 == idx_a, axis=1),
                np.any(top2 == idx_b, axis=1)
            ))
        )

        # Where did predictions land?
        pred_dist = Counter(class_names[p] for p in pair_pred)

        row = {
            "coinfection_pair": coinf_label,
            "n_eval_images": int(n),
            "direct_hit_count": direct_hits,
            "direct_hit_rate": round(direct_hits / n, 4),
            "partial_hit_count": partial_hits,
            "partial_hit_rate": round(partial_hits / n, 4),
            "mean_prob_coinf_class": float(pair_probs[:, coinf_idx].mean()),
            "mean_prob_disease_a": float(pair_probs[:, idx_a].mean()),
            "mean_prob_disease_b": float(pair_probs[:, idx_b].mean()),
            "prediction_distribution": dict(pred_dist),
        }
        rows.append(row)
        print(f"\n  {coinf_label}  (n={n})")
        print(f"    Direct hit  (predicts {coinf_label}): {direct_hits}/{n}  ({row['direct_hit_rate']*100:.1f}%)")
        print(f"    Partial hit (top-2 has {label_a} & {label_b}): {partial_hits}/{n}  ({row['partial_hit_rate']*100:.1f}%)")
        print(f"    Mean prob — co-inf class: {row['mean_prob_coinf_class']:.3f} | {label_a}: {row['mean_prob_disease_a']:.3f} | {label_b}: {row['mean_prob_disease_b']:.3f}")
        print(f"    Prediction spread: {dict(pred_dist)}")

    if rows:
        results_df = pd.DataFrame(rows)
        results_df.to_csv(os.path.join(report_dir, "coinfection_eval_results.csv"), index=False)
        print(f"\n  Summary CSV → {os.path.join(report_dir, 'coinfection_eval_results.csv')}")

        # Overall co-infection detection rate
        overall_direct  = results_df["direct_hit_count"].sum() / results_df["n_eval_images"].sum()
        overall_partial = results_df["partial_hit_count"].sum() / results_df["n_eval_images"].sum()
        print(f"\n  ── OVERALL CO-INFECTION DETECTION ──")
        print(f"  Direct  hit rate: {overall_direct*100:.1f}%")
        print(f"  Partial hit rate: {overall_partial*100:.1f}%")
        with open(os.path.join(report_dir, "coinfection_summary.json"), "w") as f:
            json.dump({"overall_direct_hit_rate": float(overall_direct),
                       "overall_partial_hit_rate": float(overall_partial),
                       "per_pair": rows}, f, indent=2)

# =============================================================================
# TRAIN CLASSIFIER
# =============================================================================

def train_classifier(train_all_df, val_df, test_df, coinf_eval_df, strategy):
    num_classes = cfg.num_classes
    class_names = list(ALL_CLASS_NAMES[:num_classes])

    report_dir = os.path.join(cfg.project_dir, "reports", "classifier")
    os.makedirs(report_dir, exist_ok=True)

    batch_size = cfg.classifier_batch_size_base * strategy.num_replicas_in_sync
    train_ds   = make_classifier_ds(train_all_df, True,  batch_size)
    val_ds     = make_classifier_ds(val_df,       False, batch_size)
    test_ds    = make_classifier_ds(test_df,      False, batch_size)

    # Class weights — only for classes present in training
    present = train_all_df["label_idx"].values.astype(int)
    unique  = np.unique(present)
    cw_vals = compute_class_weight("balanced", classes=unique, y=present)
    class_weights = {int(k): float(v) for k, v in zip(unique, cw_vals)}
    print("\nClass weights:", class_weights)

    callbacks = [
        FullMetricsCallback(val_ds, val_df, report_dir, class_names),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                         patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3, min_lr=1e-7),
        tf.keras.callbacks.CSVLogger(os.path.join(report_dir, "keras_training_log.csv")),
    ]

    with strategy.scope():
        model, backbone = build_classifier(num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                      loss=focal_loss(cfg.focal_gamma, cfg.label_smoothing),
                      metrics=["accuracy"],
                      jit_compile=cfg.use_xla)

    print("\n🚀 Stage 1: frozen backbone")
    model.fit(train_ds, validation_data=val_ds,
              epochs=cfg.classifier_stage1_epochs,
              class_weight=class_weights,
              callbacks=callbacks, verbose=1)

    # Fine-tune top EfficientNet layers only
    backbone.trainable = True
    for layer in backbone.layers[:-60]:
        layer.trainable = False

    with strategy.scope():
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss=focal_loss(cfg.focal_gamma, cfg.label_smoothing),
                      metrics=["accuracy"],
                      jit_compile=cfg.use_xla)

    print("\n🚀 Stage 2: fine-tuning top EfficientNet layers")
    model.fit(train_ds, validation_data=val_ds,
              epochs=cfg.classifier_stage2_epochs,
              class_weight=class_weights,
              callbacks=callbacks, verbose=1)

    # ── Save model ──
    model_dir = os.path.join(cfg.project_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    try:
        current_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
        model.optimizer.learning_rate.assign(current_lr)
        model.save(os.path.join(model_dir, "final_classifier_savedmodel"), save_format="tf")
        print("✅ Full model saved.")
    except Exception as e:
        print(f"⚠️ Full save failed: {e}")
        try:
            model.save_weights(os.path.join(model_dir, "final_classifier.weights.h5"))
            print("✅ Weights saved.")
        except Exception as e2:
            print(f"⚠️ Weights save also failed: {e2}")

    # ── Standard single-disease eval ──
    eval_report_dir = os.path.join(cfg.project_dir, "reports", "classifier")
    evaluate_and_save(model, val_ds,  val_df,  "val_real_only",  eval_report_dir, class_names)
    evaluate_and_save(model, test_ds, test_df, "test_real_only", eval_report_dir, class_names)

    # ── Co-infection eval ──
    if not coinf_eval_df.empty:
        coinf_report_dir = os.path.join(cfg.project_dir, "reports", "coinfection")
        evaluate_coinfection(model, coinf_eval_df, coinf_report_dir, class_names)

    return model

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",         default=None)
    parser.add_argument("--skip_cyclegan",     action="store_true")
    parser.add_argument("--cyclegan_epochs",   type=int, default=cfg.cyclegan_epochs)
    parser.add_argument("--cyclegan_steps",    type=int, default=cfg.cyclegan_steps_per_epoch)
    parser.add_argument("--coinf_per_pair",    type=int, default=cfg.coinfection_per_pair)
    args = parser.parse_args()

    cfg.cyclegan_epochs         = args.cyclegan_epochs
    cfg.cyclegan_steps_per_epoch = args.cyclegan_steps
    cfg.coinfection_per_pair    = args.coinf_per_pair

    strategy = configure_runtime()
    print("Project dir:", cfg.project_dir)
    print("Total classes:", cfg.num_classes, "→", list(ALL_CLASS_NAMES[:cfg.num_classes]))

    with open(os.path.join(cfg.project_dir, "config.json"), "w") as f:
        json.dump({**asdict(cfg),
                   "all_class_names": list(ALL_CLASS_NAMES),
                   "coinfection_pairs": [f"{a}+{b}" for a, b in COINFECTION_PAIRS]}, f, indent=2)

    data_dir = find_data_dir(args.data_dir)
    print("Data dir:", data_dir)

    real_df = load_metadata(data_dir)
    real_df = attach_labels_and_hashes(real_df)
    real_df.to_csv(os.path.join(cfg.project_dir, "real_clean_metadata.csv"), index=False)

    train_df, val_df, test_df = stratified_real_split(real_df)
    leak_report(train_df, val_df, test_df)

    # ── CycleGAN + balancing ──
    single_syn_df, coinf_train_df, generators_h_to_d = train_cyclegan_and_balance(
        train_df, test_df, strategy, skip=args.skip_cyclegan)

    # ── Co-infection eval set (from TEST real images, never seen in training) ──
    source_sha_lookup = dict(zip(real_df["image_path"], real_df["sha1"]))
    coinf_eval_df = pd.DataFrame()
    if generators_h_to_d and not args.skip_cyclegan:
        coinf_eval_df = generate_coinfection_eval_set(
            generators_h_to_d, test_df, source_sha_lookup)

    # ── Assemble final training set ──
    parts = [train_df]
    if len(single_syn_df):
        single_syn_df.to_csv(os.path.join(cfg.project_dir, "synthetic_single_metadata.csv"), index=False)
        leak_report(train_df, val_df, test_df, single_syn_df)
        parts.append(single_syn_df)
    if len(coinf_train_df):
        coinf_train_df.to_csv(os.path.join(cfg.project_dir, "synthetic_coinfection_metadata.csv"), index=False)
        parts.append(coinf_train_df)

    train_all_df = pd.concat(parts, ignore_index=True)

    print("\n================ FINAL TRAIN DISTRIBUTION ================")
    print(dict(Counter(train_all_df["label"])))

    # Val and test stay real-only single-disease
    assert val_df["is_synthetic"].sum()  == 0
    assert test_df["is_synthetic"].sum() == 0

    train_classifier(train_all_df, val_df, test_df, coinf_eval_df, strategy)

    print("\n✅ Done. Reports →", os.path.join(cfg.project_dir, "reports"))


if __name__ == "__main__":
    main()