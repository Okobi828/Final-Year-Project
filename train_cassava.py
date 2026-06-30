#!/usr/bin/env python3
"""
🌿 EfficientNetB4 + MSA Cassava Disease Classifier
Detection of Cassava Leaf Disease Classes
Author: Okobi Ebubechukwu Moses | 22CD009441 | Landmark University
Supervisor: Dr. Odeyemi Temitope
"""

import os
import sys
import glob
import shutil
import hashlib
import warnings
import joblib
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, UnidentifiedImageError

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# ── GLOBAL CONFIGURATION ────────────────────────────────────────────────────
SEED = 42
tf.keras.utils.set_random_seed(SEED)

IMG_SIZE = 224
NUM_CLASSES = 5
THRESHOLD = 0.5
SMOOTHING = 0.1
AUTOTUNE = tf.data.AUTOTUNE

CLASS_NAMES = ['Healthy', 'CMD', 'CBB', 'CGM', 'CBSD']
PROJECT_DIR = os.path.join(os.getcwd(), 'cassava_fyp')
os.makedirs(PROJECT_DIR, exist_ok=True)

# ── GPU / ACCELERATOR CONFIGURATION ─────────────────────────────────────────
GPUS = tf.config.list_physical_devices('GPU')
if GPUS:
    print(f'✅ GPU detected: {len(GPUS)} device(s)')
    for gpu in GPUS:
        print('  ', gpu)
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print(f'  Could not set memory growth for {gpu}: {exc}')
else:
    print('⚠️ No GPU detected. Training will run on CPU unless you enable a GPU runtime.')

# XLA disabled — not supported on DirectML (Intel/AMD GPU on Windows)
USE_XLA = False

# Mixed precision disabled for DirectML stability
USE_MIXED_PRECISION = False

if USE_MIXED_PRECISION and GPUS:
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print('✅ Mixed precision enabled:', tf.keras.mixed_precision.global_policy())
    except Exception as exc:
        USE_MIXED_PRECISION = False
        print('⚠️ Mixed precision could not be enabled:', exc)
else:
    tf.keras.mixed_precision.set_global_policy('float32')

if USE_XLA:
    tf.config.optimizer.set_jit(True)
    print('✅ XLA JIT enabled')
else:
    print('ℹ️ XLA JIT disabled (DirectML / non-CUDA backend)')

STRATEGY = tf.distribute.MirroredStrategy() if GPUS else tf.distribute.get_strategy()
print('✅ TensorFlow strategy:', type(STRATEGY).__name__)
print('✅ Replicas in sync:', STRATEGY.num_replicas_in_sync)

BATCH_SIZE = 32 * STRATEGY.num_replicas_in_sync
print(f'Project directory: {PROJECT_DIR}')
print(f'Batch size: {BATCH_SIZE}\n')


# ── DATASET PROCESSING & CLEANING FUNCTIONS ─────────────────────────────────
def find_data_directory():
    metadata_path = os.path.join(PROJECT_DIR, 'data', 'processed', 'clean_metadata.csv')
    csv_root = None
    if os.path.exists(metadata_path):
        try:
            preview_df = pd.read_csv(metadata_path)
            if 'image_path' in preview_df.columns and not preview_df.empty:
                existing_paths = [p for p in preview_df['image_path'].astype(str) if os.path.exists(p)]
                if existing_paths:
                    csv_root = os.path.commonpath(existing_paths)
                    if os.path.isfile(csv_root):
                        csv_root = os.path.dirname(csv_root)
        except Exception:
            pass

    candidates = [
        csv_root,
        os.path.join(os.getcwd(), 'data'),
        os.path.join(os.getcwd(), 'train'),
        os.path.join(os.getcwd(), '..', 'data'),
        os.path.join(PROJECT_DIR, '..', 'data'),
        os.path.join(PROJECT_DIR, 'data', 'raw', 'data'),
        os.path.join(PROJECT_DIR, 'data'),
        os.path.join(os.getcwd(), 'cassava_fyp', 'data'),
    ]

    DATA_DIR = None
    for c in candidates:
        if not c:
            continue
        if os.path.exists(c) and any(fn.lower().endswith(('.jpg', '.jpeg', '.png')) for fn in glob.glob(os.path.join(c, '**', '*'), recursive=True)):
            DATA_DIR = c
            break

    if DATA_DIR is None:
        if os.path.exists(metadata_path):
            raise FileNotFoundError('Found clean_metadata.csv, but none of its image_path entries exist on disk.')
        raise FileNotFoundError('Could not find a local directory containing dataset images. Place your dataset inside a folder named "./data".')

    print('Using DATA_DIR =', DATA_DIR)
    return DATA_DIR


def load_and_clean_metadata(DATA_DIR):
    expected_folders = {
        'Cassava___healthy': 'Healthy',
        'Cassava___mosaic_disease': 'CMD',
        'Cassava___bacterial_blight': 'CBB',
        'Cassava___green_mottle': 'CGM',
        'Cassava___brown_streak_disease': 'CBSD',
    }
    records = []
    for folder_name, label in expected_folders.items():
        folder_path = os.path.join(DATA_DIR, folder_name)
        if os.path.exists(folder_path):
            image_paths = sorted(glob.glob(os.path.join(folder_path, '*')))
            for p in image_paths:
                if p.lower().endswith(('.jpg', '.jpeg', '.png')):
                    records.append({'image_path': p, 'label': label})

    if not records:
        for sub in sorted(glob.glob(os.path.join(DATA_DIR, '*'))):
            if os.path.isdir(sub):
                imgs = [p for p in glob.glob(os.path.join(sub, '*')) if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if imgs:
                    label = os.path.basename(sub)
                    mapped = None
                    for k, v in expected_folders.items():
                        if k.lower() in label.lower() or v.lower() == label.lower():
                            mapped = v
                            break
                    if mapped is None:
                        mapped = label
                    for p in imgs:
                        records.append({'image_path': p, 'label': mapped})

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f'No valid images found under folder path: {DATA_DIR}.')

    print(f'Found {len(df):,} images before cleaning.')

    clean_rows = []
    bad_files = []
    seen_paths = set()
    for row in df.itertuples(index=False):
        p = row.image_path
        if p in seen_paths:
            continue
        seen_paths.add(p)
        try:
            with Image.open(p) as img:
                img.verify()
            clean_rows.append({'image_path': p, 'label': row.label})
        except Exception:
            bad_files.append(p)

    df_clean = pd.DataFrame(clean_rows)
    print(f'Clean images: {len(df_clean):,}')
    print(f'Removed unreadable images: {len(bad_files):,}')

    os.makedirs(os.path.join(PROJECT_DIR, 'data', 'processed'), exist_ok=True)
    out_path = os.path.join(PROJECT_DIR, 'data', 'processed', 'clean_metadata.csv')
    df_clean.to_csv(out_path, index=False)
    print('Saved clean metadata to', out_path)
    return df_clean


# ── DATA PIPELINE GENERATION ────────────────────────────────────────────────
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method='bicubic')
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.12)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.image.random_saturation(image, 0.85, 1.15)
    return image, label


def _dataset_options():
    options = tf.data.Options()
    options.experimental_deterministic = False
    return options


def make_image_dataset(frame, training=False, batch_size=BATCH_SIZE):
    paths = frame['image_path'].values
    labels = frame['label_idx'].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_dataset_options())
    ds = ds.map(lambda p, y: (preprocess_image(p), y), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=training).prefetch(AUTOTUNE)
    return ds


def make_balanced_train_dataset(frame, batch_size=BATCH_SIZE):
    class_frames = [frame[frame['label_idx'] == class_index] for class_index in range(NUM_CLASSES)]
    max_class_size = max(len(class_frame) for class_frame in class_frames) if class_frames else 0
    balanced_parts = []
    for class_frame in class_frames:
        if class_frame.empty:
            continue
        repeats = int(np.ceil(max_class_size / len(class_frame)))
        repeated_frame = pd.concat([class_frame] * repeats, ignore_index=True)
        repeated_frame = repeated_frame.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        balanced_parts.append(repeated_frame.iloc[:max_class_size].copy())

    if not balanced_parts:
        return make_image_dataset(frame, training=True, batch_size=batch_size), len(frame)

    balanced_frame = pd.concat(balanced_parts, ignore_index=True)
    balanced_frame = balanced_frame.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    total_samples = len(balanced_frame)
    return make_image_dataset(balanced_frame, training=True, batch_size=batch_size), total_samples


# ── LEAK DETECTION ──────────────────────────────────────────────────────────
def file_sha1(path, chunk_size=1024 * 1024):
    hasher = hashlib.sha1()
    try:
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return path


def split_leak_report(train_frame, val_frame, test_frame):
    splits = {'train': train_frame, 'val': val_frame, 'test': test_frame}
    print('\n--- RUNNING INTEGRITY LEAK CHECK ---')
    path_sets = {name: set(frame['image_path'].astype(str)) for name, frame in splits.items()}
    for left_name in splits:
        for right_name in splits:
            if left_name >= right_name:
                continue
            overlap = path_sets[left_name] & path_sets[right_name]
            print(f'  {left_name} vs {right_name}: {len(overlap)} shared paths')

    hash_sets = {}
    for name, frame in splits.items():
        hashes = [file_sha1(path) for path in frame['image_path'].astype(str) if os.path.exists(path)]
        hash_sets[name] = set(hashes)

    for left_name in splits:
        for right_name in splits:
            if left_name >= right_name:
                continue
            overlap = hash_sets[left_name] & hash_sets[right_name]
            print(f'  {left_name} vs {right_name}: {len(overlap)} shared data image hashes')


# ── MODEL ARCHITECTURE ──────────────────────────────────────────────────────
def msa_block(feature_map):
    channels = int(feature_map.shape[-1])
    squeeze = layers.GlobalAveragePooling2D()(feature_map)
    squeeze = layers.Dense(max(channels // 8, 1), activation='relu')(squeeze)
    squeeze = layers.Dense(channels, activation='sigmoid')(squeeze)
    squeeze = layers.Reshape((1, 1, channels))(squeeze)
    channel_refined = layers.Multiply()([feature_map, squeeze])

    spatial = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(channel_refined)
    return layers.Multiply()([channel_refined, spatial])


def build_model():
    backbone = EfficientNetB4(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    backbone.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=False)
    x = msa_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

    model = Model(inputs, outputs, name='cassava_efficientnetb4_msa')
    return model, backbone


def smooth_sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_true = tf.one_hot(y_true, NUM_CLASSES)
    y_true = y_true * (1.0 - SMOOTHING) + (SMOOTHING / NUM_CLASSES)
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))


# ── DRIVER EXECUTION PIPELINE ───────────────────────────────────────────────
def main():
    # 1. Pipeline Prep
    data_dir = find_data_directory()
    df_clean = load_and_clean_metadata(data_dir)

    label_to_idx = {'Healthy': 0, 'CMD': 1, 'CBB': 2, 'CGM': 3, 'CBSD': 4}

    if not any(lbl in df_clean['label'].values for lbl in label_to_idx):
        unique_labels = df_clean['label'].unique()
        label_to_idx = {name: idx for idx, name in enumerate(unique_labels[:NUM_CLASSES])}
        global CLASS_NAMES
        CLASS_NAMES = list(label_to_idx.keys())
        print(f"Automatically re-mapped custom classes: {CLASS_NAMES}")

    df_clean['label_idx'] = df_clean['label'].map(label_to_idx).astype(int)

    # Split allocations
    train_df, temp_df = train_test_split(df_clean, test_size=0.30, random_state=SEED, stratify=df_clean['label_idx'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df['label_idx'])

    split_leak_report(train_df, val_df, test_df)

    # Datasets
    train_ds, total_train_samples = make_balanced_train_dataset(train_df)
    val_ds = make_image_dataset(val_df, training=False)
    test_ds = make_image_dataset(test_df, training=False)

    steps_per_epoch = max(total_train_samples // BATCH_SIZE, 1)
    validation_steps = max(len(val_df) // BATCH_SIZE, 1)

    # 2. Build model
    with STRATEGY.scope():
        model, backbone = build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss=smooth_sparse_categorical_crossentropy,
            metrics=['accuracy'],
            jit_compile=USE_XLA
        )

    # Callbacks — ModelCheckpoint removed: .keras serialization crashes on
    # DirectML due to EagerTensor lr not being JSON-serializable. The final
    # model is saved manually after training completes instead.
    training_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
    ]

    # 3. Stage 1: Frozen Backbone
    print('\n🚀 Starting Stage 1: Frozen Backbone...')
    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        epochs=12,
        callbacks=training_callbacks,
        verbose=1
    )

    # 4. Stage 2: Fine-Tuning Top Backbone Layers
    backbone.trainable = True
    for layer in backbone.layers[:-40]:
        layer.trainable = False

    with STRATEGY.scope():
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-6),
            loss=smooth_sparse_categorical_crossentropy,
            metrics=['accuracy'],
            jit_compile=USE_XLA
        )

    print('\n🚀 Starting Stage 2: Fine-Tuning Top Backbone Layers...')
    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        epochs=12,
        callbacks=training_callbacks,
        verbose=1
    )

    # Save final model using TF SavedModel format.
    # model.export() is only available in Keras 3+; model.save() is the
    # correct API for TF2 Keras Functional models and saves a full
    # TF SavedModel directory (weights + computation graph).
    os.makedirs(os.path.join(PROJECT_DIR, 'model'), exist_ok=True)
    saved_model_path = os.path.join(PROJECT_DIR, 'model', 'final_model')
    model.save(saved_model_path, save_format='tf')
    print(f'📦 Model saved to {saved_model_path}')

    joblib.dump(
        {'class_names': CLASS_NAMES, 'sampling': 'balanced_oversampling', 'label_smoothing': SMOOTHING},
        os.path.join(PROJECT_DIR, 'model', 'training_metadata.joblib')
    )
    print('📦 Training metadata saved.')

    # 5. Final Evaluation
    print('\n📊 Running Final Test Evaluations...')
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_df['label_idx'].values.astype(int)

    print('\n📋 Final Target Evaluation Metrics Classification Report:')
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))


if __name__ == '__main__':
    main()