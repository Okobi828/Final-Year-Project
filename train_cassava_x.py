#!/usr/bin/env python3
"""
Cassava CycleGAN Co-infection Classification Pipeline  (v3)
=============================================================

FOUR EXPERIMENTAL VARIANTS — each gets its own HPT + cross-validation run,
then a final model trained on the full training set with the best found HPs.

  variant A  — hard_undersample_3000
               CMD capped at 3,000 training images. Fixed blend alpha=0.55.
               Single 11-class softmax head. This is the controlled hard
               undersampling condition for comparison.

  variant B  — cmd_blend_fix
               No undersampling but CMD blend alpha raised to 0.70 so CMD
               features are not washed out during co-infection blending.
               Single 11-class softmax head.

  variant C  — modest_undersample
               CMD capped at 6,000 training images (≈2.3× next largest class).
               Blend alpha 0.55 (default). Single 11-class softmax head.

  variant D  — two_stage
               No undersampling.  Stage-1: 5-class single-disease head.
               Stage-2: co-infection detector that re-uses the Stage-1 backbone
               and fires when Stage-1 top-2 probabilities both exceed a threshold.
               HPT tunes both stages independently.

Hyperparameters tuned per variant (via random search + 3-fold stratified CV):
  - learning_rate_stage1 : {1e-3, 3e-4, 1e-4}
  - learning_rate_stage2 : {1e-5, 5e-6, 1e-6}
  - dropout_1            : {0.40, 0.50, 0.60}
  - dropout_2            : {0.30, 0.40, 0.50}
  - focal_gamma          : {1.5, 2.0, 2.5}
  - weight_decay         : {1e-4, 5e-4, 1e-3}
  - [variant D only]
    coinf_threshold       : {0.20, 0.25, 0.30}

Cross-validation note
---------------------
Full 5-fold image-level CV on a 21k dataset with CycleGAN synthesis would
take many days on a single GPU.  The script uses a FAST-CV approach:
  • CycleGAN is trained ONCE on the full training split (not per fold).
  • The classifier head is cross-validated (3 folds) on the combined
    real + synthetic training set, keeping val/test splits as fixed holdouts.
  • This is the standard practice in medical imaging papers when generative
    augmentation is involved (Shin et al. 2018, Frid-Adar et al. 2018).

Usage
-----
Run all variants sequentially:
    python train_cassava_co.py --data_dir ./data

Run a single variant:
    python train_cassava_co.py --data_dir ./data --variant A
    python train_cassava_co.py --data_dir ./data --variant D

Skip CycleGAN if generators already saved:
    python train_cassava_co.py --data_dir ./data --skip_cyclegan

Outputs
-------
cassava_fyp_cyclegan/
  reports/
    variant_A_hard_undersample_3000/  variant_B_cmd_blend_fix/  variant_C_modest_undersample/  variant_D_two_stage/
      hpt_results.csv
      cv_fold_metrics.csv
      final_test_metrics.json
      final_coinfection_results.csv
      confusion_matrix.png
  model/
    variant_A_final/  variant_B_final/  ...
"""

import os, glob, json, math, hashlib, argparse, warnings, copy, time
from dataclasses import dataclass, asdict
from collections import Counter
from itertools import combinations, product as iterproduct
import random

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS & GLOBAL CLASS DEFINITIONS
# =============================================================================

SINGLE_DISEASE_NAMES = ("Healthy", "CMD", "CBB", "CGM", "CBSD")
COINFECTION_PAIRS    = list(combinations(("CMD", "CBB", "CGM", "CBSD"), 2))
COINFECTION_NAMES    = tuple(f"{a}+{b}" for a, b in COINFECTION_PAIRS)
ALL_CLASS_NAMES      = SINGLE_DISEASE_NAMES + COINFECTION_NAMES
# indices 0-4  → single disease/healthy
# indices 5-10 → co-infections

# Per-pair CMD blend alphas (variant B) — give CMD more weight so its subtle
# mosaic features are not drowned out by the other disease's bolder lesions
CMD_BLEND_ALPHA = {
    ("CMD", "CBB"):  0.70,
    ("CMD", "CGM"):  0.70,
    ("CMD", "CBSD"): 0.68,
    ("CBB", "CGM"):  0.55,
    ("CBB", "CBSD"): 0.55,
    ("CGM", "CBSD"): 0.55,
}

# Hyperparameter search space (shared across all variants)
HP_SPACE = {
    "lr_stage1":   [1e-3, 3e-4, 1e-4],
    "lr_stage2":   [1e-5, 5e-6, 1e-6],
    "dropout_1":   [0.40, 0.50, 0.60],
    "dropout_2":   [0.30, 0.40, 0.50],
    "focal_gamma": [1.5,  2.0,  2.5],
    "weight_decay":[1e-4, 5e-4, 1e-3],
}
HP_SPACE_TWOSTAGE = {**HP_SPACE, "coinf_threshold": [0.20, 0.25, 0.30]}

N_HPT_TRIALS = 8   # random search trials per variant (raise to 20+ for final experiments)
CV_FOLDS     = 3   # folds for cross-validation

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class CFG:
    seed:                      int   = 42
    project_dir:               str   = os.path.join(os.getcwd(), "cassava_fyp_cyclegan")
    img_size:                  int   = 224
    cyclegan_img_size:         int   = 128
    num_single_classes:        int   = 5
    num_coinfection_classes:   int   = len(COINFECTION_PAIRS)
    batch_size_base:           int   = 16
    classifier_batch_size:     int   = 32
    cyclegan_epochs:           int   = 50
    cyclegan_steps_per_epoch:  int   = 200
    stage1_epochs:             int   = 20
    stage2_epochs:             int   = 30
    cv_stage1_epochs:          int   = 10   # fewer epochs during CV for speed
    cv_stage2_epochs:          int   = 15
    max_synthetic_per_source:  int   = 3
    synthetic_ratio_cap:       float = 1.5
    coinfection_per_pair:      int   = 500
    cmd_hard_undersample_cap:  int   = 3000  # variant A only
    cmd_undersample_cap:       int   = 6000  # variant C only
    min_psnr_reject:           float = 36.0
    min_image_std:             float = 0.03
    val_size:                  float = 0.15
    test_size:                 float = 0.15
    use_mixed_precision:       bool  = False
    use_xla:                   bool  = False

    @property
    def num_classes(self):
        return self.num_single_classes + self.num_coinfection_classes

cfg  = CFG()
AUTO = tf.data.AUTOTUNE

# =============================================================================
# RUNTIME
# =============================================================================

def configure_runtime():
    os.makedirs(cfg.project_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"✅ {len(gpus)} GPU(s) detected")
        for g in gpus:
            try: tf.config.experimental.set_memory_growth(g, True)
            except: pass
    else:
        print("⚠️  No GPU — will be slow")
    if cfg.use_mixed_precision and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    tf.config.optimizer.set_jit(cfg.use_xla)
    return tf.distribute.get_strategy()

# =============================================================================
# DATA LOADING
# =============================================================================

def file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_data_dir(user_dir=None):
    for c in [user_dir, os.path.join(os.getcwd(), "data"),
              os.path.join(os.getcwd(), "train"), os.path.join(os.getcwd(), "..", "data")]:
        if c and os.path.exists(c):
            if any(p.lower().endswith((".jpg",".jpeg",".png"))
                   for p in glob.glob(os.path.join(c, "**", "*"), recursive=True)):
                return c
    raise FileNotFoundError("No image data found.")


def load_metadata(data_dir):
    folder_map = {
        "Cassava___healthy":             "Healthy",
        "Cassava___mosaic_disease":      "CMD",
        "Cassava___bacterial_blight":    "CBB",
        "Cassava___green_mottle":        "CGM",
        "Cassava___brown_streak_disease":"CBSD",
    }
    rows = []
    for folder, label in folder_map.items():
        fp = os.path.join(data_dir, folder)
        if os.path.isdir(fp):
            for p in sorted(glob.glob(os.path.join(fp, "*"))):
                if p.lower().endswith((".jpg",".jpeg",".png")):
                    rows.append({"image_path": os.path.abspath(p), "label": label, "is_synthetic": 0})
    if not rows:
        for fp in sorted(glob.glob(os.path.join(data_dir, "*"))):
            if not os.path.isdir(fp): continue
            label = folder_map.get(os.path.basename(fp), os.path.basename(fp))
            for p in sorted(glob.glob(os.path.join(fp, "*"))):
                if p.lower().endswith((".jpg",".jpeg",".png")):
                    rows.append({"image_path": os.path.abspath(p), "label": label, "is_synthetic": 0})
    df = pd.DataFrame(rows)
    if df.empty: raise RuntimeError("No images found.")
    clean, bad, seen = [], [], set()
    for r in df.itertuples(index=False):
        if r.image_path in seen: continue
        seen.add(r.image_path)
        try:
            with Image.open(r.image_path) as im: im.verify()
            clean.append(r._asdict())
        except: bad.append(r.image_path)
    df = pd.DataFrame(clean)
    print(f"Clean: {len(df):,} | removed: {len(bad):,}")
    return df


def attach_labels_and_hashes(df):
    l2i = {n: i for i, n in enumerate(SINGLE_DISEASE_NAMES)}
    df  = df[df["label"].isin(l2i)].copy()
    df["label_idx"] = df["label"].map(l2i).astype(int)
    df["sha1"]      = df["image_path"].apply(file_sha1)
    return df


def stratified_split(df):
    tr, tmp = train_test_split(df, test_size=cfg.val_size + cfg.test_size,
                               random_state=cfg.seed, stratify=df["label_idx"])
    rel = cfg.test_size / (cfg.val_size + cfg.test_size)
    va, te = train_test_split(tmp, test_size=rel, random_state=cfg.seed,
                               stratify=tmp["label_idx"])
    for n, f in [("train", tr),("val", va),("test", te)]:
        print(f"{n:5s}: {len(f):,} | {dict(Counter(f['label']))}")
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def undersample_cmd(train_df, cap):
    """Randomly undersample CMD rows in training set down to `cap`."""
    cmd_mask = train_df["label"] == "CMD"
    cmd_df   = train_df[cmd_mask]
    other_df = train_df[~cmd_mask]
    if len(cmd_df) > cap:
        cmd_df = cmd_df.sample(n=cap, random_state=cfg.seed)
        print(f"  CMD undersampled: {len(train_df[cmd_mask]):,} → {cap:,}")
    return pd.concat([cmd_df, other_df], ignore_index=True)


def leak_check(train_df, val_df, test_df, syn_df=None):
    print("\n── Leakage check ──")
    pairs = [("train","val",train_df,val_df),("train","test",train_df,test_df),
             ("val","test",val_df,test_df)]
    for na,nb,a,b in pairs:
        ph = len(set(a["image_path"]) & set(b["image_path"]))
        hh = len(set(a["sha1"])       & set(b["sha1"]))
        print(f"  {na} vs {nb}: path={ph}, hash={hh}")
    if syn_df is not None and len(syn_df) and "source_sha1" in syn_df.columns:
        bv = len(set(syn_df["source_sha1"]) & set(val_df["sha1"]))
        bt = len(set(syn_df["source_sha1"]) & set(test_df["sha1"]))
        print(f"  Syn src leakage → val:{bv}  test:{bt}")
        assert bv == 0 and bt == 0, "Synthetic source leakage!"

# =============================================================================
# TF PIPELINES
# =============================================================================

def decode_cyclegan(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [cfg.cyclegan_img_size]*2, method="bicubic")
    return tf.cast(img, tf.float32) / 127.5 - 1.0


def decode_classifier(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [cfg.img_size]*2, method="bicubic")
    return tf.keras.applications.efficientnet.preprocess_input(tf.cast(img, tf.float32))


def random_cutout(img, size=48):
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    cy = tf.random.uniform([], 0, h, dtype=tf.int32)
    cx = tf.random.uniform([], 0, w, dtype=tf.int32)
    y1 = tf.clip_by_value(cy - size//2, 0, h); y2 = tf.clip_by_value(cy + size//2, 0, h)
    x1 = tf.clip_by_value(cx - size//2, 0, w); x2 = tf.clip_by_value(cx + size//2, 0, w)
    yy = tf.range(h)[:,None]; xx = tf.range(w)[None,:]
    mask = tf.cast(tf.logical_or(tf.logical_or(yy<y1,yy>=y2),tf.logical_or(xx<x1,xx>=x2)), img.dtype)
    return img * tf.expand_dims(mask, -1)


def augment(img, lbl):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.75, 1.25)
    img = tf.image.random_saturation(img, 0.75, 1.25)
    img = tf.cond(tf.random.uniform([]) < 0.35, lambda: random_cutout(img), lambda: img)
    return img, lbl


def make_clf_ds(df, training, batch_size=None):
    bs  = batch_size or cfg.classifier_batch_size
    paths  = df["image_path"].astype(str).values
    labels = df["label_idx"].astype(np.int32).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(df), 8192), seed=cfg.seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (decode_classifier(p), y), num_parallel_calls=AUTO)
    if training: ds = ds.map(augment, num_parallel_calls=AUTO)
    return ds.batch(bs, drop_remainder=False).prefetch(AUTO)


def make_cyclegan_ds(paths_a, paths_b):
    bs = cfg.batch_size_base
    def _ds(paths, seed):
        return (tf.data.Dataset.from_tensor_slices(paths)
                .shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
                .repeat().map(decode_cyclegan, num_parallel_calls=AUTO)
                .batch(bs, drop_remainder=True))
    return tf.data.Dataset.zip((_ds(paths_a, cfg.seed), _ds(paths_b, cfg.seed+1))).prefetch(AUTO)

# =============================================================================
# CYCLEGAN
# =============================================================================

def _down(f, s, norm=True):
    b = tf.keras.Sequential()
    b.add(layers.Conv2D(f, s, strides=2, padding="same", use_bias=False))
    if norm: b.add(layers.BatchNormalization())
    b.add(layers.LeakyReLU(0.2))
    return b

def _up(f, s, drop=False):
    b = tf.keras.Sequential()
    b.add(layers.Conv2DTranspose(f, s, strides=2, padding="same", use_bias=False))
    b.add(layers.BatchNormalization())
    if drop: b.add(layers.Dropout(0.5))
    b.add(layers.ReLU())
    return b

def build_generator(name="gen"):
    inp = layers.Input(shape=[cfg.cyclegan_img_size]*2 + [3])
    downs = [_down(64,4,False), _down(128,4), _down(256,4), _down(512,4)]
    ups   = [_up(256,4), _up(128,4), _up(64,4)]
    x, skips = inp, []
    for d in downs: x = d(x); skips.append(x)
    for u, s in zip(ups, reversed(skips[:-1])):
        x = u(x); x = layers.Concatenate()([x, s])
    x = layers.Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh")(x)
    return Model(inp, x, name=name)

def build_discriminator(name="disc"):
    inp = layers.Input(shape=[cfg.cyclegan_img_size]*2 + [3])
    x = _down(64,4,False)(inp); x = _down(128,4)(x); x = _down(256,4)(x)
    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(512, 4, strides=1, use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.ZeroPadding2D()(x); x = layers.Conv2D(1, 4, strides=1)(x)
    return Model(inp, x, name=name)

class CycleGAN(tf.keras.Model):
    def __init__(self, g_ab, g_ba, d_a, d_b, lc=10., li=5.):
        super().__init__()
        self.g_ab, self.g_ba, self.d_a, self.d_b = g_ab, g_ba, d_a, d_b
        self.lc, self.li = lc, li
        self._m = {k: tf.keras.metrics.Mean(name=k)
                   for k in ["gen","disc","cycle","iden"]}
    @property
    def metrics(self): return list(self._m.values())

    def compile(self, og, of, od_a, od_b):
        super().compile()
        self.og, self.of, self.od_a, self.od_b = og, of, od_a, od_b

    @staticmethod
    def bce(lab, log): return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(lab, log))
    def d_loss(self, r, f): return .5*(self.bce(tf.ones_like(r),r)+self.bce(tf.zeros_like(f),f))
    def g_loss(self, f):    return self.bce(tf.ones_like(f), f)

    def train_step(self, batch):
        ra, rb = batch
        with tf.GradientTape(persistent=True) as t:
            fb=self.g_ab(ra,training=True); ca=self.g_ba(fb,training=True)
            fa=self.g_ba(rb,training=True); cb=self.g_ab(fa,training=True)
            sa=self.g_ba(ra,training=True); sb=self.g_ab(rb,training=True)
            dra=self.d_a(ra,training=True); drb=self.d_b(rb,training=True)
            dfa=self.d_a(fa,training=True); dfb=self.d_b(fb,training=True)
            cyc=self.lc*(tf.reduce_mean(tf.abs(ra-ca))+tf.reduce_mean(tf.abs(rb-cb)))
            ide=self.li*(tf.reduce_mean(tf.abs(ra-sa))+tf.reduce_mean(tf.abs(rb-sb)))
            tgab=self.g_loss(dfb)+cyc+ide; tgba=self.g_loss(dfa)+cyc+ide
            tda=self.d_loss(dra,dfa);      tdb=self.d_loss(drb,dfb)
        for opt,loss,vars in [
            (self.og, tgab, self.g_ab.trainable_variables),
            (self.of, tgba, self.g_ba.trainable_variables),
            (self.od_a,tda, self.d_a.trainable_variables),
            (self.od_b,tdb, self.d_b.trainable_variables)]:
            opt.apply_gradients(zip(t.gradient(loss,vars),vars))
        self._m["gen"].update_state(tgab+tgba); self._m["disc"].update_state(tda+tdb)
        self._m["cycle"].update_state(cyc);     self._m["iden"].update_state(ide)
        return {m.name: m.result() for m in self.metrics}

# =============================================================================
# SYNTHETIC GENERATION
# =============================================================================

def _u8(img):
    return tf.cast(tf.clip_by_value((img+1.)*127.5, 0, 255), tf.uint8).numpy()

def _psnr(a, b):
    mse = np.mean((a.astype(np.float32)-b.astype(np.float32))**2)
    return 99. if mse == 0 else 20*np.log10(255./np.sqrt(mse))


def gen_single_disease(generator, src_paths, label, idx, needed, out_dir, sha_lut):
    os.makedirs(out_dir, exist_ok=True)
    rows = []; rng = np.random.default_rng(cfg.seed + idx)
    sel  = list(rng.choice(src_paths, size=min(len(src_paths), needed*cfg.max_synthetic_per_source), replace=False))
    for src in sel:
        if len(rows) >= needed: break
        real = decode_cyclegan(src)
        fake = generator(tf.expand_dims(real,0), training=False)[0]
        fu8, ru8 = _u8(fake), _u8(real)
        if fu8.astype(np.float32).std()/255. < cfg.min_image_std: continue
        if _psnr(ru8, fu8) > cfg.min_psnr_reject: continue
        p = os.path.join(out_dir, f"syn_{label}_{len(rows):05d}.jpg")
        Image.fromarray(fu8).resize((cfg.img_size,)*2).save(p, quality=95)
        rows.append({"image_path":os.path.abspath(p),"label":label,"label_idx":idx,
                     "is_synthetic":1,"sha1":file_sha1(p),"source_path":os.path.abspath(src),
                     "source_sha1":sha_lut.get(os.path.abspath(src),"")})
    return pd.DataFrame(rows)


def gen_coinfection_pair(g_a, g_b, h_paths, la, lb, coinf_label, coinf_idx,
                         n, out_dir, sha_lut, alpha=None):
    """
    Blend disease_A and disease_B images translated from healthy source images.
    alpha controls the weight of disease A in the blend.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []; rng = np.random.default_rng(cfg.seed + coinf_idx*100)
    if alpha is None: alpha = 0.55
    sel = list(rng.choice(h_paths, size=min(len(h_paths), n*2),
                          replace=len(h_paths) < n*2))
    for src in sel:
        if len(rows) >= n: break
        real = decode_cyclegan(src)
        rex  = tf.expand_dims(real, 0)
        fa   = g_a(rex, training=False)[0]
        fb   = g_b(rex, training=False)[0]
        bl   = alpha*fa + (1.-alpha)*fb
        bu8  = _u8(bl)
        if bu8.astype(np.float32).std()/255. < cfg.min_image_std: continue
        p = os.path.join(out_dir, f"coinf_{coinf_label.replace('+','_')}_{len(rows):05d}.jpg")
        Image.fromarray(bu8).resize((cfg.img_size,)*2).save(p, quality=95)
        rows.append({"image_path":os.path.abspath(p),"label":coinf_label,
                     "label_idx":coinf_idx,"is_synthetic":1,"sha1":file_sha1(p),
                     "source_path":os.path.abspath(src),
                     "source_sha1":sha_lut.get(os.path.abspath(src),""),
                     "component_a":la,"component_b":lb,"blend_alpha":alpha})
    print(f"  Co-inf {coinf_label}: {len(rows)}/{n}  (alpha={alpha:.2f})")
    return pd.DataFrame(rows)


def train_cyclegan_and_generate(train_df, test_df, strategy, variant_name,
                                skip=False, cmd_blend_fix=False):
    """
    Train one CycleGAN per disease, generate single-disease balance images
    and co-infection training + eval images.

    cmd_blend_fix=True  → use CMD_BLEND_ALPHA dict for per-pair alphas (variant B)
    cmd_blend_fix=False → use flat 0.55 alpha for all pairs
    """
    sha_lut  = dict(zip(train_df["image_path"], train_df["sha1"]))
    h_paths  = train_df[train_df["label"]=="Healthy"]["image_path"].astype(str).tolist()
    log_dir  = os.path.join(cfg.project_dir, "reports", variant_name, "cyclegan")
    os.makedirs(log_dir, exist_ok=True)

    real_counts  = Counter(train_df["label"])
    target_count = max(real_counts.values())
    print(f"\n[{variant_name}] Balance target: {target_count} | real counts: {dict(real_counts)}")

    if skip or len(h_paths) < 4:
        print("  Skipping CycleGAN training.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    generators = {}   # {label: gen_H_to_disease}
    single_parts = []

    for label in [l for l in SINGLE_DISEASE_NAMES if l != "Healthy"]:
        idx = SINGLE_DISEASE_NAMES.index(label)
        d_paths = train_df[train_df["label"]==label]["image_path"].astype(str).tolist()
        if len(d_paths) < 4: continue

        needed = min(max(0, target_count - real_counts[label]),
                     int(real_counts[label] * cfg.synthetic_ratio_cap))

        print(f"\n🚀 [{variant_name}] CycleGAN  Healthy ↔ {label}  (need {needed} synthetic)")
        ds = make_cyclegan_ds(h_paths, d_paths)
        with strategy.scope():
            g_hd = build_generator(f"gH_{label}"); g_dh = build_generator(f"g{label}_H")
            d_h  = build_discriminator("dH");       d_d  = build_discriminator(f"d{label}")
            cgan = CycleGAN(g_hd, g_dh, d_h, d_d)
            opt  = lambda: tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
            cgan.compile(opt(), opt(), opt(), opt())
        try:
            hist = cgan.fit(ds, epochs=cfg.cyclegan_epochs,
                            steps_per_epoch=cfg.cyclegan_steps_per_epoch, verbose=1)
            pd.DataFrame(hist.history).to_csv(os.path.join(log_dir, f"H_{label}.csv"), index=False)
        except Exception as e:
            print(f"  ⚠️ CycleGAN failed for {label}: {e}"); continue

        generators[label] = g_hd

        if needed > 0:
            out_dir = os.path.join(cfg.project_dir, "data", variant_name, "syn_single", label)
            df = gen_single_disease(g_hd, h_paths, label, idx, needed, out_dir, sha_lut)
            print(f"  Single-disease {label}: {len(df)}/{needed}")
            single_parts.append(df)

    single_df = pd.concat(single_parts, ignore_index=True) if single_parts else pd.DataFrame()

    # ── Co-infection training images ──
    coinf_train_parts = []
    if len(generators) >= 2:
        print(f"\n🔀 [{variant_name}] Generating co-infection TRAINING images …")
        for i, (la, lb) in enumerate(COINFECTION_PAIRS):
            if la not in generators or lb not in generators: continue
            coinf_label = f"{la}+{lb}"; coinf_idx = cfg.num_single_classes + i
            alpha = CMD_BLEND_ALPHA.get((la,lb), 0.55) if cmd_blend_fix else 0.55
            out = os.path.join(cfg.project_dir, "data", variant_name,
                               "syn_coinf_train", coinf_label.replace("+","_"))
            df = gen_coinfection_pair(generators[la], generators[lb], h_paths,
                                      la, lb, coinf_label, coinf_idx,
                                      cfg.coinfection_per_pair, out, sha_lut, alpha=alpha)
            coinf_train_parts.append(df)
    coinf_train_df = pd.concat(coinf_train_parts, ignore_index=True) if coinf_train_parts else pd.DataFrame()

    # ── Co-infection EVAL images from test healthy (no leakage) ──
    h_test_paths = test_df[test_df["label"]=="Healthy"]["image_path"].astype(str).tolist()
    coinf_eval_parts = []
    if generators and len(h_test_paths) >= 10:
        print(f"\n🧪 [{variant_name}] Generating co-infection EVAL set …")
        for i, (la, lb) in enumerate(COINFECTION_PAIRS):
            if la not in generators or lb not in generators: continue
            coinf_label = f"{la}+{lb}"; coinf_idx = cfg.num_single_classes + i
            alpha = CMD_BLEND_ALPHA.get((la,lb), 0.55) if cmd_blend_fix else 0.55
            out = os.path.join(cfg.project_dir, "data", variant_name,
                               "syn_coinf_eval", coinf_label.replace("+","_"))
            df = gen_coinfection_pair(generators[la], generators[lb], h_test_paths,
                                      la, lb, coinf_label, coinf_idx,
                                      min(200, len(h_test_paths)), out, sha_lut, alpha=alpha)
            coinf_eval_parts.append(df)
    coinf_eval_df = pd.concat(coinf_eval_parts, ignore_index=True) if coinf_eval_parts else pd.DataFrame()

    return single_df, coinf_train_df, coinf_eval_df, generators

# =============================================================================
# CLASSIFIER ARCHITECTURE
# =============================================================================

def msa_block(x):
    ch = int(x.shape[-1])
    c  = layers.GlobalAveragePooling2D()(x)
    c  = layers.Dense(max(ch//8,1), activation="relu")(c)
    c  = layers.Dense(ch, activation="sigmoid")(c)
    c  = layers.Reshape((1,1,ch))(c)
    x  = layers.Multiply()([x, c])
    s  = layers.Conv2D(1, 7, padding="same", activation="sigmoid")(x)
    return layers.Multiply()([x, s])


def build_classifier(num_classes, dropout_1=0.50, dropout_2=0.40, weight_decay=1e-4):
    bb = EfficientNetB4(include_top=False, weights="imagenet",
                        input_shape=(cfg.img_size,)*2+(3,))
    bb.trainable = False
    inp = layers.Input(shape=(cfg.img_size,)*2+(3,))
    x   = bb(inp, training=False)
    x   = msa_block(x)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation="relu",
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(dropout_1)(x)
    x   = layers.Dense(128, activation="relu",
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(dropout_2)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inp, out, name="cassava_clf"), bb


def focal_loss(gamma=2.0, ls=0.05, num_classes=None):
    nc = num_classes or cfg.num_classes
    def fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true,[-1]), tf.int32)
        y_oh   = tf.one_hot(y_true, nc)
        y_oh   = y_oh*(1.-ls) + ls/nc
        pt     = tf.reduce_sum(y_pred*y_oh, axis=-1, keepdims=True)
        ce     = -tf.reduce_sum(y_oh*tf.math.log(tf.clip_by_value(y_pred,1e-7,1.)), axis=-1, keepdims=True)
        return tf.reduce_mean(tf.pow(1.-pt, gamma)*ce)
    return fn

# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def get_class_weights(df):
    y  = df["label_idx"].values.astype(int)
    u  = np.unique(y)
    cw = compute_class_weight("balanced", classes=u, y=y)
    return {int(k): float(v) for k, v in zip(u, cw)}

# =============================================================================
# TRAINING LOOP (used for both HPT CV folds and final model)
# =============================================================================

def train_one_model(train_df, val_df, hps, strategy,
                    num_classes, n_s1_epochs, n_s2_epochs,
                    variant_name="", fold=None, save_path=None):
    """
    Train a two-stage EfficientNetB4 with the given HPs.
    Returns (model, val_metrics_dict).
    """
    tag = f"[{variant_name}" + (f" fold={fold}]" if fold is not None else "]")
    bs  = cfg.classifier_batch_size
    tr_ds  = make_clf_ds(train_df, True,  bs)
    val_ds = make_clf_ds(val_df,   False, bs)
    cw     = get_class_weights(train_df)

    callbacks_s1 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True, mode="min"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=2, min_lr=1e-7),
    ]
    callbacks_s2 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7,
                                         restore_best_weights=True, mode="min"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3, min_lr=1e-7),
    ]

    with strategy.scope():
        model, backbone = build_classifier(
            num_classes,
            dropout_1=hps["dropout_1"],
            dropout_2=hps["dropout_2"],
            weight_decay=hps["weight_decay"])
        model.compile(optimizer=tf.keras.optimizers.Adam(hps["lr_stage1"]),
                      loss=focal_loss(hps["focal_gamma"], num_classes=num_classes),
                      metrics=["accuracy"])

    print(f"\n  {tag} Stage 1 (lr={hps['lr_stage1']}, γ={hps['focal_gamma']})")
    model.fit(tr_ds, validation_data=val_ds, epochs=n_s1_epochs,
              class_weight=cw, callbacks=callbacks_s1, verbose=1)

    backbone.trainable = True
    for layer in backbone.layers[:-60]: layer.trainable = False

    with strategy.scope():
        model.compile(optimizer=tf.keras.optimizers.Adam(hps["lr_stage2"]),
                      loss=focal_loss(hps["focal_gamma"], num_classes=num_classes),
                      metrics=["accuracy"])

    print(f"  {tag} Stage 2 (lr={hps['lr_stage2']})")
    model.fit(tr_ds, validation_data=val_ds, epochs=n_s2_epochs,
              class_weight=cw, callbacks=callbacks_s2, verbose=1)

    # Evaluate on val
    probs = model.predict(val_ds, verbose=0)
    pred  = probs.argmax(axis=1)
    ytrue = val_df["label_idx"].values.astype(int)
    present = sorted(set(ytrue))
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(ytrue, pred)),
        "macro_f1":          float(f1_score(ytrue, pred, average="macro", zero_division=0)),
        "accuracy":          float(accuracy_score(ytrue, pred)),
        "mean_confidence":   float(np.max(probs,axis=1).mean()),
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            model.optimizer.learning_rate.assign(float(tf.keras.backend.get_value(model.optimizer.learning_rate)))
            model.save(save_path, save_format="tf")
            print(f"  ✅ Model saved → {save_path}")
        except Exception as e:
            wb = save_path.replace("_savedmodel", ".weights.h5")
            model.save_weights(wb)
            print(f"  ✅ Weights saved → {wb}")

    return model, metrics

# =============================================================================
# TWO-STAGE PREDICTOR (variant D)
# =============================================================================

class TwoStagePredictor:
    """
    Stage 1: 5-class single-disease EfficientNetB4 classifier.
    Stage 2: if the top-2 class probabilities from Stage 1 both exceed
             `threshold`, the image is flagged as a co-infection of those
             two classes; otherwise the top-1 class is returned.

    The co-infection class index maps directly to ALL_CLASS_NAMES.
    """
    def __init__(self, stage1_model, threshold=0.25):
        self.model     = stage1_model
        self.threshold = threshold
        # Build a lookup: frozenset({idx_a, idx_b}) → co-infection class index
        self._coinf_map = {}
        for i, (la, lb) in enumerate(COINFECTION_PAIRS):
            ia = SINGLE_DISEASE_NAMES.index(la)
            ib = SINGLE_DISEASE_NAMES.index(lb)
            self._coinf_map[frozenset([ia, ib])] = cfg.num_single_classes + i

    def predict_proba(self, ds):
        """
        Returns a (N, num_classes) probability array.
        Co-infection entries are filled with geometric mean of component probs.
        """
        stage1_probs = self.model.predict(ds, verbose=0)   # (N, 5)
        N = stage1_probs.shape[0]
        nc = cfg.num_classes
        out = np.zeros((N, nc), dtype=np.float32)
        out[:, :cfg.num_single_classes] = stage1_probs

        top2 = np.argsort(stage1_probs, axis=1)[:, -2:]
        for n in range(N):
            ia, ib = int(top2[n,0]), int(top2[n,1])
            pa, pb = stage1_probs[n, ia], stage1_probs[n, ib]
            if pa >= self.threshold and pb >= self.threshold:
                key = frozenset([ia, ib])
                if key in self._coinf_map:
                    ci = self._coinf_map[key]
                    # Assign co-infection probability as geometric mean
                    out[n, ci]  = float(np.sqrt(pa * pb))
                    # Zero out the individual disease probs so argmax picks co-inf
                    out[n, ia] *= 0.5
                    out[n, ib] *= 0.5
        return out

    def predict(self, ds):
        return self.predict_proba(ds).argmax(axis=1)


def train_two_stage(train_df, val_df, hps, strategy, n_s1, n_s2, save_path=None):
    """Train stage-1 (5-class) model; return TwoStagePredictor wrapper."""
    # For Stage 1 we only use single-disease rows
    tr_single = train_df[train_df["label_idx"] < cfg.num_single_classes].copy()
    va_single = val_df[val_df["label_idx"]   < cfg.num_single_classes].copy()

    model, metrics = train_one_model(
        tr_single, va_single, hps, strategy,
        num_classes=cfg.num_single_classes,
        n_s1_epochs=n_s1, n_s2_epochs=n_s2,
        variant_name="variant_D", save_path=save_path)

    threshold = hps.get("coinf_threshold", 0.25)
    predictor = TwoStagePredictor(model, threshold=threshold)
    return predictor, metrics

# =============================================================================
# HYPERPARAMETER TUNING + CROSS-VALIDATION
# =============================================================================

def random_hp_sample(space, seed):
    rng = np.random.default_rng(seed)
    return {k: rng.choice(v).item() for k, v in space.items()}


def run_hpt_cv(train_df, val_df, strategy, variant_name,
               is_two_stage=False, n_trials=N_HPT_TRIALS, n_folds=CV_FOLDS):
    """
    Random-search HPT with stratified K-fold cross-validation on train_df.
    val_df is used as a held-out sanity check but NOT for HP selection.
    Returns best_hps, hpt_results_df.
    """
    space = HP_SPACE_TWOSTAGE if is_two_stage else HP_SPACE
    # Use only single-disease rows for CV stratification (co-inf rows all synthetic)
    cv_df    = train_df[train_df["label_idx"] < cfg.num_single_classes].reset_index(drop=True)
    coinf_df = train_df[train_df["label_idx"] >= cfg.num_single_classes].reset_index(drop=True)

    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed)
    splits = list(skf.split(cv_df, cv_df["label_idx"]))

    results = []
    print(f"\n{'='*60}")
    print(f"  HPT + {n_folds}-fold CV  |  {variant_name}  |  {n_trials} trials")
    print(f"{'='*60}")

    for trial in range(n_trials):
        hps  = random_hp_sample(space, cfg.seed + trial * 31)
        fold_metrics = []

        for fold_idx, (tr_idx, va_idx) in enumerate(splits):
            print(f"\n  Trial {trial+1}/{n_trials}  Fold {fold_idx+1}/{n_folds}  HPs: {hps}")
            fold_tr = cv_df.iloc[tr_idx].copy()
            fold_va = cv_df.iloc[va_idx].copy()

            # Re-attach co-infection synthetic rows to training fold (they don't go in val)
            if len(coinf_df):
                fold_tr = pd.concat([fold_tr, coinf_df], ignore_index=True)

            if is_two_stage:
                _, m = train_two_stage(fold_tr, fold_va, hps, strategy,
                                       n_s1=cfg.cv_stage1_epochs,
                                       n_s2=cfg.cv_stage2_epochs)
            else:
                _, m = train_one_model(fold_tr, fold_va, hps, strategy,
                                       num_classes=cfg.num_classes,
                                       n_s1_epochs=cfg.cv_stage1_epochs,
                                       n_s2_epochs=cfg.cv_stage2_epochs,
                                       variant_name=variant_name, fold=fold_idx)
            fold_metrics.append(m)
            tf.keras.backend.clear_session()

        mean_bal_acc = float(np.mean([m["balanced_accuracy"] for m in fold_metrics]))
        mean_f1      = float(np.mean([m["macro_f1"]          for m in fold_metrics]))
        std_bal_acc  = float(np.std( [m["balanced_accuracy"] for m in fold_metrics]))

        row = {"trial": trial+1, "mean_balanced_accuracy": mean_bal_acc,
               "std_balanced_accuracy": std_bal_acc, "mean_macro_f1": mean_f1,
               **{f"fold_{i+1}_bal_acc": m["balanced_accuracy"] for i,m in enumerate(fold_metrics)},
               **hps}
        results.append(row)
        print(f"\n  ✦ Trial {trial+1} mean_bal_acc={mean_bal_acc:.4f}±{std_bal_acc:.4f}  mean_f1={mean_f1:.4f}")

    results_df = pd.DataFrame(results).sort_values("mean_balanced_accuracy", ascending=False)
    best_hps   = {k: results_df.iloc[0][k] for k in space}

    report_dir = os.path.join(cfg.project_dir, "reports", variant_name)
    os.makedirs(report_dir, exist_ok=True)
    results_df.to_csv(os.path.join(report_dir, "hpt_results.csv"), index=False)

    print(f"\n  Best HPs for {variant_name}: {best_hps}")
    print(f"  Best mean balanced_accuracy: {results_df.iloc[0]['mean_balanced_accuracy']:.4f}")
    return best_hps, results_df

# =============================================================================
# EVALUATION
# =============================================================================

def plot_cm(cm, names, path, title="Confusion Matrix"):
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(8, n), max(7, n-1)))
    im = ax.imshow(cm, cmap=plt.cm.Blues, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(n), yticks=np.arange(n),
           xticklabels=names, yticklabels=names,
           ylabel="True", xlabel="Predicted", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    th = cm.max()/2.
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j]>th else "black", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def evaluate_standard(predictor_or_model, val_df, test_df,
                       class_names, report_dir, is_two_stage=False):
    os.makedirs(report_dir, exist_ok=True)
    results = {}
    for split_name, df in [("val", val_df), ("test", test_df)]:
        ds     = make_clf_ds(df, False)
        ytrue  = df["label_idx"].values.astype(int)
        if is_two_stage:
            probs = predictor_or_model.predict_proba(ds)
        else:
            probs = predictor_or_model.predict(ds, verbose=0)
        pred   = probs.argmax(axis=1)
        present = sorted(set(ytrue))
        pnames  = [class_names[i] for i in present]

        m = {
            "split":             split_name,
            "n_samples":         int(len(ytrue)),
            "accuracy":          float(accuracy_score(ytrue, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(ytrue, pred)),
            "macro_f1":          float(f1_score(ytrue, pred, average="macro", zero_division=0)),
            "weighted_f1":       float(f1_score(ytrue, pred, average="weighted", zero_division=0)),
            "mean_confidence":   float(np.max(probs,axis=1).mean()),
        }
        try:
            m["macro_auc"] = float(roc_auc_score(ytrue, probs[:,present],
                                                  multi_class="ovr", average="macro",
                                                  labels=present))
        except: m["macro_auc"] = None

        with open(os.path.join(report_dir, f"{split_name}_metrics.json"), "w") as f:
            json.dump(m, f, indent=2)
        cm = confusion_matrix(ytrue, pred, labels=present)
        pd.DataFrame(cm, index=pnames, columns=pnames).to_csv(
            os.path.join(report_dir, f"{split_name}_cm.csv"))
        plot_cm(cm, pnames, os.path.join(report_dir, f"{split_name}_cm.png"), title=split_name)
        pd.DataFrame(classification_report(ytrue, pred, labels=present,
                      target_names=pnames, zero_division=0, output_dict=True)).T.to_csv(
            os.path.join(report_dir, f"{split_name}_report.csv"))
        print(f"\n{'='*50}\n  {split_name.upper()}\n{'='*50}")
        print(json.dumps(m, indent=2))
        print(classification_report(ytrue, pred, labels=present,
                                    target_names=pnames, zero_division=0))
        results[split_name] = m
    return results


def evaluate_coinfection(predictor_or_model, coinf_eval_df, class_names,
                          report_dir, is_two_stage=False):
    print(f"\n{'='*60}\n  CO-INFECTION DETECTION\n{'='*60}")
    os.makedirs(report_dir, exist_ok=True)
    if coinf_eval_df.empty:
        print("  No eval images."); return {}

    ds    = make_clf_ds(coinf_eval_df, False)
    if is_two_stage:
        probs = predictor_or_model.predict_proba(ds)
    else:
        probs = predictor_or_model.predict(ds, verbose=0)
    pred  = probs.argmax(axis=1)

    rows = []
    for i, (la, lb) in enumerate(COINFECTION_PAIRS):
        coinf_label = f"{la}+{lb}"; coinf_idx = cfg.num_single_classes + i
        mask = coinf_eval_df["label_idx"].values == coinf_idx
        if mask.sum() == 0: continue
        pp, prd, n = probs[mask], pred[mask], int(mask.sum())
        ia = SINGLE_DISEASE_NAMES.index(la); ib = SINGLE_DISEASE_NAMES.index(lb)
        direct  = int((prd == coinf_idx).sum())
        top2    = np.argsort(pp, axis=1)[:, -2:]
        partial = int(np.sum(np.logical_and(
            np.any(top2==ia, axis=1), np.any(top2==ib, axis=1))))
        dist = dict(Counter(class_names[p] for p in prd))
        row = {"pair": coinf_label, "n": n,
               "direct_hit_rate":  round(direct/n,4),
               "partial_hit_rate": round(partial/n,4),
               "mean_prob_coinf":  float(pp[:,coinf_idx].mean()),
               f"mean_prob_{la}":  float(pp[:,ia].mean()),
               f"mean_prob_{lb}":  float(pp[:,ib].mean()),
               "pred_dist": str(dist)}
        rows.append(row)
        print(f"\n  {coinf_label} (n={n})")
        print(f"    Direct {direct}/{n} ({direct/n*100:.1f}%) | Partial {partial}/{n} ({partial/n*100:.1f}%)")
        print(f"    Pred spread: {dist}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(report_dir, "coinfection_results.csv"), index=False)
        od = df["direct_hit_rate"].mean(); op = df["partial_hit_rate"].mean()
        summary = {"overall_direct_hit_rate": float(od), "overall_partial_hit_rate": float(op),
                   "per_pair": rows}
        with open(os.path.join(report_dir, "coinfection_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Overall direct: {od*100:.1f}%  |  partial: {op*100:.1f}%")
        return summary
    return {}

# =============================================================================
# VARIANT RUNNERS
# =============================================================================

def run_variant(variant_name, train_df, val_df, test_df,
                coinf_eval_df, strategy,
                is_two_stage=False):
    """
    Full pipeline for one variant:
      1. HPT + CV to find best HPs
      2. Retrain final model on full train_df with best HPs
      3. Evaluate on val + test (standard + co-infection)
      4. Save model and all reports
    """
    print(f"\n{'#'*70}")
    print(f"  VARIANT: {variant_name}")
    print(f"{'#'*70}")

    report_dir = os.path.join(cfg.project_dir, "reports", variant_name)
    model_dir  = os.path.join(cfg.project_dir, "model", f"{variant_name}_final")
    os.makedirs(report_dir, exist_ok=True)

    # 1. HPT + CV
    best_hps, hpt_df = run_hpt_cv(train_df, val_df, strategy,
                                   variant_name, is_two_stage=is_two_stage)

    # 2. Final training with best HPs on full training set
    save_path = os.path.join(model_dir, "savedmodel")
    print(f"\n🚀 [{variant_name}] Final model training with best HPs: {best_hps}")

    class_names = list(ALL_CLASS_NAMES[:cfg.num_classes])

    if is_two_stage:
        predictor, _ = train_two_stage(train_df, val_df, best_hps, strategy,
                                        n_s1=cfg.stage1_epochs,
                                        n_s2=cfg.stage2_epochs,
                                        save_path=save_path)
    else:
        model, _ = train_one_model(train_df, val_df, best_hps, strategy,
                                   num_classes=cfg.num_classes,
                                   n_s1_epochs=cfg.stage1_epochs,
                                   n_s2_epochs=cfg.stage2_epochs,
                                   variant_name=variant_name,
                                   save_path=save_path)
        predictor = model

    # 3. Evaluation
    std_results = evaluate_standard(predictor, val_df, test_df,
                                    class_names, report_dir,
                                    is_two_stage=is_two_stage)
    coinf_results = evaluate_coinfection(predictor, coinf_eval_df,
                                          class_names, report_dir,
                                          is_two_stage=is_two_stage)

    # 4. Save summary
    summary = {"variant": variant_name, "best_hps": best_hps,
               "standard_eval": std_results, "coinfection_eval": coinf_results}
    with open(os.path.join(report_dir, "variant_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    tf.keras.backend.clear_session()
    return summary

# =============================================================================
# COMPARISON TABLE
# =============================================================================

def print_comparison(summaries):
    print(f"\n{'='*70}")
    print("  VARIANT COMPARISON — TEST SET")
    print(f"{'='*70}")
    rows = []
    for s in summaries:
        te = s.get("standard_eval", {}).get("test", {})
        ci = s.get("coinfection_eval", {})
        rows.append({
            "variant":          s["variant"],
            "accuracy":         round(te.get("accuracy",0)*100,1),
            "balanced_acc":     round(te.get("balanced_accuracy",0)*100,1),
            "macro_f1":         round(te.get("macro_f1",0),3),
            "mean_confidence":  round(te.get("mean_confidence",0),3),
            "coinf_direct_%":   round(ci.get("overall_direct_hit_rate",0)*100,1),
            "coinf_partial_%":  round(ci.get("overall_partial_hit_rate",0)*100,1),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df

# =============================================================================
# MAIN
# =============================================================================

def main():
    global N_HPT_TRIALS, CV_FOLDS
    parser = argparse.ArgumentParser(description="Cassava CycleGAN Co-infection Pipeline v3")
    parser.add_argument("--data_dir",         default=None,    help="Path to dataset root")
    parser.add_argument("--variant",          default="all",   help="A | B | C | D | all")
    parser.add_argument("--skip_cyclegan",    action="store_true")
    parser.add_argument("--cyclegan_epochs",  type=int, default=cfg.cyclegan_epochs)
    parser.add_argument("--cyclegan_steps",   type=int, default=cfg.cyclegan_steps_per_epoch)
    parser.add_argument("--coinf_per_pair",   type=int, default=cfg.coinfection_per_pair)
    parser.add_argument("--hpt_trials",       type=int, default=N_HPT_TRIALS)
    parser.add_argument("--cv_folds",         type=int, default=CV_FOLDS)
    parser.add_argument("--cmd_hard_cap",     type=int, default=cfg.cmd_hard_undersample_cap)
    parser.add_argument("--cmd_cap",          type=int, default=cfg.cmd_undersample_cap)
    args = parser.parse_args()

    cfg.cyclegan_epochs          = args.cyclegan_epochs
    cfg.cyclegan_steps_per_epoch = args.cyclegan_steps
    cfg.coinfection_per_pair     = args.coinf_per_pair
    cfg.cmd_hard_undersample_cap = args.cmd_hard_cap
    cfg.cmd_undersample_cap      = args.cmd_cap

    N_HPT_TRIALS = args.hpt_trials
    CV_FOLDS     = args.cv_folds

    run_variants = ([args.variant] if args.variant != "all"
                    else ["A", "B", "C", "D"])

    strategy = configure_runtime()
    print("Project dir:", cfg.project_dir)
    print("Variants to run:", run_variants)

    # ── Data ──
    data_dir = find_data_dir(args.data_dir)
    print("Data dir:", data_dir)
    real_df = load_metadata(data_dir)
    real_df = attach_labels_and_hashes(real_df)
    real_df.to_csv(os.path.join(cfg.project_dir, "real_clean_metadata.csv"), index=False)

    train_df_full, val_df, test_df = stratified_split(real_df)
    leak_check(train_df_full, val_df, test_df)

    with open(os.path.join(cfg.project_dir, "config.json"), "w") as f:
        json.dump({**asdict(cfg),
                   "all_class_names": list(ALL_CLASS_NAMES),
                   "coinfection_pairs": [f"{a}+{b}" for a,b in COINFECTION_PAIRS],
                   "variants": run_variants}, f, indent=2)

    # ── CycleGAN — train once per unique blend strategy ──
    # Variants A, C, D all use flat alpha=0.55 → share one CycleGAN run
    # Variant B uses CMD-specific alphas       → needs its own CycleGAN run
    #
    # We also train CycleGAN on the full train_df (before any CMD undersampling)
    # so variant C just subsamples AFTER synthesis is done.

    shared_gens_available  = False
    cmd_fix_gens_available = False

    shared_single_df  = pd.DataFrame()
    shared_coinf_tr   = pd.DataFrame()
    shared_coinf_ev   = pd.DataFrame()

    cmdfix_single_df  = pd.DataFrame()
    cmdfix_coinf_tr   = pd.DataFrame()
    cmdfix_coinf_ev   = pd.DataFrame()

    needs_shared  = any(v in run_variants for v in ["A","C","D"])
    needs_cmdfix  = "B" in run_variants

    sha_lut = dict(zip(real_df["image_path"], real_df["sha1"]))

    if needs_shared and not args.skip_cyclegan:
        print("\n▶ Training shared CycleGAN (variants A, C, D) …")
        shared_single_df, shared_coinf_tr, shared_coinf_ev, _ = \
            train_cyclegan_and_generate(train_df_full, test_df, strategy,
                                        variant_name="shared", skip=False,
                                        cmd_blend_fix=False)
        shared_gens_available = True
        shared_single_df.to_csv(os.path.join(cfg.project_dir,"shared_single_syn.csv"), index=False)
        shared_coinf_tr.to_csv( os.path.join(cfg.project_dir,"shared_coinf_train.csv"), index=False)
        shared_coinf_ev.to_csv( os.path.join(cfg.project_dir,"shared_coinf_eval.csv"), index=False)
    elif needs_shared:
        # Try to reload previously generated CSVs
        for attr, fname in [("shared_single_df","shared_single_syn.csv"),
                             ("shared_coinf_tr", "shared_coinf_train.csv"),
                             ("shared_coinf_ev", "shared_coinf_eval.csv")]:
            p = os.path.join(cfg.project_dir, fname)
            if os.path.exists(p):
                locals()[attr] = pd.read_csv(p)
                print(f"  Loaded {fname}")
        shared_gens_available = True

    if needs_cmdfix and not args.skip_cyclegan:
        print("\n▶ Training CMD-blend-fix CycleGAN (variant B) …")
        cmdfix_single_df, cmdfix_coinf_tr, cmdfix_coinf_ev, _ = \
            train_cyclegan_and_generate(train_df_full, test_df, strategy,
                                        variant_name="cmdfix", skip=False,
                                        cmd_blend_fix=True)
        cmd_fix_gens_available = True
        cmdfix_single_df.to_csv(os.path.join(cfg.project_dir,"cmdfix_single_syn.csv"), index=False)
        cmdfix_coinf_tr.to_csv( os.path.join(cfg.project_dir,"cmdfix_coinf_train.csv"), index=False)
        cmdfix_coinf_ev.to_csv( os.path.join(cfg.project_dir,"cmdfix_coinf_eval.csv"), index=False)
    elif needs_cmdfix:
        for attr, fname in [("cmdfix_single_df","cmdfix_single_syn.csv"),
                             ("cmdfix_coinf_tr", "cmdfix_coinf_train.csv"),
                             ("cmdfix_coinf_ev", "cmdfix_coinf_eval.csv")]:
            p = os.path.join(cfg.project_dir, fname)
            if os.path.exists(p):
                locals()[attr] = pd.read_csv(p)
        cmd_fix_gens_available = True

    # ── Helper to assemble train set for a variant ──
    def build_train_df(base_df, single_syn, coinf_syn, undersample_cmd_to=None):
        parts = [base_df]
        if len(single_syn): parts.append(single_syn)
        if len(coinf_syn):  parts.append(coinf_syn)
        df = pd.concat(parts, ignore_index=True)
        if undersample_cmd_to:
            df = undersample_cmd(df, undersample_cmd_to)
        leak_check(base_df, val_df, test_df, single_syn if len(single_syn) else None)
        assert val_df["is_synthetic"].sum()  == 0
        assert test_df["is_synthetic"].sum() == 0
        print(f"\n  Train distribution: {dict(Counter(df['label']))}")
        return df

    # ── Run each requested variant ──
    all_summaries = []

    if "A" in run_variants:
        # Hard undersampling condition: cap CMD at 3,000 by default.
        train_A = build_train_df(train_df_full, shared_single_df, shared_coinf_tr,
                                 undersample_cmd_to=cfg.cmd_hard_undersample_cap)
        s = run_variant("variant_A_hard_undersample_3000", train_A, val_df, test_df,
                        shared_coinf_ev, strategy, is_two_stage=False)
        all_summaries.append(s)

    if "B" in run_variants:
        train_B = build_train_df(train_df_full, cmdfix_single_df, cmdfix_coinf_tr)
        s = run_variant("variant_B_cmd_blend_fix", train_B, val_df, test_df,
                        cmdfix_coinf_ev, strategy, is_two_stage=False)
        all_summaries.append(s)

    if "C" in run_variants:
        train_C = build_train_df(train_df_full, shared_single_df, shared_coinf_tr,
                                 undersample_cmd_to=cfg.cmd_undersample_cap)
        s = run_variant("variant_C_modest_undersample", train_C, val_df, test_df,
                        shared_coinf_ev, strategy, is_two_stage=False)
        all_summaries.append(s)

    if "D" in run_variants:
        # Two-stage: train set includes single-disease + co-infection images
        # but Stage 1 model only ever sees single-disease rows (filtered inside train_two_stage)
        train_D = build_train_df(train_df_full, shared_single_df, shared_coinf_tr)
        s = run_variant("variant_D_two_stage", train_D, val_df, test_df,
                        shared_coinf_ev, strategy, is_two_stage=True)
        all_summaries.append(s)

    # ── Final comparison ──
    if all_summaries:
        cmp_df = print_comparison(all_summaries)
        cmp_df.to_csv(os.path.join(cfg.project_dir, "reports", "variant_comparison.csv"), index=False)
        print(f"\n✅ All done. Reports → {os.path.join(cfg.project_dir, 'reports')}")


if __name__ == "__main__":
    main()