# 🌿 Cassava Leaf Disease Classification & Explainable AI Diagnosis
**A Deep Learning Framework Combining EfficientNetB4, Multi-Scale Attention (MSA), and CycleGAN Synthetic Augmentation**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Hardware](https://img.shields.io/badge/Hardware-HP%20Omen%20Workstation%20%28NVIDIA%20GPU%29-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview
Cassava (*Manihot esculenta*) is a vital staple food crop contributing to food security for over 500 million people across Sub-Saharan Africa. However, yield losses driven by viral and bacterial diseases—notably **Cassava Mosaic Disease (CMD)** and **Cassava Brown Streak Disease (CBSD)**—severely threaten agricultural productivity. 

This repository contains the complete software implementation and experimental scripts for a novel **Hybrid Deep Learning Framework** designed for high-precision leaf disease classification and explainability. Built and executed locally on a dedicated **HP Omen Development Workstation with NVIDIA GPU acceleration**, our system tackles two critical real-world agronomic challenges:
1. **Severe Class Imbalance:** Mitigated via generative adversarial image synthesis (**CycleGAN**).
2. **Subtle Foliar Symptom Overlaps & Co-Infections:** Resolved using a **Multi-Scale Attention (MSA)** classification head attached to a backbone feature extractor (**EfficientNetB4**).

---

## 🎯 Target Foliar Classes
The model is engineered to classify foliar imagery into five distinct diagnostic categories:
* `0`: **Cassava Bacterial Blight (CBB)** (*Xanthomonas phaseoli pv. manihotis*)
* `1`: **Cassava Brown Streak Disease (CBSD)** (*Cassava brown streak virus*)
* `2`: **Cassava Green Mottle (CGM)** (*Cassava green mottle virus*)
* `3`: **Cassava Mosaic Disease (CMD)** (*African cassava mosaic virus*)
* `4`: **Healthy** (Symptom-free cassava foliage)

---

## 🏗️ Methodology & Architectural Highlights

```
Raw Foliar Imagery  ──► CycleGAN Augmentation ──► EfficientNetB4 Backbone ──► Multi-Scale Attention (MSA) Head ──► Grad-CAM Explainability
```

### 1. Generative Augmentation (CycleGAN)
To overcome dataset skewness and generate realistic diagnostic variations of minority classes (such as CGM and CBSD), unpaired image-to-image translation is employed. CycleGAN synthesizes pathology features while preserving underlying leaf structural integrity without requiring pixel-aligned image pairs.

### 2. EfficientNetB4 Feature Extractor
Leveraging compound scaling across network depth, width, and input resolution ($380 \times 380$), the pre-trained EfficientNetB4 architecture extracts rich semantic feature hierarchies while maintaining computational efficiency suitable for downstream conversion to edge runtimes (e.g., TensorFlow Lite).

### 3. Multi-Scale Attention (MSA) & Two-Stage Fine-Tuning
* **Stage 1 (Warm-Up):** Backbone weights are frozen while the custom MSA head is trained at a learning rate of $\alpha = 10^{-3}$ using Adam to stabilize early gradient representations.
* **Stage 2 (End-to-End Fine-Tuning):** Top blocks of EfficientNetB4 are unfrozen alongside the attention layers and trained at a decay-scheduled learning rate ($\alpha = 10^{-5}$) to adapt feature extraction directly to cassava symptom morphology.

### 4. Explainable AI (Grad-CAM)
To foster trust among agricultural extension officers and smallholder farmers, **Gradient-Weighted Class Activation Mapping (Grad-CAM)** overlay heatmaps highlight exact lesion boundaries and chlorotic streaks driving model predictions.

---

## 📁 Repository Structure

```text
├── train_cassava.py           # Baseline EfficientNetB4 training pipeline (Single-stage & data loaders)
├── train_cassava_co.py        # Core Hybrid Pipeline: CycleGAN data synthesis + MSA + Two-stage fine-tuning ⭐
├── train_cassava_x.py         # Multi-Variant Ablation & Hyperparameter tuning experiments
├── evaluate_cassava.py        # Complete Evaluation Suite: Confusion Matrix, ROC/PR Curves & Grad-CAM overlays
├── evaluate.py                # Lightweight inference script for rapid test set benchmarking
└── cassava-fyp-...gpu.ipynb   # Interactive Jupyter Notebook for exploratory data analysis and GPU run logs
```

---

## 💻 System Setup & Local Execution

### Hardware & Environment Specifications
* **Workstation:** HP Omen Gaming & Development Workstation
* **Processor / GPU:** Dedicated NVIDIA GeForce RTX Acceleration (CUDA 12.x / cuDNN enabled)
* **Storage:** Local high-speed NVMe SSD persistent storage
* **Python Version:** `3.10+`

### Installation & Quick Start

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Okobi828/Final-Year-Project.git
   cd Final-Year-Project
   ```

2. **Create & Activate Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Core Dependencies:**
   ```bash
   pip install tensorflow==2.15.0 scikit-learn matplotlib seaborn numpy pandas opencv-python jupyter
   ```

4. **Running Training & Evaluation:**
   ```bash
   # Execute the primary hybrid CycleGAN + MSA training script
   python train_cassava_co.py

   # Generate classification reports and Grad-CAM visualizations
   python evaluate_cassava.py
   ```

---

## 🔬 Note on Dataset Storage
Due to GitHub file size limits (100 MB), raw full-resolution image archives (`data.zip`, ~2.4 GB) and heavy compiled weights (`.keras` / `.h5`) are excluded from this remote tracking repository. The dataset corresponds to the benchmark **Cassava Leaf Disease Classification** dataset available via Kaggle. Place the uncompressed `data/` directory in the project root prior to execution.

---

## 🎓 Academic Attribution & Reference
This software repository accompanies the undergraduate final-year research thesis submitted to the **Department of Computer Science, Landmark University, Omu-Aran, Nigeria (2025/2026 Academic Session)**.

> **Suggested Citation:**  
> Okobi / Landmark University (2025). *A Deep Learning Approach to Multi-Class Cassava Leaf Disease Detection Using EfficientNetB4, Multi-Scale Attention, and CycleGAN Augmentation*. B.Sc. Thesis Project Repository. Available at: `https://github.com/Okobi828/Final-Year-Project.git`
