# 🏚️ Explainable Semi-Supervised Learning for Post-Earthquake Building Damage Assessment
> Rapid, accurate, and **interpretable** post-earthquake building damage assessment using **multi-modal data** and **semi-supervised learning**.

---

## 📋 Project Overview

This repository contains the implementation for our Major Project:

> **Explainable Multi-Modal Semi-Supervised Learning for Post-Earthquake Building Damage Mapping**

### 🎯 Objective

To develop a **rapid**, **accurate**, and **interpretable** automated damage assessment system that:

- Works effectively in **low-data scenarios** (e.g., new, unseen disasters).
- Leverages **semi-supervised learning (SSL)** to reduce manual annotation.
- Fuses **optical** and **SAR imagery** for robust performance under adverse conditions.
- Provides **explainable predictions** useful to emergency responders and decision-makers.

The project is also deployed as an **interactive Streamlit web app** to demonstrate the model’s capabilities.

---

## 👥 Team

**Department of Computer Science and Engineering**  
Jaypee University of Information Technology, Waknaghat

- **Abhinav Manda** – Roll No: `221030455`
- **Khushi Bhasin** – Roll No: `221030034`
- **Aatesh** – Roll No: `221030245`
- **Vanshika Sharma** – Roll No: `221030331`

**Supervisor:**  
- **Dr. Arvind Kumar**

---

## 🧠 Key Features & Methodology

We propose a **two-stage, multi-modal, semi-supervised** framework:

1. **Building Prior Extraction**
   - A segmentation model is applied to **pre-disaster optical imagery**.
   - Accurately localizes **building footprints** to focus on relevant regions.

2. **Multi-Modal Fusion**
   - Fuses **optical** and **SAR (Synthetic Aperture Radar)** data.
   - Maintains robustness under **cloud cover, smoke, and harsh weather** where optical data alone may fail.

3. **Semi-Supervised Learning (SSL) with EMA-Teacher**
   - Implements a **FixMatch-style EMA-Teacher framework**.
   - Uses **high-confidence pseudo-labels** generated on unlabeled data.
   - Reduces dependency on **large labeled datasets**.

4. **Vision Transformer (ViT) Backbone**
   - Utilizes **Vision Transformers** to capture **global context**.
   - Better models **large-scale structural damage patterns** than traditional CNNs.

5. **Explainable AI (XAI) with Grad-CAM**
   - Integrates **Grad-CAM** to generate **saliency maps**.
   - Provides **visual explanations** of model predictions, improving trust for:
     - Disaster response teams
     - Urban planners
     - Policy makers

---

## 🏗️ High-Level Architecture

### 🔄 Pipeline Overview

1. **Input**
   - Pre- and post-disaster image pairs (Optical + SAR).

2. **Stage 1 – Localization Network (Segmentation)**
   - Extracts building footprints from **pre-disaster optical imagery**.

3. **Stage 2 – Classification Network**
   - Semi-supervised **ViT-based classifier** with **consistency regularization**.
   - Predicts **damage level** for each building.

4. **Output**
   - **Per-building damage map**
   - **Grad-CAM heatmaps** highlighting critical regions influencing predictions.

---
## 📂 Dataset: Disaster Assessment (Supervised Learning)

The supervised learning dataset used in this project is hosted on Hugging Face:

🔗 **Hugging Face Dataset:**  
https://huggingface.co/datasets/abhinav1857/disaster_assessment

The repository contains two main folders:

- `images/` – pre/post-disaster **input images**
- `masks/` – corresponding **segmentation masks / labels**

Each mask file is aligned with its corresponding image file (same filename, different folder).

---


Make sure you have `git` (and preferably `git-lfs`) installed:

```bash
# Install Git LFS (one-time setup)
git lfs install

# Clone the dataset
git clone https://huggingface.co/datasets/abhinav1857/disaster_assessment

cd disaster_assessment
ls
# images/  masks/  README.md  .gitattributes
```

## 🧹 Preprocessing Pipeline

We use the **xBD dataset** and apply the following preprocessing steps:

1. **Bitemporal Alignment**
   - Georeferencing and aligning **pre-** and **post-disaster** images.

2. **Patch Generation**
   - Original 1024×1024 scenes are cropped into **224×224 patches**.
   - Patches are centered on **building polygons** (using segmentation masks).

3. **Data Augmentation**

   - **Teacher (Weak Augmentation)**
     - Horizontal flips  
     - Random crops  

   - **Student (Strong Augmentation)**
     - RandAugment  
     - Random Erasing  
     - Color Jitter  

---

## 📊 Model Components

- **Segmentation Network**
  - Extracts building footprints from pre-disaster optical imagery.

- **Classification Network**
  - Vision Transformer (ViT) based classifier.
  - Multi-modal fusion of optical and SAR patches.
  - Trained with semi-supervised learning using consistency loss and pseudo-labels.

- **Explainability Module**
  - Grad-CAM maps generated for each prediction.
  - Helps interpret which regions influence the predicted damage class.

---

## 🖥️ Installation & Setup

### ✅ Prerequisites

- Python **3.8+**
- PyTorch **1.10+**
- CUDA-enabled GPU (**recommended**)
- pip / virtualenv or conda (recommended)

### 📦 Setup

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# (Optional) Create and activate virtual environment
# python -m venv .venv
# source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
