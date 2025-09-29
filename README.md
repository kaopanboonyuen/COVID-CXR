# 🧠 Transformer-based COVID-19 Chest X-ray Classifier  
*(ROI-guided · Shortcut Mitigation · Reliable Clinical AI)*  

This repository implements the framework from the paper:  

**📄 "Eliminating Shortcut Learning in Deep COVID-19 Chest X-ray Classifiers to Ensure Reliable Performance in Real-World Clinical Practice"**  

👨‍💻 **Author:** Kao Panboonyuen  

---

## ✨ Key Features
- 🔍 **Vision Transformers** (e.g., Swin Transformer via `timm`)  
- 🩺 **Multi-task training** → Classification + ROI mask prediction  
- 🧩 **Causal feature disentanglement** (domain adversarial branch)  
- 🌀 **Counterfactual augmentation** (random erase + spurious cue masking)  
- 🔥 **Grad-CAM & Attention visualization** for interpretability  

---

## 📂 Repository Structure
```

models/
multitask_transformer.py   # Transformer backbone with multi-task heads
train_multitask.py           # Training loop with classification + ROI + domain losses
utils.py                     # Dataset and helper functions
requirements.txt             # Python dependencies
Dockerfile                   # Containerized training setup

````

---

## ⚡️ Quickstart

### 🛠️ Installation
```bash
pip install -r requirements.txt
````

---

### 🗂️ Dataset Preparation

Organize your dataset as follows:

```
COVID-19_Radiography_Dataset/
  COVID/
    images/
    masks/
  Normal/
    images/
    masks/
  Viral Pneumonia/
    images/
    masks/
  Lung_Opacity/
    images/
    masks/
```

---

### 🎯 Training

#### 🔹 Basic multi-task training:

```bash
python train_multitask.py --csv data.csv --epochs 20 --batch-size 8
```

#### 🔹 Full ROI-guided + shortcut mitigation training:

```bash
python train_shortcut.py \
  --data-root /path/to/COVID-19_Radiography_Dataset \
  --output-dir runs/transformer_roi \
  --epochs 30 \
  --batch-size 32 \
  --img-size 256 \
  --patch-size 16 \
  --embed-dim 256 \
  --depth 6 \
  --num-heads 8 \
  --lambda-causal 1.0 \
  --lambda-roi 0.3 \
  --cf-p 0.5 \
  --use-amp
```

---

## 🐳 Docker Support

Easily containerize your experiments:

```bash
docker build -t covid-transformer .
docker run -it --rm -v /path/to/data:/data covid-transformer bash
```

---

## 📊 Visualization

* ✅ ROI mask predictions
* ✅ Grad-CAM heatmaps
* ✅ Attention rollout from Transformer layers

---