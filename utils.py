#!/usr/bin/env python3
# ============================================================
#  🧠 Transformer-based COVID-19 Chest X-ray Classifier
#  ------------------------------------------------------------
#  ROI-guided · Shortcut Mitigation · Reliable Clinical AI
#
#  Paper: "Eliminating Shortcut Learning in Deep COVID-19 
#         Chest X-ray Classifiers to Ensure Reliable 
#         Performance in Real-World Clinical Practice"
#
#  ✨ Features:
#    🔍 Vision Transformers (Swin via timm)
#    🩺 Multi-task: Classification + ROI mask prediction
#    🧩 Causal feature disentanglement
#    🌀 Counterfactual augmentation
#    🔥 Grad-CAM & Attention visualization
#
#  Author: Kao Panboonyuen
#  License: MIT
# ============================================================

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = row['label']
        if isinstance(label, str):
            label = {"COVID":0,"Normal":1,"Viral Pneumonia":2,"Lung_Opacity":3}[label]
        return img, torch.tensor(label, dtype=torch.long)
