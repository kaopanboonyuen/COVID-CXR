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

# augmentation/counterfactual.py
import cv2
import numpy as np
import random
from typing import Tuple

def to_uint8(img: np.ndarray):
    """Assume input in [0,1] or [0,255] float; convert to uint8 0-255."""
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img

def clahe_img(img: np.ndarray, clipLimit=2.0, tileGridSize=(8,8)) -> np.ndarray:
    """Apply CLAHE to grayscale or each channel (BGR). Input float or uint8."""
    img_u = to_uint8(img)
    if len(img_u.shape) == 3 and img_u.shape[2] == 3:
        lab = cv2.cvtColor(img_u, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        out = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        out = clahe.apply(img_u)
    return out.astype(np.uint8)

def histeq(img: np.ndarray) -> np.ndarray:
    i = to_uint8(img)
    if i.ndim == 3:
        ycrcb = cv2.cvtColor(i, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    else:
        out = cv2.equalizeHist(i)
    return out

def random_noise_patch(h, w):
    """Return random noise image."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

def make_counterfactual(img: np.ndarray, lung_mask: np.ndarray, p=0.5) -> np.ndarray:
    """
    Replace non-lung regions (where lung_mask==0) with randomized content with probability p.
    img: HxWx3 RGB uint8 or float
    lung_mask: HxW binary (0/1) or uint8 mask (255)
    """
    if random.random() > p:
        return img  # no change
    H, W = lung_mask.shape[:2]
    img_u = to_uint8(img)
    mask_bin = (lung_mask > 0).astype(np.uint8)
    # create random background: patch-wise or blur of other images could be used; here simple noise + blur
    noise = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    # apply gaussian blur to noise to be less harsh
    noise = cv2.GaussianBlur(noise, (9,9), 0)
    out = img_u.copy()
    mask3 = np.repeat((1 - mask_bin)[:, :, None], 3, axis=2)
    out = out * np.repeat(mask_bin[:, :, None], 3, axis=2) + noise * mask3
    out = out.astype(np.uint8)
    return out