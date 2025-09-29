#!/usr/bin/env python3
# ============================================================
#  🧠 Transformer-based COVID-19 Chest X-ray Classifier
#  ------------------------------------------------------------
#  ROI-guided · Shortcut Mitigation · Reliable Clinical AI
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
import torch.nn as nn
import torch.nn.functional as F
import timm

class MultiTaskTransformer(nn.Module):
    """
    Transformer backbone with:
      - Classification head
      - ROI segmentation head
      - Optional domain adversarial head (for causal disentanglement)
    """
    def __init__(self, backbone="swin_tiny_patch4_window7_224", num_classes=4):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)
        feat_dim = self.backbone.feature_info[-1]['num_chs']

        # classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, num_classes)
        )

        # ROI segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim//2, 1, 1)   # binary mask (lungs vs bg)
        )

        # Domain adversarial head (causal disentanglement)
        self.domain_disc = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # e.g. source vs target domain
        )

    def forward(self, x, lambda_adv=0.0):
        feats = self.backbone(x)[-1]  # B, C, H, W

        # Classification
        cls_logits = self.classifier(feats)

        # ROI mask prediction
        roi_mask = self.seg_head(feats)

        # Domain adversarial
        pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        domain_logits = self.domain_disc(pooled)

        return {
            "cls_logits": cls_logits,
            "roi_mask": roi_mask,
            "domain_logits": domain_logits
        }