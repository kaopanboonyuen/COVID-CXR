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


# models/patch_transformer_roi.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, E, H/patch, W/patch]
        B, E, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]
        return x

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio),
                                                   dropout=drop, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x, src_key_padding_mask=None):
        # x: [B, N, E]
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

class TransformerROIModel(nn.Module):
    """
    Outputs:
      - class logits (C classes)
      - predicted ROI mask logits at patch resolution (N patches)
      - intermediate feature vector phi (pooled feature)
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=256,
                 depth=6, num_heads=8, num_classes=4):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.encoder = SimpleTransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        # heads
        self.class_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        # ROI decoder - predict binary mask per patch (we will upsample to image size)
        self.roi_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)  # per-patch logit
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # linear layers get default initialization

    def forward(self, x):
        # x: [B, C, H, W]
        B = x.size(0)
        x_p = self.patch_embed(x)          # [B, N, E]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B,1,E]
        x_all = torch.cat((cls_tokens, x_p), dim=1)    # [B, 1+N, E]
        x_all = x_all + self.pos_embed
        enc = self.encoder(x_all)                      # [B, 1+N, E]
        cls_feat = enc[:, 0]                           # [B, E]
        patch_feats = enc[:, 1:]                       # [B, N, E]
        logits_cls = self.class_head(cls_feat)         # [B, num_classes]
        logits_roi = self.roi_head(patch_feats).squeeze(-1)  # [B, N]
        # produce patch-level features phi for causal loss: use cls_feat or pooled patch_feats
        phi = cls_feat
        return logits_cls, logits_roi, phi

    def roi_logits_to_mask(self, logits_roi, img_size=None, patch_size=None):
        # logits_roi: [B,N]; reshape to grid and upsample to image size
        B, N = logits_roi.shape
        if img_size is None:
            img_size = self.patch_embed.img_size
        if patch_size is None:
            patch_size = self.patch_embed.patch_size
        g = self.patch_embed.grid_size
        x = logits_roi.view(B, 1, g, g)  # [B,1,g,g]
        x = F.interpolate(x, size=(img_size, img_size), mode='bilinear', align_corners=False)
        return x  # [B,1,H,W] logits
