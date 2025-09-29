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

import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.multitask_transformer import MultiTaskTransformer
from utils import ChestXrayDataset

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomApply([transforms.RandomErasing()], p=0.3),  # counterfactual augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_ds = ChestXrayDataset(args.csv, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # model
    model = MultiTaskTransformer(num_classes=args.num_classes).to(device)

    # losses
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    ce_domain = nn.CrossEntropyLoss()

    optim_all = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            out = model(imgs)
            loss_cls = ce_loss(out["cls_logits"], labels)

            # ROI loss (dummy target if masks available)
            if hasattr(train_ds, "mask_tensor"):
                mask_gt = train_ds.mask_tensor.to(device)
                loss_roi = bce_loss(out["roi_mask"], mask_gt)
            else:
                loss_roi = 0.0

            # Domain loss (if domain labels available)
            if hasattr(train_ds, "domain_labels"):
                dom_labels = train_ds.domain_labels.to(device)
                loss_dom = ce_domain(out["domain_logits"], dom_labels)
            else:
                loss_dom = 0.0

            total_loss = loss_cls + args.lambda_roi*loss_roi + args.lambda_dom*loss_dom

            optim_all.zero_grad()
            total_loss.backward()
            optim_all.step()

            pbar.set_postfix({"cls": float(loss_cls), "roi": float(loss_roi), "dom": float(loss_dom)})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="training CSV")
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-roi", type=float, default=1.0)
    ap.add_argument("--lambda-dom", type=float, default=0.1)
    args = ap.parse_args()
    train(args)