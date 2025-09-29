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

# train_shortcut.py
import os
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image

from data_loader import COVIDSegDataset
from models.patch_transformer_roi import TransformerROIModel
from augmentation.counterfactual import make_counterfactual, clahe_img
import torch.cuda.amp as amp

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    # default collate is fine; batch items: image tensor, mask tensor, class idx, fname
    return tuple(zip(*batch))

def train(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms: convert PIL to tensor, normalize to [0,1]
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),  # 0..1
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    ds = COVIDSegDataset(root=args.data_root, transform=transform, mask_transform=mask_transform)
    n = len(ds)
    val_len = int(n * args.val_split)
    train_len = n - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = TransformerROIModel(img_size=args.img_size, patch_size=args.patch_size,
                                embed_dim=args.embed_dim, depth=args.depth,
                                num_heads=args.num_heads, num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    scaler = amp.GradScaler(enabled=args.use_amp)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_metric = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train")
        running_loss = 0.0
        for batch in pbar:
            images, masks, cls_idxs, fnames = batch
            # images, masks are tuples -> stack
            images = torch.stack(images).to(device)  # [B,C,H,W]
            masks = torch.stack(masks).to(device)    # [B,1,H,W]
            labels = torch.tensor(cls_idxs, dtype=torch.long, device=device)

            # create counterfactual images (numpy path): use original PIL->numpy conversion path
            # Convert images back to uint8 for augmentation
            imgs_np = ((images.cpu().permute(0,2,3,1).numpy()*0.5 + 0.5) * 255).astype('uint8')
            masks_np = (masks.cpu().numpy()[:,0,:,:] > 0.5).astype('uint8')

            cf_np = []
            for im_np, m_np in zip(imgs_np, masks_np):
                # apply CLAHE randomly too (simulate enhancement)
                if random.random() < 0.5:
                    im_np = clahe_img(im_np)
                cf = make_counterfactual(im_np, m_np, p=args.cf_p)
                cf_np.append(cf)
            # back to tensor normalized
            cf_t = torch.stack([transforms.ToTensor()(x) for x in cf_np]).to(device)
            cf_t = (cf_t - 0.5) / 0.5  # normalize

            optimizer.zero_grad()
            with amp.autocast(enabled=args.use_amp):
                logits, roi_logits, phi = model(images)
                logits_cf, roi_logits_cf, phi_cf = model(cf_t)

                # classification loss
                loss_ce = criterion_ce(logits, labels)

                # ROI loss: ground-truth masks are full-size; transform predicted roi_logits to mask
                pred_roi_logits = model.roi_logits_to_mask(roi_logits, img_size=args.img_size, patch_size=args.patch_size)
                # pred_roi_logits: [B,1,H,W]
                # compute BCE with mask (mask values 0/1)
                gt_masks = masks.float()
                loss_roi = criterion_bce(pred_roi_logits, gt_masks)

                # causal (invariance) loss: L2 between phi and phi_cf
                loss_causal = torch.mean((phi - phi_cf).pow(2))

                loss_total = loss_ce + args.lambda_causal * loss_causal + args.lambda_roi * loss_roi

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss_total.item() * images.size(0)
            pbar.set_postfix({
                "loss": running_loss / ((pbar.n + 1) * images.size(0))
            })

        # validation: compute accuracy and ACS proxy (mask overlap)
        model.eval()
        correct = 0
        total = 0
        acs_scores = []
        with torch.no_grad():
            for images, masks, cls_idxs, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                labels = torch.tensor(cls_idxs, dtype=torch.long, device=device)
                logits, roi_logits, _ = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
                # ACS proxy: IoU between predicted roi mask and lung mask
                pred_mask_logits = model.roi_logits_to_mask(roi_logits, img_size=args.img_size, patch_size=args.patch_size)
                pred_mask = (torch.sigmoid(pred_mask_logits) > 0.5).float()
                inter = (pred_mask * masks).sum(dim=[1,2,3])
                union = ((pred_mask + masks) > 0).float().sum(dim=[1,2,3])
                iou = (inter / (union + 1e-7)).cpu().numpy()
                acs_scores.extend(iou.tolist())

        val_acc = correct / total
        val_acs = float(np.mean(acs_scores)) if len(acs_scores) > 0 else 0.0
        print(f"Epoch {epoch} VAL acc={val_acc:.4f} ACS_proxy={val_acs:.4f}")

        # save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }
        torch.save(ckpt, os.path.join(args.output_dir, f"ckpt_epoch{epoch}.pth"))
        if val_acc > best_val_metric:
            best_val_metric = val_acc
            torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="runs/transformer_roi")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--lambda-causal", type=float, dest="lambda_causal", default=1.0)
    parser.add_argument("--lambda-roi", type=float, dest="lambda_roi", default=0.3)
    parser.add_argument("--cf-p", type=float, default=0.5)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)