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

# metrics/shortcut_metrics.py
import numpy as np
import torch

def attribution_consistency_score(pred_mask, gt_lung_mask):
    """
    ACS (proxy): IoU between predicted saliency/pred_mask and lung roi mask.
    pred_mask: torch.Tensor [B,1,H,W] binary or probabilities
    gt_lung_mask: torch.Tensor [B,1,H,W] binary
    returns mean IoU
    """
    preds = (pred_mask > 0.5).float()
    inter = (preds * gt_lung_mask).sum(dim=[1,2,3])
    union = ((preds + gt_lung_mask) > 0).float().sum(dim=[1,2,3])
    iou = (inter / (union + 1e-7)).cpu().numpy()
    return float(np.mean(iou))

def shortcut_sensitivity_index(model, images_tensor, lung_masks_tensor, perturb_fn):
    """
    SSI: how sensitive predictions are to perturbations of nuisance regions.
    perturb_fn: a function(img_np, lung_mask_np) -> perturbed_img_np
    Returns average absolute change in predicted class probabilities (L1) across examples.
    """
    device = next(model.parameters()).device
    model.eval()
    imgs = images_tensor.detach().cpu().permute(0,2,3,1).numpy()  # B,H,W,C in 0..1 normalized? depends
    masks = lung_masks_tensor.detach().cpu().numpy()[:,0,:,:]  # B,H,W
    changes = []
    with torch.no_grad():
        for i in range(len(imgs)):
            img = imgs[i]
            mask = masks[i]
            pert = perturb_fn(img, mask)
            # convert to tensor normalized using same preprocessing as training: assume inputs in 0..1
            x_orig = torch.tensor(img.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(device)
            x_pert = torch.tensor(pert.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(device)
            logits_orig, _, _ = model(x_orig)
            logits_pert, _, _ = model(x_pert)
            probs_o = torch.softmax(logits_orig, dim=1)
            probs_p = torch.softmax(logits_pert, dim=1)
            l1 = torch.abs(probs_o - probs_p).sum().item()
            changes.append(l1)
    return float(np.mean(changes))