"""Training losses. All inputs in normalised model space: pred [M,B,K,6,H,W] (M ensemble members), tgt [B,K,6,H,W]."""
from __future__ import annotations
import numpy as np
import torch

H = 32


def lat_weights(device):
    w = torch.cos(torch.deg2rad(torch.arange(6.5, 37.6, 1.0, device=device)))
    return (w / w.mean()).view(1, 1, 1, H, 1)


def af_crps(pred, tgt, w, alpha=0.95):
    """Almost-fair CRPS (Lang et al. 2024, ECMWF AIFS-CRPS) over M members. Rewards accurate AND sharp, well-spread members."""
    M = pred.shape[0]
    skill = (pred - tgt[None]).abs().mean(0)
    if M == 1:
        return (skill * w).mean()
    eps = (1 - alpha) / M
    spread = sum((pred[i] - pred[j]).abs() for i in range(M) for j in range(M) if i != j) / (M * (M - 1))
    return ((skill - 0.5 * (1 - eps) * spread) * w).mean()


def spectral(pred, tgt):
    """L1 between log 2-D amplitude spectra (fp32): penalises the loss of small-scale structure (blur)."""
    a = torch.fft.rfft2(pred.float(), norm="ortho").abs(); b = torch.fft.rfft2(tgt.float(), norm="ortho").abs()
    return (torch.log1p(a) - torch.log1p(b)[None]).abs().mean()


def physics(pred_phys_tp, pred_phys_tcwv):
    """Soft constraints in physical units: no meaningful rain from a very dry column (< 10 kg/m2)."""
    dry = torch.relu(10.0 - pred_phys_tcwv) / 10.0
    return (dry * torch.relu(pred_phys_tp - 1.0)).mean()
