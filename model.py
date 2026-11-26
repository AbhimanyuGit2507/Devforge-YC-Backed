"""Monsoon Digital Twin: a local-stencil residual CNN stepped autoregressively one day at a time.

Shared by train.py and serve.py. Pure torch/numpy, no I/O beyond loading the files in weights/.
Receptive field is RADIUS cells per day, so information travels at most ~RADIUS*110 km/day (finite propagation speed):
a shock cannot teleport, it can only spread outward step by step.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

NC, NLAT, NLON = 6, 32, 36
TP = 1  # precipitation channel, modelled in log1p space
LAT = np.arange(6.5, 37.6, 1.0)
LON = np.arange(65.5, 100.6, 1.0)
N_IN = 3 * NC + NC + 2 + 2 + 2  # 3 past days, target-day climatology, static(2), doy sin/cos(2), lat/lon(2)


def doy_index(sincos) -> int:
    """[sin, cos] of 2*pi*doy/365 -> 0..364 climatology slot."""
    ang = float(np.arctan2(sincos[0], sincos[1])) % (2 * np.pi)
    return int(round(ang * 365 / (2 * np.pi)) - 1) % 365


class Block(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.conv = nn.Conv2d(c, c, k, padding=k // 2)
        self.mix = nn.Conv2d(c, c, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.mix(self.act(self.conv(self.act(x))))


class StepNet(nn.Module):
    """One day forward. Spatial radius = number of 3x3 convs (here 3)."""
    RADIUS = 3

    def __init__(self, width=64):
        super().__init__()
        self.inp = nn.Conv2d(N_IN, width, 3, padding=1)
        self.body = nn.Sequential(Block(width, 1), Block(width, 3), Block(width, 1), Block(width, 1), Block(width, 1))
        self.out = nn.Conv2d(width, NC, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(nn.functional.gelu(self.body(self.inp(x))))


class Normaliser:
    """Physical <-> model space. tp goes through log1p before standardising."""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, NC, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, NC, 1, 1)

    def fwd(self, x):  # [..., 6, H, W] physical -> normalised
        x = x.clone(); x[..., TP, :, :] = torch.log1p(x[..., TP, :, :].clamp_min(0))
        return (x - self.mean) / self.std

    def inv(self, z):
        x = z * self.std + self.mean
        x = x.clone(); x[..., TP, :, :] = torch.expm1(x[..., TP, :, :]).clamp_min(0)
        return x


class Twin(nn.Module):
    """Holds the step network plus everything it conditions on (climatology, static maps, coords)."""

    def __init__(self, clim_n, static_n, mean, std, width=64):
        super().__init__()
        self.net = StepNet(width)
        self.norm = Normaliser(mean, std)
        self.register_buffer("clim", torch.as_tensor(clim_n, dtype=torch.float32))        # [365,6,H,W] normalised
        la, lo = np.meshgrid((LAT - LAT.mean()) / 10, (LON - LON.mean()) / 10, indexing="ij")
        aux = np.concatenate([static_n, la[None], lo[None]]).astype("f4")
        self.register_buffer("aux", torch.as_tensor(aux))                                   # [4,H,W]
        self.tp_scale = torch.ones(16)  # per-lead precipitation calibration (fit on validation)

    def step(self, hist, doy):
        """hist [B,3,6,H,W] normalised, doy [B] long index of the TARGET day -> next state normalised [B,6,H,W]."""
        B = hist.shape[0]
        ang = 2 * np.pi * (doy.float() + 1) / 365
        sc = torch.stack([torch.sin(ang), torch.cos(ang)], 1)[:, :, None, None].expand(B, 2, NLAT, NLON)
        x = torch.cat([hist.reshape(B, 3 * NC, NLAT, NLON), self.clim[doy], self.aux.expand(B, -1, -1, -1), sc], 1)
        return hist[:, -1] + self.net(x)

    def unroll(self, hist, doy_last, steps):
        """Autoregressive rollout in normalised space -> [B,steps,6,H,W]."""
        outs = []
        for k in range(steps):
            nxt = self.step(hist, (doy_last + k + 1) % 365)
            outs.append(nxt)
            hist = torch.cat([hist[:, 1:], nxt[:, None]], 1)
        return torch.stack(outs, 1)

    @torch.no_grad()
    def rollout(self, init_states: np.ndarray, doy_sincos, steps: int, perturbation=None) -> np.ndarray:
        """Physical in, physical out. init_states [3,6,H,W]; perturbation [6,H,W] delta added to the last day."""
        init = np.array(init_states, dtype="f4", copy=True)
        if perturbation is not None:
            init[-1] += np.asarray(perturbation, dtype="f4")
        init[:, TP] = np.maximum(init[:, TP], 0)
        hist = self.norm.fwd(torch.from_numpy(init))[None]
        z = self.unroll(hist, torch.tensor([doy_index(doy_sincos)]), steps)[0]
        out = self.norm.inv(z)
        sc = self.tp_scale[torch.arange(steps).clamp_max(len(self.tp_scale) - 1)].clamp_min(0)
        out[:, TP] *= sc[:, None, None]
        out[:, 0] = out[:, 0].clamp(240.0, 330.0)
        return np.nan_to_num(out.numpy(), nan=0.0, posinf=0.0, neginf=0.0).astype("f4")


def load(weights_dir: Path) -> Twin:
    meta = json.loads((weights_dir / "meta.json").read_text())
    clim_n = np.load(weights_dir / "climatology_norm.npy")
    static_n = np.load(weights_dir / "static_norm.npy")
    m = Twin(clim_n, static_n, meta["mean"], meta["std"], meta["width"])
    m.net.load_state_dict(torch.load(weights_dir / "model.pt", map_location="cpu", weights_only=True))
    m.tp_scale = torch.tensor(meta.get("tp_scale", [1.0] * 16), dtype=torch.float32)
    return m.eval()
