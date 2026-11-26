"""Inverse shock discovery: find the smallest smooth change to today's weather that produces a target rain anomaly.

Freeze one trained Twin, optimise a delta on the last input day with Adam, backpropagating through the rollout:
    loss = (mean rain in box at day k - target)^2 / target^2 + lam * ||delta / std||^2
The gradient at step 0 is the adjoint sensitivity map: where a change today would move that rain the most.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from .net import LAT, LON, TP, doy_index

CHANNELS = (0, 2, 3, 5)   # shock t2m, winds and moisture; rain/pressure follow from the model
BOUND = (5.0, 5.0, 5.0, 10.0)   # max physical change: 5 K, 5 m/s, 5 m/s, 10 kg/m2 (keeps triggers realistic)


def box(lat0, lat1, lon0, lon1):
    return torch.from_numpy(((LAT >= lat0) & (LAT <= lat1))[:, None] & ((LON >= lon0) & (LON <= lon1))[None, :])


def smooth(x):  # 3 passes of a 3x3 mean filter keep the discovered shock physically smooth
    for _ in range(3):
        x = F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
    return x


def discover(twin, init_states, doy_sincos, target_box, day, target_mm, iters=60, lam=0.2):
    """-> (delta [6,32,36] physical, sensitivity [32,36], history of reached rain)."""
    twin.eval()
    for p in twin.parameters(): p.requires_grad_(False)
    init = torch.from_numpy(np.asarray(init_states, "f4"))
    hist0 = twin.to_norm(init)[None]
    m = box(*target_box)
    z = torch.zeros(1, twin.net.cond[0].in_features - 3)
    doy = torch.tensor([doy_index(doy_sincos)])
    raw = torch.zeros(1, len(CHANNELS), 32, 36, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.05)
    sens, reached = None, []
    bound = (torch.tensor(BOUND) / twin.std[0, list(CHANNELS), 0, 0]).view(1, -1, 1, 1)   # physical -> normalised
    shape = lambda r: bound * torch.tanh(smooth(r) / bound)                                # smooth, then softly capped
    for it in range(iters):
        d = torch.zeros(1, 6, 32, 36); d[:, list(CHANNELS)] = shape(raw)
        hist = torch.cat([hist0[:, :2], (hist0[:, 2] + d)[:, None]], 1)
        rain = twin.to_phys(twin.unroll(hist, doy, day, z))[0, day - 1, TP][m].mean()
        loss = (rain - target_mm) ** 2 / max(target_mm, 1.0) ** 2 + lam * (d ** 2).mean()
        opt.zero_grad(); loss.backward()
        if it == 0:
            sens = raw.grad.abs().sum(1)[0].numpy()
        opt.step(); reached.append(float(rain.detach()))
    with torch.no_grad():
        d = torch.zeros(1, 6, 32, 36); d[:, list(CHANNELS)] = shape(raw)
        delta = (d * twin.std)[0].numpy()                       # normalised -> physical (rain channel untouched)
    return delta.astype("f4"), (sens / max(sens.max(), 1e-12)).astype("f4"), reached
