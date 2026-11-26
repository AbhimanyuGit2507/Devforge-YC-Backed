"""Fit the per-lead, per-channel blend  pred = clim + a*(ensemble - clim) + b*(baseline - clim)  on 2019 only
(lat-weighted least squares), then refit the rain scale. Writes blend_a / blend_b / tp_scale into postproc.json.

    python fit_blend.py --weights weights_v2
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, torch
from core import data as D
from core.net import Ensemble
import model as baseline

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/india_climate_train.nc"); ap.add_argument("--weights", default="weights_v2")
a = ap.parse_args(); torch.set_num_threads(4)
data, _, doy, year, _ = D.load(a.data)
pp_path = Path(a.weights) / "postproc.json"; pp = json.loads(pp_path.read_text()) if pp_path.exists() else {}
pp.pop("blend_a", None); pp.pop("blend_b", None); pp_path.write_text(json.dumps(pp))
e = Ensemble(a.weights); base = baseline.load(Path("weights"))
w = np.cos(np.deg2rad(np.arange(6.5, 37.6, 1.0)))[:, None] * np.ones((1, 36))
S = 10; A = np.zeros((S, 6, 2, 2)); y = np.zeros((S, 6, 2))
for i in D.window_starts(D.year_mask(year, "val"), S)[::2]:
    sc = [np.sin(2 * np.pi * (doy[i] + 1) / 365), np.cos(2 * np.pi * (doy[i] + 1) / 365)]
    E = e.members(data[i - 2:i + 1], sc, S).mean(0).numpy(); B = base.rollout(data[i - 2:i + 1], sc, S)
    C = e.clim_phys(sc, S).numpy(); T = data[i + 1:i + S + 1]
    for k in range(S):
        for c in range(6):
            X = np.stack([(E[k, c] - C[k, c]).ravel(), (B[k, c] - C[k, c]).ravel()], 1); t = (T[k, c] - C[k, c]).ravel(); ww = w.ravel()
            A[k, c] += X.T @ (X * ww[:, None]); y[k, c] += X.T @ (t * ww)
coef = np.array([[np.linalg.solve(A[k, c] + 1e-6 * np.eye(2), y[k, c]) for c in range(6)] for k in range(S)])   # [S,6,2]
pp["blend_a"] = coef[..., 0].round(4).tolist(); pp["blend_b"] = coef[..., 1].round(4).tolist()
pp_path.write_text(json.dumps(pp))
print("a (ensemble) lead1/4/8:", coef[[0, 3, 7], :, 0].round(2).tolist()); print("b (baseline) lead1/4/8:", coef[[0, 3, 7], :, 1].round(2).tolist())
