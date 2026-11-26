"""Fit per-lead rain scaling for a core.net.Ensemble on the 2019 validation year only (max heavy-rain CSI > 10 mm/day).

    python fit_postproc.py --data data/india_climate_train.nc --weights weights_v2
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from core import data as D
from core.net import Ensemble

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/india_climate_train.nc"); ap.add_argument("--weights", default="weights_v2")
a = ap.parse_args(); torch.set_num_threads(4)
data, _, doy, year, _ = D.load(a.data)
e = Ensemble(a.weights); e.pp = None
idx = D.window_starts(D.year_mask(year, "val"), 10)[::2]
P, T = [], []
for i in idx:
    sc = [np.sin(2 * np.pi * (doy[i] + 1) / 365), np.cos(2 * np.pi * (doy[i] + 1) / 365)]
    P.append(e.rollout(data[i - 2:i + 1], sc, 10)[:, D.TP]); T.append(data[i + 1:i + 11, D.TP])
P, T = np.stack(P, 1), np.stack(T, 1)                      # [10, n, H, W]
scales = []
for k in range(10):
    t = T[k] > 10; best = (-1, 1.0)
    for s in np.arange(0.7, 3.01, 0.02):
        p = P[k] * s > 10; c = (p & t).sum() / max((p | t).sum(), 1)
        best = max(best, (c, float(s)))
    scales.append(best[1]); print(f"lead {k + 1}: scale {best[1]:.2f} CSI {best[0]:.3f}")
(Path(a.weights) / "postproc.json").write_text(json.dumps({"tp_scale": scales}))
