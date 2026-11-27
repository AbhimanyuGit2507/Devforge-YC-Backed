"""Calibrate the ensemble spread on 2019 only: per lead day and variable, scale = RMSE(ensemble mean) / mean spread,
so that reported uncertainty matches real forecast error. Writes spread_scale [10][6] into weights_v2/postproc.json.

    python fit_spread.py --data data/india_climate_train.nc --weights weights_v2 [--check test]
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from core import data as D
from core.net import Ensemble

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/india_climate_train.nc"); ap.add_argument("--weights", default="weights_v2")
ap.add_argument("--check", choices=["val", "test"], help="only report spread/error with the stored scale on these years")
a = ap.parse_args(); torch.set_num_threads(4)
data, _, doy, year, _ = D.load(a.data)
pp_path = Path(a.weights) / "postproc.json"; pp = json.loads(pp_path.read_text()); e = Ensemble(a.weights)
L = 10; w = np.cos(np.deg2rad(np.arange(6.5, 37.6, 1.0)))[:, None]; w = w / w.mean()
sp, er = np.zeros((L, 6)), np.zeros((L, 6))
scale = np.array(pp["spread_scale"]) if a.check else np.ones((L, 6))
for i in D.window_starts(D.year_mask(year, a.check or "val"), L)[::6]:
    sc = [np.sin(2 * np.pi * (doy[i] + 1) / 365), np.cos(2 * np.pi * (doy[i] + 1) / 365)]
    m = e.members(data[i - 2:i + 1], sc, L).numpy(); mu, sd, tr = m.mean(0), m.std(0) * scale[:, :, None, None], data[i + 1:i + L + 1]
    sp += np.sqrt((sd ** 2 * w).mean(axis=(2, 3))); er += np.sqrt(((mu - tr) ** 2 * w).mean(axis=(2, 3)))
ratio = sp / er
print("spread/error at days 1, 4, 8 (t2m tp u10 v10 msl tcwv):"); [print(f"  day {k + 1}:", np.round(ratio[k], 2)) for k in (0, 3, 7)]
if not a.check:
    pp["spread_scale"] = np.round(1 / ratio, 3).tolist(); pp_path.write_text(json.dumps(pp)); print("wrote spread_scale")
