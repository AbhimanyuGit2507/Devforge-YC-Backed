"""Offline self-evaluation on the held-out years (2020-2021), using the organisers' own scoring functions (monsoon_r1 kit).

    python evaluate.py --data data/india_climate_train.nc
Prints RMSE skill vs persistence/climatology, heavy-rain CSI skill, anti-blur variance ratios, shock causality, timing.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import xarray as xr
from monsoon_r1 import scoring as S
from monsoon_r1.contract import BOX_BAY_OF_BENGAL, BOX_ARABIAN_SEA, LEAD_DAYS
import model as twin
from core.net import Ensemble

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/india_climate_train.nc")
ap.add_argument("--weights", default="weights")
ap.add_argument("--every", type=int, default=7)
ap.add_argument("--v2", help="evaluate a core.net.Ensemble directory instead of the baseline weights")
ap.add_argument("--json", help="also write the summary numbers to this JSON file")
ap.add_argument("--years", default="test", choices=["test", "val"])
a = ap.parse_args()

ds = xr.open_dataset(a.data); data = ds["data"].values.astype("f4"); t = ds["time"].to_index()
std = np.array(ds["data"].std(dim=("time", "lat", "lon")).values, "f4")
doy = np.minimum(t.dayofyear.values - 1, 364)
tr = t.year.values < (2020 if not a.v2 else 2019)  # climatology reference years
clim = np.zeros((365,) + data.shape[1:]); cnt = np.zeros(365)
np.add.at(clim, doy[tr], data[tr]); np.add.at(cnt, doy[tr], 1); clim /= cnt[:, None, None, None]
clim = np.mean([np.roll(clim, o, 0) for o in range(-7, 8)], 0).astype("f4")
m = Ensemble(a.v2) if a.v2 else twin.load(Path(a.weights))

ev = (t.year.values >= 2020) if a.years == "test" else (t.year.values == 2019)
idx = [i for i in np.where(ev)[0] if ev[i - 2] and i + 10 < len(t) and ev[i + 10]][::a.every]
ss, cs, ratios, sp, ms = [], [], [], [], []
for i in idx:
    sc = [float(np.sin(2 * np.pi * (doy[i] + 1) / 365)), float(np.cos(2 * np.pi * (doy[i] + 1) / 365))]
    t0 = time.perf_counter(); out = m.rollout(data[i - 2:i + 1], sc, 10); ms.append((time.perf_counter() - t0) * 1e3)
    assert not S.validity(out, 10), S.validity(out, 10)
    L = [k - 1 for k in LEAD_DAYS]
    true = data[[i + k for k in LEAD_DAYS]]; pred = out[L]
    pers = np.repeat(data[i][None], 3, 0); cl = clim[[(doy[i] + k) % 365 for k in LEAD_DAYS]]
    r = S.skill_vs_best_ref(pred, true, pers, cl); ss.append(np.nanmean(r["ss"])); sp.append(np.nanmean(r["ss_persist"][None]))
    c = [S.csi_skill(S.csi(pred[:, 1], true[:, 1]), S.csi(pers[:, 1], true[:, 1]), S.csi(cl[:, 1], true[:, 1]))]
    cs.append(c[0]); ratios.append(S.variance_ratio(out[7], data[i + 8]))
fp = S.forecast_points(ss, cs)
print(f"windows {len(idx)}  mean SS {fp['mean_ss']:.3f}  SS vs persistence {np.mean(sp):.3f}  CSI skill {fp['mean_csi_skill']:.3f}")
print(f"forecast points {fp['points']:.1f}/45 (rmse {fp['rmse_pts']:.1f}, csi {fp['csi_pts']:.1f})")
R = np.array(ratios); print("day-8 variance ratio per channel", np.round(np.nanmean(R, 0), 2), f"anti-blur pts {S.anti_blur_points(R):.1f}/10")

res = []
for i in idx[::6]:
    sc = [float(np.sin(2 * np.pi * (doy[i] + 1) / 365)), float(np.cos(2 * np.pi * (doy[i] + 1) / 365))]
    init = data[i - 2:i + 1]; base = m.rollout(init, sc, 10)
    null_ok = S.null_consistent(base, m.rollout(init, sc, 10, np.zeros((6, 32, 36), "f4")), std)
    rr = [S.causality_checks(base, m.rollout(init, sc, 10, S.make_shock(n, init[-1], std)), box, std)
          for n, box in (("warm_ocean", BOX_BAY_OF_BENGAL), ("monsoon_stall", BOX_ARABIAN_SEA))]
    res.append(S.causality_points(rr, null_ok))
    print(t[i].date(), "null", null_ok, [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rr])
print(f"causality pts mean {np.mean(res):.1f}/15   inference ms (10 steps) median {np.median(ms):.0f}")

if a.json:
    import json
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps({"windows": len(idx), "mean_ss": fp["mean_ss"], "ss_persist": float(np.mean(sp)),
        "csi_skill": fp["mean_csi_skill"], "forecast_pts": fp["points"], "var_ratio": np.nanmean(R, 0).round(3).tolist(),
        "anti_blur_pts": S.anti_blur_points(R), "causality_pts": float(np.mean(res)), "ms_median": float(np.median(ms))}, indent=1))
