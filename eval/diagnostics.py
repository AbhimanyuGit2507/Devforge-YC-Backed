"""Diagnostics for the brief, on the held-out 2020-21 test years (never used for tuning after the first gate):
skill per lead day, per season and on named extreme events, for the submitted ensemble, the baseline,
persistence and climatology. Writes eval/diagnostics.json.

    python eval/diagnostics.py --data data/india_climate_train.nc --weights weights_v2
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import data as D
from core.net import Ensemble
import model as baseline
from monsoon_r1 import scoring as S

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/india_climate_train.nc"); ap.add_argument("--weights", default="weights_v2")
a = ap.parse_args(); torch.set_num_threads(4)
data, _, doy, year, t = D.load(a.data)
tr = D.year_mask(year, "train"); clim = D.climatology(data[tr], doy[tr])
e = Ensemble(a.weights); base = baseline.load(Path("weights"))
EVENTS = {  # last input day = 2 days before the event peak, so the forecast must see it coming
    "Cyclone Amphan (Bay of Bengal, May 2020)": "2020-05-18", "Cyclone Nisarga (Arabian Sea, Jun 2020)": "2020-06-01",
    "Monsoon onset (Kerala, Jun 2020)": "2020-05-30", "Mumbai extreme rain (Aug 2020)": "2020-08-03",
    "Cyclone Tauktae (Arabian Sea, May 2021)": "2021-05-14", "Cyclone Yaas (Bay of Bengal, May 2021)": "2021-05-23",
    "Western disturbance (Jan 2021)": "2021-01-20", "Heatwave NW India (Jul 2021)": "2021-06-28"}
SEASON = {12: "Winter (DJF)", 1: "Winter (DJF)", 2: "Winter (DJF)", 3: "Pre-monsoon (MAM)", 4: "Pre-monsoon (MAM)", 5: "Pre-monsoon (MAM)",
          6: "Monsoon (JJAS)", 7: "Monsoon (JJAS)", 8: "Monsoon (JJAS)", 9: "Monsoon (JJAS)", 10: "Post-monsoon (ON)", 11: "Post-monsoon (ON)"}
L = 10

def window(i):
    sc = [float(np.sin(2 * np.pi * (doy[i] + 1) / 365)), float(np.cos(2 * np.pi * (doy[i] + 1) / 365))]
    init = data[i - 2:i + 1]; true = data[i + 1:i + L + 1]
    return {"model": e.rollout(init, sc, L), "baseline": base.rollout(init, sc, L), "persistence": np.repeat(data[i][None], L, 0),
            "climatology": clim[[(doy[i] + k) % 365 for k in range(1, L + 1)]]}, true

def skill(f, true, leads):  # mean over channels of 1 - RMSE/min(RMSE_persistence, RMSE_climatology), at the given leads
    return float(np.nanmean(S.skill_vs_best_ref(f["model"][leads], true[leads], f["persistence"][leads], f["climatology"][leads])["ss"]))

idx = D.window_starts(D.year_mask(year, "test"), L)[::3]
rm = {k: [] for k in ("model", "baseline", "persistence", "climatology")}; csi = {k: [] for k in rm}; seas = {}
for i in idx:
    f, true = window(i)
    for k in rm:
        rm[k].append([S.rmse_lat(f[k][j][None], true[j][None]) for j in range(L)])               # [L,6]
        csi[k].append([(int(((f[k][j, 1] > 10) & (true[j, 1] > 10)).sum()), int(((f[k][j, 1] > 10) | (true[j, 1] > 10)).sum())) for j in range(L)])
    seas.setdefault(SEASON[t[i].month], []).append(skill(f, true, [0, 3, 7]))
rmse = {k: np.mean(v, 0) for k, v in rm.items()}                                                  # [L,6]
best = np.minimum(rmse["persistence"], rmse["climatology"])
out = {"windows": len(idx), "channels": list(D.CHANNELS),
       "lead_skill": {k: (1 - rmse[k] / best).mean(1).round(4).tolist() for k in ("model", "baseline")},
       "lead_rmse_norm": {k: (rmse[k] / best).mean(1).round(4).tolist() for k in rmse},
       "lead_csi": {k: [round(sum(h for h, _ in w) / max(sum(u for _, u in w), 1), 4) for w in zip(*v)] for k, v in csi.items()},
       "season_skill": {s: [round(float(np.mean(v)), 4), len(v)] for s, v in seas.items()}, "events": {}}
for name, day in EVENTS.items():
    i = t.get_loc(day); f, true = window(i)
    out["events"][name] = {"skill_d1_4_8": round(skill(f, true, [0, 3, 7]), 3),
                           "rain_max_true_d2": round(float(true[1, 1].max()), 1), "rain_max_model_d2": round(float(f["model"][1, 1].max()), 1)}
Path(__file__).with_name("diagnostics.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("lead_skill", "lead_csi", "season_skill", "events")}, indent=1))
