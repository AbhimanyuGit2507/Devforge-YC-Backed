"""Build dashboard/samples.json: a handful of real 3-day starting windows (from the training file) plus land mask and a
monthly rain climatology used by the crop-stress module. Small on purpose: the dashboard drives the live simulation
endpoint, it never shows pre-rendered forecasts.

    python dashboard/make_samples.py --data data/india_climate_train.nc
"""
import argparse, json
from pathlib import Path
import numpy as np
import xarray as xr

WINDOWS = {  # name -> last input day
    "Monsoon onset (Kerala)": "2021-06-03",
    "Peak monsoon, active spell": "2019-07-25",
    "Pre-monsoon heatwave": "2019-06-01",
    "Cyclone Fani approach": "2019-05-01",
    "Winter western disturbance": "2020-01-14",
    "Retreating monsoon": "2020-10-10",
}

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/india_climate_train.nc"); a = ap.parse_args()
ds = xr.open_dataset(a.data)
t = ds["time"].to_index(); data = ds["data"]
out = {"lat": ds["lat"].values.tolist(), "lon": ds["lon"].values.tolist(), "channels": list(ds["channel"].values),
       "land": np.round(ds["static"].values[0], 2).tolist(), "samples": []}
r1 = lambda x: np.round(np.asarray(x, "f8"), 2).tolist()
tp = data.sel(channel="tp")
out["tp_clim_month"] = r1(tp.groupby("time.month").mean("time").values)  # [12,32,36] mm/day
for name, day in WINDOWS.items():
    i = t.get_loc(day); d = t[i]
    doy = d.dayofyear
    out["samples"].append({"name": name, "month": int(d.month), "doy_sincos": [float(np.sin(2 * np.pi * doy / 365)), float(np.cos(2 * np.pi * doy / 365))],
                           "init_states": r1(data.values[i - 2:i + 1])})
Path(__file__).with_name("samples.json").write_text(json.dumps(out, separators=(",", ":")))
print("wrote", len(out["samples"]), "samples")
