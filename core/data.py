"""Dataset loading, chronological splits, normalisation and climatology. Used by training and evaluation only
(the server never reads the dataset; it loads the small normalisation/climatology files saved next to the weights)."""
from __future__ import annotations
import numpy as np

NC, TP = 6, 1
CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
SPLITS = {"train": (2005, 2018), "val": (2019, 2019), "test": (2020, 2021), "all": (2005, 2021)}


def load(path):
    import xarray as xr
    ds = xr.open_dataset(path)
    data = ds["data"].values.astype("f4")                   # [N,6,32,36] physical units
    static = ds["static"].values.astype("f4")               # [2,32,36] land_sea_mask, orography_m
    t = ds["time"].to_index()
    doy = np.minimum(t.dayofyear.values - 1, 364)           # 0..364, same slot convention as model.doy_index
    return data, static, doy, t.year.values, t


def to_model_space(data):
    x = data.copy(); x[:, TP] = np.log1p(np.maximum(x[:, TP], 0)); return x


def fit_norm(data, mask):
    """mean/std per channel in model space (rain as log1p), fitted on the masked (training) days only."""
    lg = to_model_space(data[mask])
    return lg.mean(axis=(0, 2, 3)).astype("f4"), lg.std(axis=(0, 2, 3)).astype("f4")


def normalise(data, mean, std):
    return ((to_model_space(data) - mean[None, :, None, None]) / std[None, :, None, None]).astype("f4")


def climatology(x, doy, smooth=15):
    """[N,...] fields + 0..364 slots -> [365,...] per-slot mean with a circular running mean."""
    clim = np.zeros((365,) + x.shape[1:], "f8"); cnt = np.zeros(365)
    np.add.at(clim, doy, x); np.add.at(cnt, doy, 1)
    clim /= np.maximum(cnt, 1)[(...,) + (None,) * (x.ndim - 1)]
    h = smooth // 2
    return np.mean([np.roll(clim, o, axis=0) for o in range(-h, h + 1)], axis=0).astype("f4")


def year_mask(year, split):
    a, b = SPLITS[split]; return (year >= a) & (year <= b)


def window_starts(mask, steps):
    """Indices i (last input day) with i-2 .. i+steps all inside the mask (contiguous days)."""
    ok = np.zeros(len(mask) + steps + 3, bool); ok[:len(mask)] = mask
    idx = np.arange(2, len(mask))
    good = np.all([ok[idx + o] for o in range(-2, steps + 1)], axis=0)
    return idx[good]


def statics(static):
    return np.stack([static[0], static[1] / 1000.0]).astype("f4")
