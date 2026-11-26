"""Train the Monsoon Digital Twin on india_climate_train.nc (CPU is enough: ~1-2 h with 4 threads).

    python train.py --data data/india_climate_train.nc --out weights

Curriculum: 1-step teacher forcing, then autoregressive unrolls of 4 and 8 days with backprop through the whole rollout,
so the model is trained on its own outputs (prevents drift). Loss = latitude-weighted MSE in normalised space plus a
spatial-spread term per channel (prevents blur). Years 2020-2021 are held out for validation and tp calibration.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import xarray as xr
from model import Twin, NC, TP, LAT

VAL_FROM_YEAR = 2020


def load_data(path):
    ds = xr.open_dataset(path)
    data = ds["data"].values.astype("f4")                    # [N,6,32,36] physical
    static = ds["static"].values.astype("f4")                # [2,32,36]
    t = ds["time"].to_index()
    doy = np.minimum(t.dayofyear.values - 1, 364)            # 0..364 (Dec 31 of leap years shares slot 364)
    return data, static, doy, t.year.values


def climatology(xn, doy, smooth=15):
    clim = np.zeros((365,) + xn.shape[1:], "f8"); cnt = np.zeros(365)
    np.add.at(clim, doy, xn); np.add.at(cnt, doy, 1)
    clim /= np.maximum(cnt, 1)[:, None, None, None]
    h = smooth // 2
    return np.mean([np.roll(clim, o, axis=0) for o in range(-h, h + 1)], axis=0).astype("f4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/india_climate_train.nc")
    ap.add_argument("--out", default="weights")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--iters", default="4000,2000,1000", help="iterations for unroll 1, 4, 8")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    data, static, doy, year = load_data(a.data)
    tr = year < VAL_FROM_YEAR
    # normalisation stats in model space (tp in log1p)
    lg = data.copy(); lg[:, TP] = np.log1p(np.maximum(lg[:, TP], 0))
    mean = lg[tr].mean(axis=(0, 2, 3)); std = lg[tr].std(axis=(0, 2, 3))
    xn = ((lg - mean[None, :, None, None]) / std[None, :, None, None]).astype("f4")
    clim_n = climatology(xn[tr], doy[tr])
    static_n = np.stack([static[0], static[1] / 1000.0]).astype("f4")

    model = Twin(clim_n, static_n, mean.tolist(), std.tolist(), a.width)
    X = torch.from_numpy(xn); D = torch.from_numpy(doy).long()
    w = torch.tensor(np.cos(np.deg2rad(LAT)), dtype=torch.float32); w = (w / w.mean()).view(1, 1, 1, -1, 1)
    chw = torch.tensor([1, 2, 1, 1, 1, 1], dtype=torch.float32).view(1, 1, NC, 1, 1)  # extra weight on rain

    stages = [(1, int(n)) if i == 0 else ((4, int(n)) if i == 1 else (8, int(n))) for i, n in enumerate(a.iters.split(","))]
    total = sum(n for _, n in stages)
    opt = torch.optim.AdamW(model.net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)

    def starts(K, mask_years):
        # index i = last input day; need i-2 .. i+K all in the chosen split
        ok = np.where(mask_years)[0]
        s = set(ok.tolist())
        return np.array([i for i in ok if i - 2 in s and i + K in s])

    it, t0 = 0, time.time()
    for K, n in stages:
        idx_all = starts(K, tr)
        for _ in range(n):
            i = torch.from_numpy(rng.choice(idx_all, a.batch))
            hist = torch.stack([X[i - 2], X[i - 1], X[i]], 1)
            tgt = torch.stack([X[i + k] for k in range(1, K + 1)], 1)
            pred = model.unroll(hist, D[i], K)
            mse = (((pred - tgt) ** 2) * w * chw).mean()
            spread = ((pred.std(dim=(-2, -1)) - tgt.std(dim=(-2, -1))) ** 2).mean()  # keep spatial variance (anti-blur)
            loss = mse + 0.2 * spread
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
            opt.step(); sched.step(); it += 1
            if it % 200 == 0:
                print(f"it {it}/{total} K={K} loss {loss.item():.4f} mse {mse.item():.4f} spread {spread.item():.4f} {time.time() - t0:.0f}s", flush=True)
        torch.save(model.net.state_dict(), out / "model.pt")

    model.eval()
    tp_scale = calibrate_tp(model, data, X, D, ~tr, steps=14)
    np.save(out / "climatology_norm.npy", clim_n)
    np.save(out / "static_norm.npy", static_n)
    (out / "meta.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist(), "width": a.width,
                                               "tp_scale": tp_scale, "channels": ["t2m", "tp", "u10", "v10", "msl", "tcwv"]}, indent=1))
    print("saved to", out, "tp_scale", np.round(tp_scale, 3))


@torch.no_grad()
def calibrate_tp(model, data, X, D, val, steps=14, thr=10.0):
    """Per-lead multiplicative rain scale that maximises heavy-rain CSI (> thr mm/day) on validation."""
    idx = np.array([i for i in np.where(val)[0] if val[i - 2] and i + steps < len(val) and val[i + steps]])[::3]
    preds = []
    for b in range(0, len(idx), 64):
        i = torch.from_numpy(idx[b:b + 64])
        hist = torch.stack([X[i - 2], X[i - 1], X[i]], 1)
        preds.append(model.norm.inv(model.unroll(hist, D[i], steps))[:, :, TP].numpy())
    p = np.concatenate(preds)                                                  # [n,steps,H,W]
    t = np.stack([data[idx + k, TP] for k in range(1, steps + 1)], 1)
    scales = []
    for k in range(steps):
        tk = t[:, k] > thr; best, bs = -1.0, 1.0
        for s in np.arange(0.8, 2.51, 0.02):
            pk = p[:, k] * s > thr
            c = (pk & tk).sum() / max((pk | tk).sum(), 1)
            if c > best: best, bs = c, float(s)
        scales.append(bs)
    return scales


if __name__ == "__main__":
    main()
