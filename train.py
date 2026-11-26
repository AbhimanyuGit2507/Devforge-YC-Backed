"""Train one MonsoonNet run (one seed). Runs on CUDA, Apple MPS or CPU; fits its schedule into --minutes.

    python train.py --data data/india_climate_train.nc --out runs/run_s0 --seed 0 --minutes 150

Recipe: almost-fair CRPS over 2 noise-conditioned members (ensemble training), latitude weighting, extra weight on heavy
rain, spectral (anti-blur) loss, soft physics penalty, autoregressive rollout curriculum 1 -> 4 -> 10 days with
lead-decayed weights, input noise (pushforward trick), AdamW + time-based cosine LR, EMA weights.
Chronological split: train 2005-2018, validate 2019 (2020-2021 untouched test). --split all trains on every year.
"""
from __future__ import annotations
import argparse, copy, json, math, time
from pathlib import Path
import numpy as np
import torch
from core import data as D
from core.net import Twin, ZDIM, save_run
from core.losses import af_crps, spectral, physics, lat_weights

CH_W = torch.tensor([1.0, 1.5, 1.0, 1.0, 1.0, 1.0]).view(1, 1, 6, 1, 1)


def device_auto(name):
    if name != "auto": return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/india_climate_train.nc")
    ap.add_argument("--out", default="runs/run_s0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=150)
    ap.add_argument("--split", default="train", choices=["train", "all"])
    ap.add_argument("--width", type=int, default=128); ap.add_argument("--reach", type=int, default=4); ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--members", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--init", help="resume from a saved run directory (its net.pt)")
    ap.add_argument("--stages", default="1:0.35,4:0.40,10:0.25", help="rollout_days:share_of_time, comma separated")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    dev = device_auto(a.device); torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = dev.type == "cuda"
    print("device", dev, "amp", amp, flush=True)

    data, static, doy, year, _ = D.load(a.data)
    tr = D.year_mask(year, "train" if a.split == "train" else "all")
    mean, std = D.fit_norm(data, tr)
    xn = D.normalise(data, mean, std)
    clim_n = D.climatology(xn[tr], doy[tr])
    twin = Twin(clim_n, D.statics(static), mean.tolist(), std.tolist(), a.width, a.reach, a.depth).to(dev)
    if a.init:
        twin.net.load_state_dict(torch.load(Path(a.init) / "net.pt", map_location="cpu", weights_only=True))
    ema = copy.deepcopy(twin).eval()
    X = torch.from_numpy(xn).to(dev); DOY = torch.from_numpy(doy).long().to(dev)
    RAIN10 = torch.from_numpy(data[:, D.TP] > 10.0).to(dev)
    lw = lat_weights(dev); chw = CH_W.to(dev)
    tpm, tps = float(mean[D.TP]), float(std[D.TP])

    opt = torch.optim.AdamW(twin.net.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    budget = a.minutes * 60; t0 = time.time(); last_save = t0; it = 0
    cfg = {"width": a.width, "reach": a.reach, "depth": a.depth, "seed": a.seed, "split": a.split, "members": a.members}

    def lr_at(frac):
        return a.lr * (min(1.0, frac / 0.03) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))

    stage_end = 0.0
    for K, share in [(int(k), float(f)) for k, f in (x.split(":") for x in a.stages.split(","))]:
        stage_end += share
        starts = torch.from_numpy(D.window_starts(tr, K)).to(dev)
        if share <= 0: continue
        lead_w = torch.exp(-0.1 * torch.arange(K, device=dev, dtype=torch.float32)); lead_w = (lead_w / lead_w.mean()).view(1, 1, K, 1, 1, 1)
        while (time.time() - t0) < stage_end * budget:
            frac = (time.time() - t0) / budget
            for g in opt.param_groups: g["lr"] = lr_at(frac)
            B = a.batch if K <= 4 else max(4, a.batch * 4 // K)   # long rollouts: smaller batch keeps memory flat (T4 OOM at K=10)
            i = starts[torch.randint(len(starts), (B,), device=dev)]
            hist = torch.stack([X[i - 2], X[i - 1], X[i]], 1)
            hist = hist + 0.02 * torch.randn_like(hist)                      # pushforward-style input noise
            tgt = torch.stack([X[i + k] for k in range(1, K + 1)], 1)
            z = torch.randn(a.members, B, ZDIM, device=dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                M = a.members
                h = hist.repeat(M, 1, 1, 1, 1)
                pred = twin.unroll(h, DOY[i].repeat(M), K, z.reshape(M * B, ZDIM)).view(M, B, K, 6, 32, 36)
            pred = pred.float()
            heavy = torch.stack([RAIN10[i + k] for k in range(1, K + 1)], 1)[:, :, None].float()     # [B,K,1,H,W]
            w = lw * chw * (1 + 2 * heavy * (torch.arange(6, device=dev) == D.TP).view(1, 1, 6, 1, 1))
            loss_c = af_crps(pred * lead_w, tgt * lead_w[0], w)
            loss_s = spectral(pred, tgt)
            tp_phys = torch.expm1((pred[:, :, :, D.TP] * tps + tpm).clamp_max(8)).clamp_min(0)
            tw_phys = pred[:, :, :, 5] * float(std[5]) + float(mean[5])
            loss_p = physics(tp_phys, tw_phys)
            loss = loss_c + 0.05 * loss_s + 0.05 * loss_p
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(twin.net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); it += 1
            with torch.no_grad():
                d = min(0.999, (1 + it) / (10 + it))
                for pe, p in zip(ema.net.parameters(), twin.net.parameters()):
                    pe.mul_(d).add_(p.detach(), alpha=1 - d)
            if it % 100 == 0:
                print(f"it {it} K={K} {time.time() - t0:.0f}s lr {lr_at(frac):.2e} crps {loss_c.item():.4f} spec {loss_s.item():.4f} phys {loss_p.item():.4f}", flush=True)
            if time.time() - last_save > 600:
                save_run(ema, Path(a.out), {**cfg, "iters": it, "done": False}); last_save = time.time()
    save_run(ema, Path(a.out), {**cfg, "iters": it, "done": True})
    print(f"done {it} iters in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
