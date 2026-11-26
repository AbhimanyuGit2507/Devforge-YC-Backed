"""MonsoonNet: hybrid physics + ML one-day step, and the ensemble wrapper used for serving.

step(x) = advect(x_t, wind_t) + learned_local_correction(...)
 * Physics core: semi-Lagrangian advection of every field along the 10 m wind (bilinear resampling), with a learned
   per-channel steering factor; displacement is capped at MAX_SHIFT cells/day.
 * Learned correction: ConvNeXt-style blocks with a hard spatial reach budget (each 3x3 depthwise conv adds 1 cell;
   total REACH cells/day). Season, ensemble-member noise z and lead day enter every block through FiLM (AdaLN-style).
 * Information therefore travels at most MAX_SHIFT + REACH cells per simulated day: shocks start local and spread
   outward, never teleport.
All parameters are real float32 (portable across CPU / CUDA / MPS); checkpoints are CPU float32 state_dicts.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NC, H, W, TP = 6, 32, 36, 1
LAT = np.arange(6.5, 37.6, 1.0)
LON = np.arange(65.5, 100.6, 1.0)
ZDIM = 16
MAX_SHIFT = 2.0
DEG_M = 111_000.0


def doy_index(sincos) -> int:
    """[sin, cos] of 2*pi*doy/365 for the last input day -> 0..364 climatology slot (doy - 1)."""
    ang = float(np.arctan2(sincos[0], sincos[1])) % (2 * np.pi)
    return int(round(ang * 365 / (2 * np.pi)) - 1) % 365


class FiLMBlock(nn.Module):
    def __init__(self, c, cond, spatial: bool):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c) if spatial else nn.Identity()
        self.film = nn.Linear(cond, 2 * c)
        self.mlp = nn.Sequential(nn.Conv2d(c, 2 * c, 1), nn.GELU(), nn.Conv2d(2 * c, c, 1))
        self.gamma = nn.Parameter(torch.full((1, c, 1, 1), 0.1))
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        s, b = self.film(cond)[:, :, None, None].chunk(2, 1)
        h = self.dw(x)
        h = F.layer_norm(h.movedim(1, -1), (h.shape[1],)).movedim(-1, 1) * (1 + s) + b  # per-pixel norm: no global coupling
        return x + self.gamma * self.mlp(h)


class MonsoonNet(nn.Module):
    N_IN = 3 * NC + NC + NC + 2 + 2 + 1   # 3 days, anomaly x_t-clim_t, clim_{t+1}, statics, lat/lon, advected tp

    def __init__(self, width=128, reach=4, depth=8):
        super().__init__()
        cond = 2 + ZDIM + 1
        self.cond = nn.Sequential(nn.Linear(cond, 128), nn.GELU(), nn.Linear(128, 128))
        self.stem = nn.Conv2d(self.N_IN + NC, width, 3, padding=1)               # +NC: advected state
        spatial = [i < reach - 2 for i in range(depth)]                           # stem(1) + (reach-2) dw + head(1)
        self.blocks = nn.ModuleList([FiLMBlock(width, 128, s) for s in spatial])
        self.head = nn.Conv2d(width, NC, 3, padding=1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.steer = nn.Parameter(torch.full((NC,), 0.3))                         # learned advection strength
        self.reach = reach

    def advect(self, x, u_ms, v_ms):
        """Semi-Lagrangian: value at (i,j) tomorrow comes from upstream point (i,j) - steer*wind*1day."""
        B = x.shape[0]
        lat = torch.as_tensor(LAT, dtype=x.dtype, device=x.device).view(1, H, 1)
        dx = (u_ms * 86400 / (DEG_M * torch.cos(torch.deg2rad(lat))))            # cells/day east
        dy = (v_ms * 86400 / DEG_M)                                                # cells/day north
        out = []
        for c in range(NC):
            sx = (self.steer[c] * dx).clamp(-MAX_SHIFT, MAX_SHIFT); sy = (self.steer[c] * dy).clamp(-MAX_SHIFT, MAX_SHIFT)
            jj = (torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, W) - sx).clamp(0, W - 1)  # clamp = border padding (MPS lacks it)
            ii = (torch.arange(H, device=x.device, dtype=x.dtype).view(1, H, 1) - sy).clamp(0, H - 1)
            grid = torch.stack([2 * jj / (W - 1) - 1, 2 * ii / (H - 1) - 1], -1)
            out.append(F.grid_sample(x[:, c:c + 1], grid, mode="bilinear", padding_mode="zeros", align_corners=True))
        return torch.cat(out, 1)

    def forward(self, hist, clim_next, clim_now, aux, cond, wind_ms):
        """hist [B,3,6,H,W] normalised; clim_* [B,6,H,W]; aux [B,4,H,W]; cond [B,cond]; wind_ms [B,2,H,W] physical."""
        B = hist.shape[0]; xt = hist[:, -1]
        adv = self.advect(xt, wind_ms[:, 0], wind_ms[:, 1])
        x = torch.cat([hist.reshape(B, 3 * NC, H, W), xt - clim_now, clim_next, aux, adv[:, TP:TP + 1], adv], 1)
        c = self.cond(cond)
        h = F.gelu(self.stem(x))
        for blk in self.blocks:
            h = blk(h, c)
        return adv + self.head(F.gelu(h))


class Twin(nn.Module):
    """One trained MonsoonNet plus everything it conditions on. Works in normalised space internally."""

    def __init__(self, clim_n, static_n, mean, std, width=128, reach=4, depth=8):
        super().__init__()
        self.net = MonsoonNet(width, reach, depth)
        self.register_buffer("clim", torch.as_tensor(clim_n, dtype=torch.float32))
        la, lo = np.meshgrid((LAT - LAT.mean()) / 10, (LON - LON.mean()) / 10, indexing="ij")
        self.register_buffer("aux", torch.as_tensor(np.concatenate([static_n, la[None], lo[None]]).astype("f4")))
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, NC, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, NC, 1, 1))

    def wind(self, xn):  # physical u10/v10 from normalised state
        return xn[:, 2:4] * self.std[:, 2:4] + self.mean[:, 2:4]

    def step(self, hist, doy_next, z, lead):
        B = hist.shape[0]
        ang = 2 * np.pi * (doy_next.float() + 1) / 365
        cond = torch.cat([torch.sin(ang)[:, None], torch.cos(ang)[:, None], z, lead.float()[:, None] / 10], 1)
        return self.net(hist, self.clim[doy_next], self.clim[(doy_next - 1) % 365], self.aux.expand(B, -1, -1, -1), cond, self.wind(hist[:, -1]))

    def unroll(self, hist, doy_last, steps, z):
        outs = []
        for k in range(steps):
            nxt = self.step(hist, (doy_last + k + 1) % 365, z, torch.full_like(doy_last, k))
            outs.append(nxt); hist = torch.cat([hist[:, 1:], nxt[:, None]], 1)
        return torch.stack(outs, 1)

    def to_norm(self, phys):  # out-of-place so gradients flow (inverse shock search)
        x = torch.cat([phys[..., :TP, :, :], torch.log1p(phys[..., TP:TP + 1, :, :].clamp_min(0)), phys[..., TP + 1:, :, :]], -3)
        return (x - self.mean) / self.std

    def to_phys(self, z):
        x = z * self.std + self.mean
        return torch.cat([x[..., :TP, :, :], torch.expm1(x[..., TP:TP + 1, :, :].clamp_max(8.0)).clamp_min(0), x[..., TP + 1:, :, :]], -3)


def member_noise(n, seed=1234):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, ZDIM, generator=g); z[0] = 0  # member 0 is the noise-free "control" run
    return z


def save_run(twin: Twin, out: Path, cfg: dict):
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().float() for k, v in twin.net.state_dict().items()}, out / "net.pt")
    np.save(out / "climatology_norm.npy", twin.clim.cpu().numpy()); np.save(out / "aux.npy", twin.aux.cpu().numpy())
    (out / "config.json").write_text(json.dumps({**cfg, "mean": twin.mean.flatten().tolist(), "std": twin.std.flatten().tolist()}, indent=1))


def load_run(d: Path) -> Twin:
    cfg = json.loads((d / "config.json").read_text())
    aux = np.load(d / "aux.npy")
    t = Twin(np.load(d / "climatology_norm.npy"), aux[:2], cfg["mean"], cfg["std"], cfg["width"], cfg["reach"], cfg["depth"])
    t.net.load_state_dict(torch.load(d / "net.pt", map_location="cpu", weights_only=True))
    return t.eval()


class Ensemble:
    """Several trained Twins x fixed noise members. Deterministic: the same input always gives the same output.
    Continuous fields: ensemble mean (+ per-lead post-processing). Rain: probability-matched mean (keeps peaks)."""

    def __init__(self, root: Path, members=None):
        root = Path(root)
        self.runs = [load_run(d) for d in sorted(root.glob("run_*"))]
        pp = root / "postproc.json"
        self.pp = json.loads(pp.read_text()) if pp.exists() else None
        # CRPS-trained members are individually noisy: the mean needs several of them (set in postproc.json)
        self.z = member_noise(members or (self.pp or {}).get("members", 8))

    @torch.no_grad()
    def members(self, init_states, doy_sincos, steps, perturbation=None):
        """-> physical [R*M, steps, 6, H, W]"""
        init = np.array(init_states, dtype="f4", copy=True)
        if perturbation is not None:
            init[-1] += np.asarray(perturbation, dtype="f4")
        init[:, TP] = np.maximum(init[:, TP], 0)
        doy = doy_index(doy_sincos); M = len(self.z); outs = []
        for r in self.runs:
            hist = r.to_norm(torch.from_numpy(init)).unsqueeze(0).expand(M, -1, -1, -1, -1)
            outs.append(r.to_phys(r.unroll(hist, torch.full((M,), doy), steps, self.z)))
        return torch.cat(outs, 0)

    def clim_phys(self, doy_sincos, steps):
        """Physical climatology of each forecast day [steps,6,H,W] (from the first run's training-years climatology)."""
        r = self.runs[0]; d0 = doy_index(doy_sincos)
        return r.to_phys(r.clim[[(d0 + k + 1) % 365 for k in range(steps)]])

    def rollout(self, init_states, doy_sincos, steps, perturbation=None) -> np.ndarray:
        ens = self.members(init_states, doy_sincos, steps, perturbation)
        out = ens.mean(0)
        if self.pp and "blend_a" in self.pp:   # per-lead, per-channel linear blend of ensemble, baseline model and climatology
            out = self.blend(out, init_states, doy_sincos, steps, perturbation)
        if (self.pp or {}).get("prob_matched", False):   # sharper rain, but rank re-sorting can jump under small shocks
            out[:, TP] = prob_matched_mean(ens[:, :, TP])
        if self.pp:
            k = torch.arange(steps).clamp_max(len(self.pp["tp_scale"]) - 1)
            out[:, TP] *= torch.tensor(self.pp["tp_scale"])[k][:, None, None]
        out[:, 0] = out[:, 0].clamp(240.0, 330.0); out[:, TP] = out[:, TP].clamp_min(0)
        return np.nan_to_num(out.numpy(), nan=0.0, posinf=0.0, neginf=0.0).astype("f4")


def _blend(self, out, init_states, doy_sincos, steps, perturbation):
    """pred = clim + a*(ensemble - clim) + b*(baseline - clim) for every channel except rain; a, b fitted on 2019."""
    import model as baseline                                     # first model (repo root), loaded lazily once
    if not hasattr(self, "base"):
        self.base = baseline.load(Path(__file__).resolve().parents[1] / "weights")
    B = torch.from_numpy(self.base.rollout(init_states, doy_sincos, steps, perturbation))
    C = self.clim_phys(doy_sincos, steps)
    k = torch.arange(steps).clamp_max(len(self.pp["blend_a"]) - 1)
    a = torch.tensor(self.pp["blend_a"])[k][:, :, None, None]; b = torch.tensor(self.pp["blend_b"])[k][:, :, None, None]
    mixed = C + a * (out - C) + b * (B - C)
    keep = torch.zeros(1, NC, 1, 1, dtype=torch.bool); keep[0, TP] = True
    return torch.where(keep, out, mixed)


Ensemble.blend = _blend


def prob_matched_mean(r):
    """r [M,steps,H,W] -> [steps,H,W]: spatial ranking of the ensemble mean, values from the pooled member distribution."""
    M, S = r.shape[:2]; out = torch.empty(S, H * W)
    for k in range(S):
        mean = r[:, k].mean(0).flatten()
        pooled = r[:, k].flatten().sort().values[M - 1::M]      # every M-th value -> H*W quantiles
        out[k, mean.argsort()] = pooled
    return out.view(S, H, W)
