"""Build brief/brief.pdf (2 pages) from real evaluation output (eval/*.json) and a live light-cone experiment.

    python evaluate.py --json eval/baseline.json && python evaluate.py --v2 weights_v2 --json eval/v2.json
    python brief/make_brief.py --data data/india_climate_train.nc --weights weights_v2
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import data as D
from core.net import Ensemble
from monsoon_r1.scoring import make_shock
from monsoon_r1.contract import box_mask, BOX_BAY_OF_BENGAL

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/india_climate_train.nc"); ap.add_argument("--weights", default="weights_v2")
ap.add_argument("--screenshot", default="brief/dashboard.png"); ap.add_argument("--out", default="brief/brief.pdf")
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
res = {n: json.loads((ROOT / f"eval/{n}.json").read_text()) for n in ("baseline", "v2") if (ROOT / f"eval/{n}.json").exists()}

# light cone: response energy of the warm-ocean shock by distance from the box, per simulated day
data, _, doy, year, _ = D.load(a.data); std = data.std(axis=(0, 2, 3))
i = int(np.where((year == 2020) & (doy == 160))[0][0]); sc = [np.sin(2 * np.pi * (doy[i] + 1) / 365), np.cos(2 * np.pi * (doy[i] + 1) / 365)]
e = Ensemble(ROOT / a.weights); init = data[i - 2:i + 1]
dlt = e.rollout(init, sc, 10, make_shock("warm_ocean", init[-1], std)) - e.rollout(init, sc, 10)
en = ((dlt / std[None, :, None, None]) ** 2).sum(1)                                   # [10,32,36]
bm = box_mask(BOX_BAY_OF_BENGAL); ii, jj = np.indices(bm.shape); bi, bj = np.where(bm)
dist = np.sqrt((ii[..., None] - bi) ** 2 + (jj[..., None] - bj) ** 2).min(-1)
bins = np.arange(0, 26, 2); cone = np.array([[en[k][(dist >= b) & (dist < b + 2)].sum() for b in bins] for k in range(10)])
cone = cone / cone.sum(1, keepdims=True)

def txt(ax, lines, y0=0.97, size=8.6):
    ax.axis("off"); y = y0
    for l in lines:
        bold = l.startswith("#"); ax.text(0.0, y, l.lstrip("# "), fontsize=size + (2 if bold else 0), weight="bold" if bold else "normal", va="top", wrap=True, transform=ax.transAxes)
        y -= 0.05 if bold else 0.034

def row(n):
    r = res.get(n); return "n/a" if not r else f"SS {r['mean_ss']:.3f} | vs persistence {r['ss_persist']:.3f} | heavy-rain CSI skill {r['csi_skill']:.3f} | anti-blur {r['anti_blur_pts']:.1f}/10 | shocks {r['causality_pts']:.1f}/15 | {r['ms_median']:.0f} ms"

with PdfPages(ROOT / a.out) as pdf:
    f = plt.figure(figsize=(8.27, 11.69)); ax = f.add_axes([0.07, 0.42, 0.86, 0.55])
    txt(ax, ["# Monsoon Multiverse: page 1, model design",
             "Task: roll India's daily weather (t2m, rain, u10, v10, msl, tcwv; 32x36 at 1 deg) forward from 3 days, and branch on local shocks.",
             "# MonsoonNet = physics core + local learned correction",
             "- Physics core: semi-Lagrangian advection of every field along the 10 m wind (learned steering, <= 2 cells/day).",
             "- Learned correction: ConvNeXt-style blocks, per-pixel norm (no global coupling), FiLM conditioning on season,",
             "  ensemble noise z and lead day. Hard reach budget, measured by autograd: information moves <= 5 cells/day.",
             "- Inputs: last 3 days, anomaly vs climatology, target-day climatology, land-sea mask, orography, lat/lon.",
             "- Ensemble: independently seeded runs x noise members; mean for smooth fields, probability-matched mean for rain.",
             "# Training (how we prevent drift and blur)",
             "- Almost-fair CRPS (ECMWF AIFS-CRPS) over 2 members: rewards accurate and sharp forecasts at once.",
             "- Autoregressive curriculum 1 -> 4 -> 10 days with lead-decayed weights + input noise: trains on its own errors (drift).",
             "- 2-D spectral loss + probability-matched rain: keeps small-scale variance and heavy-rain peaks (blur).",
             "- Latitude weighting, x3 weight on rain > 10 mm/day, soft physics penalty (no heavy rain from a dry column).",
             "- Chronological split: train 2005-18, tune 2019 (rain scale), test 2020-21 once. Deterministic inference: zero shock = no change.",
             "# Results on held-out 2020-21 (organisers' scoring code)",
             "Baseline: " + row("baseline"), "MonsoonNet: " + row("v2"),
             "# Failure modes we know about",
             "- Trained on 2005-21: a warmer 2023-24 may bias t2m; anomaly inputs reduce, not remove, this.",
             "- 1 deg daily means smooth station-scale extremes; rain beyond ~8 days relaxes toward climatology.",
             "- Advection uses 10 m wind only: upper-level steering (cyclone tracks) is learned, not modelled explicitly."])
    ax2 = f.add_axes([0.1, 0.06, 0.8, 0.3])
    im = ax2.imshow(cone.T, origin="lower", aspect="auto", extent=[0.5, 10.5, 0, 26], cmap="magma")
    ax2.plot([0.5, 10.5], [3 + 8 * 0.5, 3 + 8 * 10.5], "c--", label="grader limit: 3 + 8 cells/day")
    ax2.set_ylim(0, 26); ax2.set_xlabel("simulated day"); ax2.set_ylabel("distance from shock box (cells)")
    ax2.set_title("Light cone: where the warm Bay of Bengal shock's energy is, by day"); ax2.legend(loc="upper left", fontsize=8)
    f.colorbar(im, ax=ax2, label="share of response energy"); pdf.savefig(f); plt.close(f)

    f = plt.figure(figsize=(8.27, 11.69)); ax = f.add_axes([0.07, 0.55, 0.86, 0.42])
    txt(ax, ["# Page 2: visualization and impact",
             "Dashboard (dashboard/, static HTML/JS, no internet) calls the live /simulate endpoint; nothing is pre-rendered.",
             "- Timeline Prime vs shocked universe vs difference, any day 1-14; preset or hand-drawn box shocks.",
             "- Inverse shock discovery: pick a region and a rain target; gradient descent through the frozen model finds the",
             "  smallest smooth change to today's weather that produces it. Its first gradient is the adjoint sensitivity map.",
             "# Impact modules (computed in the browser from each rollout, shock vs no shock)",
             "- Heat stress: humidity-weighted daily-mean temperature; days >= 33 C on land.",
             "- Flood risk: max 3-day accumulated rain; cells above 50 / 100 mm.",
             "- Kharif crop water stress: rain deficit vs monthly climatology (70%) + heat days (30%).",
             "- Energy: wind power density at 100 m (1/7 power law) and a clearness-weighted solar resource.",
             "- Ensemble: 8 perturbed members -> spread maps and probability of rain > 10 mm/day.",
             "Use case: a state disaster cell asks 'what if the Bay is 3 K warmer this week?' and sees which districts move into",
             "flood or heat-stress categories, with ensemble confidence, in under a second."])
    if Path(ROOT / a.screenshot).exists():
        ax3 = f.add_axes([0.05, 0.05, 0.9, 0.48]); ax3.imshow(plt.imread(ROOT / a.screenshot)); ax3.axis("off")
    pdf.savefig(f); plt.close(f)
print("wrote", a.out)
