"""Build brief/brief.pdf (2 A4 pages) from real evaluation output and a live light-cone experiment.

    python evaluate.py --json eval/baseline.json && python evaluate.py --v2 weights_v2 --json eval/v2.json
    python eval/diagnostics.py && python brief/make_brief.py
Page 1: model design, drift/blur prevention, results, failure modes. Page 2: visualization and impact.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from core import data as D
from core.net import Ensemble
from monsoon_r1.scoring import make_shock
from monsoon_r1.contract import box_mask, BOX_BAY_OF_BENGAL

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/india_climate_train.nc"); ap.add_argument("--weights", default="weights_v2")
ap.add_argument("--out", default="brief/brief.pdf"); a = ap.parse_args()
J = lambda n: json.loads((ROOT / f"eval/{n}.json").read_text())
base, v2, dg = J("baseline"), J("v2"), J("diagnostics")
INK, MUTE, ACC, ACC2, WARN, GRID = "#1b2733", "#5b6b78", "#1f77b4", "#d9822b", "#b03a2e", "#d9dee3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2, "axes.edgecolor": MUTE, "axes.labelcolor": INK,
                     "xtick.color": MUTE, "ytick.color": MUTE, "axes.spines.top": False, "axes.spines.right": False})

# ---- light cone: response energy of the warm-ocean shock by distance from the box, per simulated day ----
data, _, doy, year, _ = D.load(ROOT / a.data); std = data.std(axis=(0, 2, 3))
i = int(np.where((year == 2020) & (doy == 160))[0][0]); sc = [np.sin(2 * np.pi * (doy[i] + 1) / 365), np.cos(2 * np.pi * (doy[i] + 1) / 365)]
e = Ensemble(ROOT / a.weights); init = data[i - 2:i + 1]
dlt = e.rollout(init, sc, 10, make_shock("warm_ocean", init[-1], std)) - e.rollout(init, sc, 10)
en = ((dlt / std[None, :, None, None]) ** 2).sum(1)
bm = box_mask(BOX_BAY_OF_BENGAL); ii, jj = np.indices(bm.shape); bi, bj = np.where(bm)
dist = np.sqrt((ii[..., None] - bi) ** 2 + (jj[..., None] - bj) ** 2).min(-1)
bins = np.arange(0, 26, 1); cone = np.array([[en[k][(dist >= b) & (dist < b + 1)].sum() for b in bins] for k in range(10)])
cone = cone / cone.sum(1, keepdims=True)


def text(fig, x, y, s, size=7.2, color=INK, weight="normal", **kw):
    fig.text(x, y, s, fontsize=size, color=color, weight=weight, va="top", **kw)


def para(fig, x, y, lines, size=7.0, gap=0.0128):
    for l in lines:
        bold = l.startswith("**"); l = l.strip("*")
        text(fig, x, y, l, size, INK, "bold" if bold else "normal"); y -= gap
    return y


def head(fig, x, y, s):
    text(fig, x, y, s.upper(), 7.6, ACC, "bold"); fig.add_artist(plt.Line2D([x, x + 0.43], [y - 0.014, y - 0.014], color=GRID, lw=0.8))
    return y - 0.022


def pts(r):
    return (30 * min(r["mean_ss"] / 0.12, 1)) + (15 * min(r["csi_skill"] / 0.10, 1)) + r["anti_blur_pts"] + r["causality_pts"] + 15


def diagram(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0.5, 4.05); ax.axis("off")
    def box(x, y, w, h, s, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=MUTE, lw=0.6))
        ax.text(x + w / 2, y + h / 2, s, ha="center", va="center", fontsize=6.3, color=INK)
    def arrow(x0, y0, x1, y1):
        ax.add_artist(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=MUTE, lw=0.7))
    box(0.0, 2.55, 2.3, 1.45, "Inputs (normalised)\nlast 3 days x 6 vars\nanomaly vs climatology\nclimatology of target day\nland-sea, orography, lat/lon", "#eef3f8")
    box(3.0, 3.05, 3.0, 0.95, "Physics core\nsemi-Lagrangian advection along\n10 m wind (learned steering, <= 2 cells/day)", "#fdf1e6")
    box(3.0, 1.75, 3.0, 1.05, "Local learned correction\n3x3 stem + 8 FiLM blocks + 3x3 head\nper-pixel norm, reach <= 5 cells/day\nFiLM: season, noise z, lead day", "#eaf4ea")
    box(6.7, 2.35, 1.2, 0.95, "x(t+1) =\nadvected\n+ correction", "#f4f4f4")
    box(8.3, 2.35, 1.7, 0.95, "autoregressive\nrollout\n1 ... 14 days", "#f4f4f4")
    arrow(2.3, 3.3, 3.0, 3.5); arrow(2.3, 3.0, 3.0, 2.3); arrow(6.0, 3.5, 6.7, 3.0); arrow(6.0, 2.3, 6.7, 2.7); arrow(7.9, 2.8, 8.3, 2.8)
    ax.text(9.15, 2.15, "feeds back as input", ha="center", va="top", fontsize=5.8, color=MUTE, style="italic")
    box(0.0, 0.55, 10.0, 0.95, "Serving: 3 seeds x 8 fixed noise members, averaged (deterministic: zero shock = identical output)  ->  per-lead blend\n"
        "clim + a(ensemble - clim) + b(baseline - clim), fitted on 2019  ->  rain >= 0, t2m clamped, NaN-safe  ->  1.3-2.8 s per 10-day call on 4 cores\n"
        "speed guard: times this CPU at startup and drops noise members to fit 2.5 s (all 8 on 4 cores; < 2 s per 10-day call even on 1 core)", "#eef3f8")


with PdfPages(ROOT / a.out) as pdf:
    # ================================= PAGE 1 =================================
    f = plt.figure(figsize=(8.27, 11.69))
    text(f, 0.05, 0.975, "Monsoon Multiverse", 17, INK, "bold")
    text(f, 0.95, 0.975, "Team: YC backed", 11, ACC, "bold", ha="right")
    text(f, 0.05, 0.952, "A physics-aware, causal digital twin of India's daily weather  ·  Track 2  ·  page 1: model design, drift & blur, limits", 8.2, MUTE)
    y = head(f, 0.05, 0.925, "What we built")
    y = para(f, 0.05, y, [
        "From the last 3 days of India's state (t2m, rain, u10, v10, msl, tcwv; 32 x 36 cells at 1 deg) the twin rolls the weather",
        "forward day by day, and branches alternate universes when a local shock is applied. MonsoonNet = physics + local ML:"])
    ax = f.add_axes([0.05, 0.70, 0.9, 0.175]); diagram(ax)

    y = head(f, 0.05, 0.68, "How we prevent drift and blur")
    para(f, 0.05, y, [
        "**Drift (errors compounding over the rollout)",
        "- Curriculum 1 -> 4 -> 10-day rollouts, backprop through the",
        "  whole chain, lead-decayed weights: trained on its own errors.",
        "- Input noise (pushforward trick); EMA weights; residual step.",
        "- Blend toward baseline + climatology at long leads (fit on 2019).",
        "**Blur (MSE 'average' forecasts)",
        "- Almost-fair CRPS over noise-conditioned members (ECMWF",
        "  AIFS-CRPS idea): members stay sharp, their mean is accurate.",
        "- 2-D spectral loss keeps small-scale variance; x3 weight on",
        "  rain > 10 mm/day; per-lead rain scale for heavy-rain hits.",
        "- Day-8 variance ratio 0.80-1.00 of the real field (need 0.4-2.5).",
        "**Causality by construction",
        "- Local reach measured with autograd: <= 5 cells per day.",
        "- Per-pixel norm (a global GroupNorm leaked; caught & fixed).",
    ], size=6.8, gap=0.0118)
    y2 = head(f, 0.52, 0.68, "Results on held-out 2020-21 (organisers' scoring code)")
    rows = [("", "Baseline CNN", "Submitted"), ("Forecast skill (full at 0.12)", f"{base['mean_ss']:.3f}", f"{v2['mean_ss']:.3f}"),
            ("Heavy-rain CSI skill (full at 0.10)", f"{base['csi_skill']:.3f}", f"{v2['csi_skill']:.3f}"),
            ("Anti-blur (10)", f"{base['anti_blur_pts']:.1f}", f"{v2['anti_blur_pts']:.1f}"),
            ("Shock causality (15)", f"{base['causality_pts']:.1f}", f"{v2['causality_pts']:.1f}"),
            ("Automated points (/85, repo 15 incl.)", f"{pts(base):.1f}", f"{pts(v2):.1f}"), ("10-day latency, 4 CPU cores", f"{base['ms_median']:.0f} ms", f"{v2['ms_median'] / 1000:.1f} s")]
    for r, (k, b, s) in enumerate(rows):
        yy = y2 - r * 0.0135; w = "bold" if r in (0, 5) else "normal"
        text(f, 0.52, yy, k, 6.8, INK, w); text(f, 0.80, yy, b, 6.8, MUTE, w); text(f, 0.895, yy, s, 6.8, ACC if r else INK, "bold")
    text(f, 0.52, y2 - 7 * 0.0135, f"{v2['windows']} windows + 17 shock windows · tuning on 2019 only · zero shock = 0 change", 6.2, MUTE)

    L = np.arange(1, 11)
    ax1 = f.add_axes([0.56, 0.355, 0.39, 0.14])
    ax1.axhline(0, color=GRID, lw=0.8); ax1.plot(L, dg["lead_skill"]["model"], "-o", ms=2.5, color=ACC, label="submitted")
    ax1.plot(L, dg["lead_skill"]["baseline"], "-o", ms=2.5, color=ACC2, label="baseline CNN")
    ax1.set_title("RMSE skill vs the better of persistence / climatology", fontsize=7, color=INK, loc="left")
    ax1.set_xlabel("forecast day", fontsize=6.5); ax1.legend(frameon=False, fontsize=6); ax1.set_xticks(L)
    ax2 = f.add_axes([0.56, 0.175, 0.39, 0.13])
    for k, c, lab in (("model", ACC, "submitted"), ("persistence", MUTE, "persistence"), ("climatology", GRID, "climatology")):
        ax2.plot(L, dg["lead_csi"][k], "-o", ms=2.3, color=c if k != "climatology" else "#9aa7b2", label=lab)
    ax2.set_title("Heavy-rain CSI (> 10 mm/day)", fontsize=7, color=INK, loc="left"); ax2.set_xlabel("forecast day", fontsize=6.5)
    ax2.legend(frameon=False, fontsize=6, ncol=3); ax2.set_xticks(L)

    ax3 = f.add_axes([0.08, 0.19, 0.36, 0.24])
    im = ax3.imshow(cone.T, origin="lower", aspect="auto", extent=[0.5, 10.5, 0, 26], cmap="magma")
    ax3.plot([0.5, 10.5], [3 + 8 * 0.5, 3 + 8 * 10.5], color="#4fd1ff", ls="--", lw=1, label="grader limit 3 + 8 cells/day")
    ax3.set_ylim(0, 26); ax3.set_xlabel("simulated day", fontsize=6.5); ax3.set_ylabel("cells from shock box", fontsize=6.5)
    ax3.set_title("Light cone: warm Bay of Bengal shock energy by distance", fontsize=7, color=INK, loc="left")
    ax3.legend(frameon=False, fontsize=6, loc="upper left", labelcolor="white"); f.colorbar(im, ax=ax3, fraction=0.04, pad=0.02)

    y = head(f, 0.05, 0.135, "Failure modes we know about")
    para(f, 0.05, y, [
        "- Learns correlation, not causation: a wind-only Arabian-Sea stall raises offshore rain (seen in data with storms); a physically consistent stall",
        "  (-40 % wind, -20 % moisture, +2 hPa) cuts west-coast rain by 4-6 mm/day as expected - the dashboard preset uses this coupled version.",
        "- Peak rain under-forecast in convective events (Kerala onset 2020: 26 vs 102 mm/day max; Mumbai Aug 2020: 58 vs 145) - 1 deg daily means smooth cloudbursts.",
        "- Trained on 2005-21; warmer 2023-24 may bias t2m (anomaly inputs + climatology blend reduce, not remove, it).",
        "- Only 10 m wind drives advection: cyclone steering by upper-level winds is learned indirectly (Amphan rain over-forecast 151 vs 91).",
        "- Skill fades toward climatology after ~8 days (expected for daily weather); impact modules are transparent proxies, not warnings.",
        "- 2020-21 was consulted for the model gate; all later choices used 2019 only. The real test (2023-24 extremes) is unseen.",
    ], size=6.7, gap=0.0125)
    pdf.savefig(f); plt.close(f)

    # ================================= PAGE 2 =================================
    f = plt.figure(figsize=(8.27, 11.69))
    text(f, 0.05, 0.975, "Monsoon Multiverse", 17, INK, "bold")
    text(f, 0.95, 0.975, "Team: YC backed", 11, ACC, "bold", ha="right")
    text(f, 0.05, 0.952, "page 2: visualization, multiverse shocks and impact  ·  dashboard drives the live server - nothing is pre-rendered", 8.2, MUTE)
    for k, (shot, cap, yb) in enumerate([("dashboard.png", "Timeline Prime vs warm-Bay universe vs difference (rain, day 4). The shock stays local, then spreads. Starting days include Amphan, Nisarga, Tauktae and Yaas.", 0.665),
                                          ("dashboard_inverse.png", "Inverse shock discovery: smallest smooth change today that brings Mumbai-box rain to 40 mm on day 3, plus the sensitivity map.", 0.39)]):
        axs = f.add_axes([0.05, yb, 0.9, 0.255]); axs.imshow(plt.imread(ROOT / "brief" / shot)); axs.axis("off")
        text(f, 0.05, yb - 0.004, cap, 6.6, MUTE)
    y = head(f, 0.05, 0.365, "Impact modules (prime vs shocked)")
    para(f, 0.05, y, [
        "- Heat stress: NWS heat index (feels-like) from T and humidity (dew point",
        "  = tcwv / 2.5 km scale height, Magnus); 32-41 C caution, >= 41 C danger.",
        "- Flood risk: 3-day rain / each cell's own 99th pct (2005-21), land only.",
        "- Crop water stress: 0.7 x rain deficit vs daily-interpolated normal",
        "  + 0.3 x hot days (daily mean >= 32 C); try the monsoon-stall shock.",
        "- Energy (indicative): 100 m wind (shear 0.2 land / 0.11 sea), power",
        "  x1.91 Weibull factor; clearness-weighted solar; % change from shock.",
        "- Uncertainty: the trained 24-member ensemble, spread calibrated on 2019",
        "  (spread/error 0.89-1.05 on unseen 2020-21) -> +/- std, P(rain > 10 mm).",
        "- Shocks: one-day kick (official test) or held every day (warm sea);",
        "  monsoon stall preset = -40 % wind, -20 % moisture, +2 hPa (consistent).",
        "- Inverse search: gradient descent on today's state (weights frozen),",
        "  capped at 5 K / 5 m/s / 10 kg m-2; first gradient = adjoint map.",
        "**Use case: a state disaster cell asks 'what if the Bay is 3 K warmer",
        "**this week?' and sees which districts move into flood or heat-stress",
        "**categories, with ensemble confidence, in about a second, offline.",
    ], size=6.8, gap=0.0123)
    y2 = head(f, 0.52, 0.365, "Is it good only on average? (2020-21, unseen)")
    seasons = dg["season_skill"]; axs = f.add_axes([0.70, 0.235, 0.25, 0.085])
    names = list(seasons); vals = [seasons[s][0] for s in names]
    axs.barh(range(len(names)), vals, color=ACC, height=0.55); axs.set_yticks(range(len(names))); axs.set_yticklabels([n.replace(" (", " ").rstrip(")") for n in names], fontsize=6)
    axs.axvline(0.12, color=WARN, lw=0.8, ls="--"); axs.set_xlim(0, 0.2); axs.set_title("Skill by season; dashed = full points", fontsize=6.8, loc="left", color=INK)
    ev = dg["events"]; yy = 0.205
    text(f, 0.52, yy, "Named extremes (forecast from 2 days before)", 6.8, INK, "bold"); text(f, 0.85, yy, "skill", 6.8, INK, "bold"); text(f, 0.905, yy, "peak rain", 6.4, INK, "bold")
    for n, r in ev.items():
        yy -= 0.0122; n = n.replace("Western disturbance", "Winter window")
        text(f, 0.52, yy, n, 6.3); text(f, 0.85, yy, f"{r['skill_d1_4_8']:+.2f}", 6.3, ACC, "bold"); text(f, 0.905, yy, f"{r['rain_max_model_d2']:.0f} / {r['rain_max_true_d2']:.0f}", 6.3, MUTE)
    text(f, 0.52, yy - 0.013, "Positive skill on all 8 events; peak rain = forecast / observed max (mm/day, day 2).", 6.1, MUTE)
    y = head(f, 0.05, 0.085, "Reproduce")
    para(f, 0.05, y, ["python serve.py --port 8000   ·   python dashboard/serve_dashboard.py   ·   python train.py --seed 1 --minutes 150",
                      "python evaluate.py --v2 weights_v2 --years test   ·   python eval/diagnostics.py   ·   details: README.md, docs/DEMO.md",
                      "Data: ERA5 (Copernicus C3S, Hersbach et al. 2020) via WeatherBench 2 (Rasp et al. 2024), CC BY 4.0, as provided in the organisers' kit."], size=6.6)
    pdf.savefig(f); plt.close(f)
print("wrote", a.out)
