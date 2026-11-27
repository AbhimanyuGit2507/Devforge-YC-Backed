"""Cross-check the dashboard's map formulas (run in Node from dashboard/app.js) against an independent NumPy
implementation and published reference values. Needs /tmp/monsoon_rollouts.json from tests/scenarios.py.

    python tests/scenarios.py && python tests/check_maps.py
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "dashboard/samples.json").read_text()); R = json.loads(Path("/tmp/monsoon_rollouts.json").read_text())
js = json.loads(subprocess.run(["node", str(ROOT / "tests/maps.js"), str(ROOT / "dashboard/samples.json"), "/tmp/monsoon_rollouts.json"],
                               capture_output=True, text=True, check=True).stdout)
land = np.array(S["land"]).ravel() >= 0.5; fails = []   # same land rule as the dashboard
def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}"); (None if ok else fails.append(name))

# reference values: NWS heat index table (deg F) for 90F/60%=100, 100F/40%=109, 80F/40%=80 (Steadman), 95F/70%=124
for (t, rh, ref), v in zip([(90, 60, 100), (100, 40, 109), (80, 40, 80), (95, 70, 124)], js["hi"]):
    f = v * 9 / 5 + 32; check(f"heat index {t}F/{rh}% vs NWS table {ref}F", abs(f - ref) <= 2, f"{f:.1f}F")

p95, p99 = np.array(S["flood_p95"]).ravel(), np.array(S["flood_p99"]).ravel(); clim = np.array(S["tp_clim_month"]).reshape(12, -1)
for s in S["samples"]:
    r = np.array(R[s["name"]]).reshape(10, 6, -1); j = {k: np.array(v, float) for k, v in js[s["name"]].items()}; n = s["name"][:22]
    # flood: max 3-day running sum on land / per-cell 99th percentile
    cs = np.array([r[max(0, k - 2):k + 1, 1].sum(0) for k in range(10)]).max(0)            # mm; the map then divides by p99
    check(f"{n}: flood 3-day max (mm) == independent, sea masked", np.allclose(np.nan_to_num(j["flood"], nan=-1), np.where(land, cs, -1), atol=1e-2),
          f"index range {np.nanmin(j['flood'] / p99):.2f}-{np.nanmax(j['flood'] / p99):.2f}")
    # crop: 0.7 * deficit vs daily-interpolated monthly normal + 0.3 * min(1, 2 * hot-day share)
    d0 = round(np.arctan2(*s["doy_sincos"]) / 2 / np.pi * 365 + 365) % 365; nor = np.zeros(land.size)
    for k in range(10):
        m = ((d0 + k + 1) % 365 / 365 * 12 - 0.5 + 12) % 12; m0 = int(np.floor(m)); fr = m - m0; nor += (1 - fr) * clim[m0] + fr * clim[(m0 + 1) % 12]
    dfc = np.where(nor > 10, np.maximum(0, 1 - r[:, 1].sum(0) / np.where(nor > 0, nor, 1)), 0); hot = ((r[:, 0] - 273.15) >= 32).sum(0)
    ref = 0.7 * dfc + 0.3 * np.minimum(1, hot / 10 * 2)
    check(f"{n}: crop stress == independent", np.allclose(np.nan_to_num(j["crop"], nan=-1), np.where(land, ref, -1), atol=1e-3))
    # wind: 100 m speed with shear 0.2 land / 0.11 sea, PD = 1.91 * 0.5 * 1.225 * v^3
    v100 = np.hypot(r[:, 2], r[:, 3]) * np.where(land, 10 ** 0.2, 10 ** 0.11)
    check(f"{n}: wind speed & power == independent", np.allclose(j["speed"], v100.mean(0), rtol=1e-4) and np.allclose(j["wind"], (1.91 * 0.5 * 1.225 * v100 ** 3).mean(0), rtol=1e-4))
    check(f"{n}: wind speed in 0-40 m/s, solar in 0-500 W/m2", j["speed"].min() >= 0 and j["speed"].max() < 40 and 0 <= j["sun"].min() and j["sun"].max() < 500,
          f"speed max {j['speed'].max():.1f}, solar {j['sun'].min():.0f}-{j['sun'].max():.0f}")
    hl = j["heat"][land]; check(f"{n}: heat index on land within -45..60 C (cold air: feels-like = T), sea masked", np.isfinite(hl).all() and -45 < hl.min() and hl.max() < 60 and np.isnan(j["heat"][~land]).all(),
                                  f"{hl.min():.1f}..{hl.max():.1f} C")
    cl = j["crop"][land]; check(f"{n}: crop stress in [0,1], sea masked", (cl >= 0).all() and (cl <= 1).all() and np.isnan(j["crop"][~land]).all(), f"mean {cl.mean():.2f}")

# solar: top-of-atmosphere daily mean insolation, reference values (W/m2): equator equinox ~435, 30N June ~480, 30N December ~230
def toa(phi, doy):
    dec = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365); phi = np.deg2rad(phi); ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    return 1361 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365)) / np.pi * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
for phi, doy, ref in [(0, 80, 435), (30, 172, 480), (30, 355, 230)]:
    check(f"TOA insolation {phi}N day {doy} vs reference {ref}", abs(toa(phi, doy) - ref) / ref < 0.04, f"{toa(phi, doy):.0f} W/m2")
print(f"\n{len(fails)} failure(s)"); sys.exit(1 if fails else 0)
