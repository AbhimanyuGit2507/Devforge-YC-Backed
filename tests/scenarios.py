"""Scenario tests on the served model (direct calls, no HTTP): physical ranges, shock signs and locality, zero shock,
uncertainty, inverse search. Prints PASS/FAIL per check and exits non-zero on any failure.

    python tests/scenarios.py --data data/india_climate_train.nc
Also writes /tmp/monsoon_rollouts.json for tests/maps.js (dashboard formula cross-check).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve
from core.inverse import discover

ap = argparse.ArgumentParser(); ap.add_argument("--samples", default="dashboard/samples.json"); a = ap.parse_args()
S = json.loads(Path(a.samples).read_text()); land = np.array(S["land"]) > 0.5
LAT = np.arange(6.5, 37.6, 1.0)[:, None]; LON = np.arange(65.5, 100.6, 1.0)[None, :]
inbox = lambda b: (LAT >= b[0]) & (LAT <= b[1]) & (LON >= b[2]) & (LON <= b[3])
BAY, ARAB = inbox((8, 20, 82, 94)), inbox((8, 20, 65, 75))
fails = []
def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}"); (None if ok else fails.append(name))

RANGES = {0: (233.15, 333.15), 1: (0, 500), 2: (-60, 60), 3: (-60, 60), 4: (95000, 106000), 5: (0, 90)}
dump = {}
for sm in S["samples"]:
    init = np.array(sm["init_states"], "f4"); sc = sm["doy_sincos"]
    out = serve.simulate(init, sc, 14, None)
    ok = np.isfinite(out).all() and all(RANGES[c][0] <= out[:, c].min() and out[:, c].max() <= RANGES[c][1] for c in range(6))
    check(f"14-day physical ranges: {sm['name']}", ok, f"t2m {out[:,0].min():.0f}-{out[:,0].max():.0f} K, rain max {out[:,1].max():.0f} mm, msl {out[:,4].min()/100:.0f}-{out[:,4].max()/100:.0f} hPa")
    dump[sm["name"]] = out[:10].round(3).tolist()

sm = S["samples"][1]; init = np.array(sm["init_states"], "f4"); sc = sm["doy_sincos"]; last = init[-1]
base = serve.simulate(init, sc, 10, None)
z = serve.simulate(init, sc, 10, np.zeros((6, 32, 36), "f4"))
check("zero shock identical (one-day)", np.abs(z - base).max() == 0)
b0, s0 = serve.sustained_pair(init, sc, 5, np.zeros((6, 32, 36), "f4")); check("zero shock identical (sustained)", np.abs(s0 - b0).max() == 0)

d = np.zeros((6, 32, 36), "f4"); d[0][BAY] = 3; d[5][BAY] = 0.15 * last[5][BAY]
w = serve.simulate(init, sc, 10, d) - base
check("warm Bay: box warmer on day 1", w[0, 0][BAY].mean() > 1.0, f"{w[0,0][BAY].mean():+.2f} K")
check("warm Bay: box moister on day 1", w[0, 5][BAY].mean() > 2.0, f"{w[0,5][BAY].mean():+.2f} kg/m2")
check("warm Bay: effect decays by day 10 (one-day kick)", abs(w[9, 0][BAY].mean()) < abs(w[0, 0][BAY].mean()), f"{w[9,0][BAY].mean():+.2f} K")
far = (LON < 75) & (LAT > 28)
check("warm Bay: no day-1 change far away (NW India)", np.abs(w[0, 0][far]).max() < 0.05, f"max {np.abs(w[0,0][far]).max():.3f} K")
bs, ss = serve.sustained_pair(init, sc, 10, d); ws = ss - bs
check("warm Bay sustained: box still >= +1.5 K on day 10", ws[9, 0][BAY].mean() >= 1.5, f"{ws[9,0][BAY].mean():+.2f} K")

st = np.zeros((6, 32, 36), "f4"); st[2][ARAB] = -0.4 * last[2][ARAB]; st[3][ARAB] = -0.4 * last[3][ARAB]; st[5][ARAB] = -0.2 * last[5][ARAB]; st[4][ARAB] = 200
bs, ss = serve.sustained_pair(init, sc, 10, st); coast = (LAT >= 8) & (LAT <= 20) & (LON < 77) & land
check("monsoon stall (coupled): west-coast rain down", (ss - bs)[:, 1][:, coast].mean() < 0, f"{(ss-bs)[:,1][:,coast].mean():+.2f} mm/day")

cool = np.zeros((6, 32, 36), "f4"); C = inbox((22, 30, 75, 85)) & land; cool[0][C] = -3
c = serve.simulate(init, sc, 3, cool) - base[:3]
check("land cooling −3 K: box cooler on day 1", c[0, 0][C].mean() < -1.0, f"{c[0,0][C].mean():+.2f} K")
wet = np.zeros((6, 32, 36), "f4"); wet[5][C] = 0.25 * last[5][C]
r = serve.simulate(init, sc, 3, wet) - base[:3]
check("land moisture +25 %: more rain in box on day 1", r[0, 1][C].mean() > 0, f"{r[0,1][C].mean():+.2f} mm/day")

ext = init.copy(); ext[-1, 0] += 15; ext[-1, 1] = 300
o = serve.simulate(ext, sc, 10, None)
check("extreme inputs (+15 K, 300 mm) stay valid", np.isfinite(o).all() and o[:, 1].min() >= 0 and 233.15 <= o[:, 0].min() and o[:, 0].max() <= 333.15)

serve.MODEL.z = serve.Z_ALL[:serve.members_for(10)]
m = serve.MODEL.members(init, sc, 10, None).numpy(); sd = m.std(0)
check("ensemble members finite, spread grows day 1 -> 8", np.isfinite(m).all() and sd[7].mean() > sd[0].mean(), f"{sd[0].mean():.3f} -> {sd[7].mean():.3f}")

dl, sens, hist = discover(serve.MODEL.runs[0], init, sc, (17, 21, 72, 76), 3, 40.0, iters=40)
check("inverse: reaches target within 20 %", abs(hist[-1] - 40) / 40 < 0.2, f"{hist[0]:.1f} -> {hist[-1]:.1f} mm")
check("inverse: trigger within caps (5 K, 5 m/s, 10 kg/m2)", np.abs(dl[0]).max() <= 5.01 and np.abs(dl[2:4]).max() <= 5.01 and np.abs(dl[5]).max() <= 10.01)

Path("/tmp/monsoon_rollouts.json").write_text(json.dumps(dump))
print(f"\n{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else "")); sys.exit(1 if fails else 0)
