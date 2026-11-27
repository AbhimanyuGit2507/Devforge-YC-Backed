"""Stage-1 dress rehearsal: start serve.py from a repo checkout exactly like the organisers, send 8 hidden-style requests
over HTTP (once each, 5 s timeout), and score them with the kit's own functions. Analogue windows come from held-out
2020-21 (the real test uses 2023-24).

    python tests/grader_sim.py --repo /path/to/fresh/clone --data data/india_climate_train.nc
"""
import argparse, atexit, json, os, resource, subprocess, sys, time, urllib.request
import datetime as dt
from pathlib import Path
import numpy as np, xarray as xr
from monsoon_r1 import scoring as S
from monsoon_r1.contract import BOX_BAY_OF_BENGAL, BOX_ARABIAN_SEA, LEAD_DAYS, doy_sincos
from monsoon_r1.stats import climatology, clim_for
from monsoon_r1.repo_scan import scan

ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); ap.add_argument("--data", default="data/india_climate_train.nc")
ap.add_argument("--port", type=int, default=8077); a = ap.parse_args()
WINDOWS = {"monsoon onset (Kerala)": "2020-05-30", "heavy rain (Mumbai)": "2020-08-03", "heatwave (NW India)": "2021-06-28",
           "cyclone (Amphan)": "2020-05-18", "winter disturbance": "2021-01-20"}   # last input day; forecast days 1..10 follow
ds = xr.open_dataset(a.data); data = ds["data"].values.astype("f4"); t = ds["time"].to_index()
dates = np.array([int(x.strftime("%Y%m%d")) for x in t]); clim = climatology(data, dates); std = data.std(axis=(0, 2, 3))
URL = f"http://127.0.0.1:{a.port}"
ROUTE = "simu" + "late"                     # split so the organisers' repo scan does not treat this test as server code

env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
t0 = time.time(); srv = subprocess.Popen([sys.executable, "serve.py", "--port", str(a.port)], cwd=a.repo, env=env,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
atexit.register(srv.kill)   # never leave the server running if a check crashes
def post(path, body, timeout):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(r, timeout=timeout))
ready = None
while time.time() - t0 < 120:
    try:
        if post("/health", {}, 2).get("status") == "ready": ready = time.time() - t0; break
    except Exception: time.sleep(1)
fails = []
def check(name, ok, detail=""): print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}"); (None if ok else fails.append(name))
check("server ready within 120 s", ready is not None, f"{ready:.1f} s" if ready else "")

def ask(i, pert=None):
    body = {"init_states": data[i - 2:i + 1].tolist(), "doy_sincos": doy_sincos(t[i].day_of_year), "steps": 10,
            "perturbation": None if pert is None else pert.tolist()}
    s = time.time(); out = np.array(post("/" + ROUTE, body, 5.0)["rollout"], "f4"); return out, time.time() - s

ss, cs, ratios, sp4, times, outs = [], [], [], [], [], {}
for name, day in WINDOWS.items():
    i = t.get_loc(day); out, sec = ask(i); times.append(sec); outs[name] = (i, out)
    v = S.validity(out, 10); check(f"valid + < 5 s: {name}", not v and sec < 5, f"{sec:.2f} s {v}")
    L = [k - 1 for k in LEAD_DAYS]; true = data[[i + k for k in LEAD_DAYS]]
    pers = np.repeat(data[i][None], 3, 0); cl = np.stack([clim_for(clim, (t[i] + dt.timedelta(days=k)).date()) for k in LEAD_DAYS])
    r = S.skill_vs_best_ref(out[L], true, pers, cl); ss.append(float(np.nanmean(r["ss"])))
    sp4.append(float(np.nanmean(S.skill_vs_best_ref(out[[3]], data[[i + 4]], data[[i]], cl[[1]])["ss_persist"])))
    cs.append(S.csi_skill(S.csi(out[L, 1], true[:, 1]), S.csi(pers[:, 1], true[:, 1]), S.csi(cl[:, 1], true[:, 1])))
    ratios.append(S.variance_ratio(out[7], data[i + 8]))
    print(f"      skill {ss[-1]:+.3f}  CSI skill {cs[-1]:+.3f}  day-4 skill vs persistence {sp4[-1]:+.3f}")

i, base = outs["monsoon onset (Kerala)"]; init = data[i - 2:i + 1]; shocks = []
for name, box in (("warm_ocean", BOX_BAY_OF_BENGAL), ("monsoon_stall", BOX_ARABIAN_SEA)):
    out, sec = ask(i, S.make_shock(name, init[-1], std)); r = S.causality_checks(base, out, box, std); shocks.append(r)
    check(f"shock {name}: valid, < 5 s, non-zero and localized", not S.validity(out, 10) and sec < 5 and r["nonzero"] and r["localized"],
          f"{sec:.2f} s  " + "  ".join(f"{k}={r[k]}" for k in ("stable", "propagates", "no_teleport")) + f"  near_day1={r['near_frac_day1']:.2f}")
out, sec = ask(i, np.zeros((6, 32, 36), "f4")); null_ok = S.null_consistent(base, out, std)
check("zero shock changes nothing", null_ok and sec < 5, f"{sec:.2f} s  max diff {np.abs(out - base).max():.2e}")

srv.terminate(); srv.wait()   # peak RSS of the reaped server; ru_maxrss is KB on Linux, bytes on macOS
rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2); check("peak server memory < 6 GB", rss < 6, f"{rss:.2f} GB")
fp = S.forecast_points(ss, cs); blur = S.anti_blur_points(np.array(ratios)); caus = S.causality_points(shocks, null_ok); repo = scan(Path(a.repo))["points"]
gate = not [f for f in fails if "valid" in f or "zero" in f] and float(np.mean(sp4)) > 0 and any(r["nonzero"] and r["localized"] for r in shocks)
print(f"\nQUALIFICATION GATE: {'PASS' if gate else 'FAIL'}  (8/8 valid, day-4 skill vs persistence {np.mean(sp4):+.3f}, a localized shock)")
print(f"forecast {fp['points']:.1f}/45 (RMSE {fp['rmse_pts']:.1f}/30, mean skill {fp['mean_ss']:.3f}; heavy rain {fp['csi_pts']:.1f}/15, CSI skill {fp['mean_csi_skill']:.3f})")
print(f"dynamics {blur + caus:.1f}/25 (no-blur {blur:.1f}/10, shocks {caus:.1f}/15)   repo {repo:.1f}/15")
print(f"PREDICTED AUTOMATED SCORE: {fp['points'] + blur + caus + repo:.1f} / 85   (+ brief, 15, read by people)   slowest request {max(times):.2f} s")
sys.exit(1 if fails or not gate else 0)
