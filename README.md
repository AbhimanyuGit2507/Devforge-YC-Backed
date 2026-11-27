# Monsoon Multiverse — a digital twin of India's daily weather

Track 2 "Multiverse of Monsoon". A compact, physics-aware neural simulator that rolls India's weather forward day by day
from the last 3 days (t2m, rain, u10, v10, msl, tcwv on the 32 × 36 1° grid), and branches alternate universes when a
local shock is applied. A browser dashboard drives it live and turns the rollouts into heat, flood, crop, energy and
ensemble-uncertainty impacts.

```
serve.py            inference server (FastAPI, offline, CPU) — POST /health, /simulate, /inverse
core/               MonsoonNet: data + splits, network + ensemble, losses, inverse shock search
train.py            MonsoonNet training (CUDA / Apple MPS / CPU, time-budgeted, resumable)
fit_postproc.py     per-lead rain scale fitted on 2019 only
fit_blend.py        per-lead, per-variable blend of ensemble, baseline and climatology, fitted on 2019 only
evaluate.py         self-evaluation with the organisers' scoring functions (--v2 for the ensemble)
weights_v2/         submitted ensemble (3 seeds) + climatology/static maps (~30 MB, no LFS needed)
model.py, weights/, train_baseline.py   first baseline model (used automatically if weights_v2/ is absent)
dashboard/          interactive map dashboard (static HTML/JS, separate from the server)
brief/              2-page brief (PDF) and the script that builds it
```

## Run (what the judges do)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only build (~200 MB instead of GBs of CUDA)
pip install numpy fastapi uvicorn
python serve.py --port 8000
```
Tested with Python 3.13 and torch 2.15 (CPU); any torch >= 2.0 should work. Startup takes ~10-15 s (weights + speed test)
before `/health` answers.
The server loads `weights_v2/` at import (≈2 s), listens on 0.0.0.0 and answers a 10-step `/simulate` in ≈1.3 s on 4 CPU cores.
**Speed guard:** at startup the server times a few 10-day rollouts on this machine (fastest of 3) and then uses as many of
the 8 noise members (× 3 seeds) as fit a 2.5 s model budget per request, with threads = min(4, available cores). On 4 cores
all members run (≈1.3–2.8 s per call, depending on the CPU); slower machines use fewer members, which keeps every call under
the 5 s limit at a small cost in skill (measured: 10-day calls under 2 s on 1 core, 2.5 s on 2 cores). The member count depends only on
`steps`, so identical requests and the zero shock still give identical outputs.
Check it with the kit validator: `python -m monsoon_r1.schema_validator http://localhost:8000`.

## Dashboard
```bash
python serve.py --port 8000            # terminal 1: the model
python dashboard/serve_dashboard.py    # terminal 2: http://localhost:8080
```
Pick a starting state (real 3-day windows), choose a preset shock (warm Bay of Bengal, monsoon stall) or draw your own
box on any map, press **Run**. Every map is computed in the browser from the live `/simulate` output:
Weather (prime / shocked / difference by day), Heat stress, Flood risk, Kharif crop water stress, Wind & solar energy,
and Uncertainty (the trained 24-member ensemble with spread calibrated on 2019: ± std and heavy-rain probability). Shocks can be a one-day kick (as in the official test) or held every day (like a sea that stays warm). No internet or CDN is needed.

## Train
```bash
pip install -r requirements.txt
# data from the kit release: https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/tag/v1.0.0
mkdir -p data && curl -L -o data/india_climate_train.nc https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/download/v1.0.0/india_climate_train.nc
python train.py --seed 1 --minutes 150 --out weights_v2/run_s1      # one seed; GPU recommended (T4: ~2.5 h)
python train.py --seed 1 --init weights_v2/run_s1 --stages 10:1.0 --lr 3e-4 --minutes 45 --out weights_v2/run_s1   # optional 10-day fine-tune
python fit_postproc.py --weights weights_v2                          # rain scale, fitted on 2019 only
python fit_blend.py --weights weights_v2                             # blend weights, fitted on 2019 only
python evaluate.py --v2 weights_v2 --years test                      # needs the monsoon_r1 kit installed
python dashboard/make_samples.py --data data/india_climate_train.nc   # rebuild dashboard starting states
```

## Results (held-out 2020–21, organisers' scoring code, 103 windows)
| | forecast skill | heavy-rain CSI skill | anti-blur | shocks | 10-step latency |
|---|---|---|---|---|---|
| baseline local CNN | 0.103 | 0.084 | 10/10 | 12.6/15 | 31 ms |
| MonsoonNet ensemble | 0.115 | 0.092 | 10/10 | 13.9/15 | ~0.9 s |
| Ensemble (2 seeds) + blend | 0.134 | 0.092 | 10/10 | 13.6/15 | ~0.9 s |
| **Ensemble (3 seeds) + blend (submitted)** | **0.137** | **0.096** | **10/10** | **13.6/15** | ~1.3 s |

## Model in one paragraph
MonsoonNet steps one day at a time: a physics core advects every field along the 10 m wind (semi-Lagrangian, ≤ 2 cells/day),
and a local learned correction (ConvNeXt-style blocks, per-pixel norm, FiLM conditioning on season, noise and lead day) adds
the rest. Its measured reach is 5 cells per simulated day, so shocks start local and spread outward and can never teleport.
It is trained as a CRPS ensemble (2 noise members, almost-fair CRPS) with a spectral anti-blur loss, extra weight on heavy
rain, a soft physics penalty and a 1 → 4 → 10-day rollout curriculum (anti-drift). Split: train 2005–18, tune 2019, test
2020–21. Serving averages 3 seeds × 8 fixed noise members, then a per-lead blend with the baseline and climatology (fitted on 2019) (deterministic: a zero shock returns exactly the same rollout). Details, results and failure modes
are in `brief/brief.pdf`.

## Data and licences
Training data: `india_climate_train.nc` from the organisers' team kit (v1.0.0, checksums verified against its
`SHA256SUMS.txt`). Contains modified Copernicus Climate Change Service information (ERA5, Hersbach et al. 2020),
accessed through Google WeatherBench 2 (Rasp et al. 2024); ERA5 is licensed under CC BY 4.0. The model weights,
`dashboard/samples.json` (six 3-day starting windows) and the numbers in `eval/` are derived from it.
Evaluation uses the kit's `monsoon_r1` scoring code (MIT), installed separately and never imported by the server.

## Dashboard-only server extras (never used by the grader's `/simulate` calls)
- `POST /ensemble` — forecast + calibrated uncertainty from the trained ensemble (spread scaled per lead/variable on 2019;
  spread/error 0.89–1.05 on unseen 2020–21; `fit_spread.py`).
- `POST /simulate` with `"sustained": true` — keeps the shock on every day (like a persistently warm sea) and returns a
  matched `baseline`; a zero shock still returns identical runs.
- `POST /inverse` — inverse shock discovery (smallest smooth trigger for a target rain amount).

## Tests
```bash
python tests/scenarios.py     # 20 scenario checks: physical ranges, shock signs/locality, zero shock, uncertainty, inverse
python tests/check_maps.py    # 43 checks: every dashboard map formula (run from dashboard/app.js in Node) vs an independent
                              # NumPy implementation, plus NWS heat-index table and top-of-atmosphere sunlight reference values
```
