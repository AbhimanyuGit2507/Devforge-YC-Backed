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
evaluate.py         self-evaluation with the organisers' scoring functions (--v2 for the ensemble)
weights_v2/         submitted ensemble (2 seeds) + climatology/static maps (~30 MB, no LFS needed)
model.py, weights/, train_baseline.py   first baseline model (used automatically if weights_v2/ is absent)
dashboard/          interactive map dashboard (static HTML/JS, separate from the server)
brief/              2-page brief (PDF) and the script that builds it
```

## Run (what the judges do)
```bash
pip install torch numpy fastapi uvicorn
python serve.py --port 8000
```
The server loads `weights_v2/` at import (≈2 s), listens on 0.0.0.0 and answers a 10-step `/simulate` in ≈1 s on 4 CPU cores.
Check it with the kit validator: `python -m monsoon_r1.schema_validator http://localhost:8000`.

## Dashboard
```bash
python serve.py --port 8000            # terminal 1: the model
python dashboard/serve_dashboard.py    # terminal 2: http://localhost:8080
```
Pick a starting state (real 3-day windows), choose a preset shock (warm Bay of Bengal, monsoon stall) or draw your own
box on any map, press **Run**. Every map is computed in the browser from the live `/simulate` output:
Weather (prime / shocked / difference by day), Heat stress, Flood risk, Kharif crop water stress, Wind & solar energy,
and an 8-member Ensemble (spread and heavy-rain probability). No internet or CDN is needed.

## Train
```bash
pip install -r requirements.txt
# data from the kit release: https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/tag/v1.0.0
mkdir -p data && curl -L -o data/india_climate_train.nc https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/download/v1.0.0/india_climate_train.nc
python train.py --seed 1 --minutes 150 --out weights_v2/run_s1      # one seed; GPU recommended (T4: ~2.5 h)
python train.py --seed 1 --init weights_v2/run_s1 --stages 10:1.0 --lr 3e-4 --minutes 45 --out weights_v2/run_s1   # optional 10-day fine-tune
python fit_postproc.py --weights weights_v2                          # rain scale, fitted on 2019 only
python evaluate.py --v2 weights_v2 --years test                      # needs the monsoon_r1 kit installed
python dashboard/make_samples.py --data data/india_climate_train.nc   # rebuild dashboard starting states
```

## Results (held-out 2020–21, organisers' scoring code, 103 windows)
| | forecast skill | heavy-rain CSI skill | anti-blur | shocks | 10-step latency |
|---|---|---|---|---|---|
| baseline local CNN | 0.103 | 0.084 | 10/10 | 12.6/15 | 31 ms |
| **MonsoonNet ensemble (submitted)** | **0.115** | **0.092** | **10/10** | **13.9/15** | ~0.9 s |

## Model in one paragraph
MonsoonNet steps one day at a time: a physics core advects every field along the 10 m wind (semi-Lagrangian, ≤ 2 cells/day),
and a local learned correction (ConvNeXt-style blocks, per-pixel norm, FiLM conditioning on season, noise and lead day) adds
the rest. Its measured reach is 5 cells per simulated day, so shocks start local and spread outward and can never teleport.
It is trained as a CRPS ensemble (2 noise members, almost-fair CRPS) with a spectral anti-blur loss, extra weight on heavy
rain, a soft physics penalty and a 1 → 4 → 10-day rollout curriculum (anti-drift). Split: train 2005–18, tune 2019, test
2020–21. Serving averages 2 seeds × 8 fixed noise members (deterministic: a zero shock returns exactly the same rollout). Details, results and failure modes
are in `brief/brief.pdf`.
