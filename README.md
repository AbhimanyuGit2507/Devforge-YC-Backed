# Monsoon Multiverse — a digital twin of India's daily weather

Track 2 "Multiverse of Monsoon". A compact, physics-aware neural simulator that rolls India's weather forward day by day
from the last 3 days (t2m, rain, u10, v10, msl, tcwv on the 32 × 36 1° grid), and branches alternate universes when a
local shock is applied. A browser dashboard drives it live and turns the rollouts into heat, flood, crop, energy and
ensemble-uncertainty impacts.

```
serve.py            inference server (FastAPI, offline, CPU) — POST /health, POST /simulate
model.py            the network + rollout (shared by training and serving)
train.py            training (CPU, ~45 min with 4 threads)
evaluate.py         self-evaluation on held-out 2020–21 with the organisers' scoring functions
weights/            trained weights + climatology/static maps the model conditions on (~1 MB, no LFS needed)
dashboard/          interactive map dashboard (static HTML/JS, separate from the server)
brief/              2-page brief (PDF) and the script that builds it
```

## Run (what the judges do)
```bash
pip install torch numpy fastapi uvicorn
python serve.py --port 8000
```
The server loads `weights/` at import (≈1 s), listens on 0.0.0.0 and answers a 10-step `/simulate` in well under 1 s on CPU.
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
python train.py --data data/india_climate_train.nc --out weights
python evaluate.py --data data/india_climate_train.nc        # needs the monsoon_r1 kit installed
python dashboard/make_samples.py --data data/india_climate_train.nc   # rebuild dashboard starting states
```

## Model in one paragraph
Each day is predicted as a residual on the previous day by a small CNN whose spatial receptive field is exactly 3 cells
(≈330 km): information can only travel 3 cells per simulated day, so shocks start local and spread outward and can never
teleport. The network sees the last 3 days, the target-day climatology, land-sea mask, orography, latitude/longitude and
the season (doy sin/cos). Rain is modelled in log1p space and is non-negative by construction. Training uses a curriculum
of 1-, 4- and 8-day autoregressive unrolls with backprop through the whole rollout (anti-drift) and a spatial-spread loss
term (anti-blur); a per-lead rain scale is calibrated on held-out years to maximise heavy-rain (>10 mm/day) CSI.
Inference is deterministic, so a zero shock returns exactly the unperturbed rollout. Details, results and failure modes
are in `brief/brief.pdf`.
