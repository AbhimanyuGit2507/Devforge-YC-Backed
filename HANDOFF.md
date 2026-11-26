# Handoff (session state, 26 Nov ~22:15 IST) — deadline: submit before 10:00 IST

## Done
- Safety-net submission works: `serve.py` + `model.py` + `weights/` (baseline local CNN). 2020-21 self-eval with kit scoring:
  SS 0.103, SS vs persistence 0.215, CSI skill 0.084, anti-blur 10/10, causality 12.6/15, 31 ms/rollout, repo scan 15/15.
- New model code: `core/net.py` (MonsoonNet: advection core + local FiLM blocks, reach 5 cells/day, Ensemble with
  prob-matched rain), `core/losses.py` (afCRPS, spectral, physics), `core/data.py`, `train.py` (time-budgeted, cuda/mps/cpu).
  Split: train 2005-2018, val 2019, test 2020-21.

## Training in flight (each --minutes 150, started ~22:00-22:20)
- Kaggle (account abhimanyunegi07), kernel `abhimanyunegi07/monsoon-train`: seeds 1,2 on T4x2.
  Outputs `/kaggle/working/run_s1`, `run_s2`. Fetch: `kaggle kernels output abhimanyunegi07/monsoon-train -p runs/kaggle`
  (needs ~/.kaggle/kaggle.json). Status: `kaggle kernels status abhimanyunegi07/monsoon-train`.
- Colab: seed 3 -> Google Drive file `monsoon_run_s3.zip` (updated every 15 min, shared "anyone with link").
- Mac (MPS): seed 0 -> `runs/run_s0`, then pushes branch `mac-run-s0`.

## Next steps
1. Collect runs into `weights_v2/run_s*/` (each has net.pt, config.json, climatology_norm.npy, aux.npy).
2. Fit post-processing on 2019 only -> `weights_v2/postproc.json` with `tp_scale` per lead (max CSI >10 mm/day);
   check day-8 variance ratio (keep in 0.4-2.5).
3. Evaluate on 2020-21 once (adapt `evaluate.py` to load `core.net.Ensemble('weights_v2')`). Gate: must beat the
   safety net on SS, CSI and causality. If yes, point `serve.py` at the Ensemble; else keep the safety net.
4. Demo: inverse shock search (gradient loop through rollout) + sensitivity map; dashboard button + day play.
5. Brief (2-page PDF) from real eval numbers; light-cone plot of shock response vs distance/day.
6. Final: fresh clone, `python serve.py --port 8000`, `python -m monsoon_r1.schema_validator http://localhost:8000`,
   repo scan 15/15, commit + push to branch `claude/nifty-edison-pj04ca`.
Cut on purpose: Copilot, Gaussian shocks, FNO/UNet ablations.

## Gotchas
- repo_scan treats any .py containing "/simulate" as inference code: keep dates out of those files.
- Never commit `.nc` data or any .npy shaped (>400, ..., 32, 36).
- Apple MPS lacks grid_sample border padding (fixed via coordinate clamp); run with PYTORCH_ENABLE_MPS_FALLBACK=1.
