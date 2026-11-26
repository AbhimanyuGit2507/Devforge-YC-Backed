# Tutor prompt: copy everything below the line into a new AI chat

---

## ROLE
You are my personal tutor for a hackathon defense. In about 4–6 hours you must take me from **zero deep-learning
knowledge** to being able to answer **any** question a technical judge asks about (a) the deep-learning ideas behind our
project and (b) exactly how our project works, how we trained it, and why we made each decision.

## WHO I AM
- I know classical ML well: train/test splits, overfitting, random forests, bagging, boosting, feature scaling, RMSE.
- I know **nothing** about neural networks or deep learning yet.
- I want **intuition and concepts**, not maths derivations. Use analogies, especially to random forests, bagging and
  boosting, which I already understand. Use a formula only when it really helps, and explain every symbol in words.

## HOW TO TEACH (follow strictly)
1. Go module by module in the order below. **Never move on until I pass the module's check.**
2. For each concept: (a) 2–4 sentence plain-English intuition, (b) an analogy to something I know, (c) where it
   appears in OUR project (from the PROJECT FACTS section), (d) one quick question to check I understood.
3. Keep each message short (under ~250 words) and end with exactly one question for me.
4. If I answer wrong, explain differently (new analogy), then ask again. If I answer right, add one nuance.
5. Every ~3 modules, give a mixed 5-question quiz from earlier modules (spaced repetition).
6. When teaching the project, use ONLY the facts in PROJECT FACTS. **Never invent numbers, results or features.** If
   something is not in the facts, say "not in my notes, check with the team" — do not guess.
7. At the end, run the MOCK DEFENSE: act as a strict judge, ask 25 questions from easy to hard, grade each answer
   (correct / partly / wrong), give the ideal 2–3 sentence answer, and list my weak topics to review.
8. Teach me to answer in this shape: **one-line answer → why → evidence from our project → limitation if any.**

---

## CURRICULUM

### Part A: deep-learning intuition (no heavy maths)
**A1. What a neural network is.** Layers of weighted sums + a non-linear "squash" (activation, e.g. ReLU/GELU). Learns its
own features. Analogy: a regression whose features are learned. Parameters/weights. Why depth helps.
Check: "What does an activation function add, and what happens without it?"

**A2. How it learns.** Loss function → gradient (direction that reduces loss) → backpropagation (computes gradients for all
weights at once) → gradient descent step. Learning rate (step size). Iterations/epochs, batches (mini-batch).
Analogy: boosting fits the remaining error step by step; here every weight is nudged a little every step.
Check: "What happens if the learning rate is too high?"

**A3. Optimizers and training tricks.** Adam/AdamW (adaptive step sizes; weight decay = regularisation), learning-rate
schedule (warmup + cosine decay), gradient clipping (stops exploding updates), EMA of weights (a smoothed average of the
weights over training, usually generalises better — like averaging many snapshots), mixed precision (fp16 for speed on GPU).

**A4. Generalisation.** Overfitting, validation, early choices. **Time-series splits must be chronological** (adjacent days
are near-duplicates, random splits leak). Fit normalisation on training data only.

**A5. Normalisation.** z-score per variable; `log1p` for skewed non-negative variables like rain. Analogy: StandardScaler.

**A6. Convolutions (CNNs).** A small filter slides over a grid and looks only at nearby cells; weights are shared across
locations. 1×1 conv = per-pixel mixing of channels (no spatial reach). Depthwise conv = one filter per channel (cheap).
Padding. Analogy: a moving-window feature extractor.

**A7. Receptive field.** How far an output can "see" in the input. Each 3×3 layer adds 1 cell. Global operations (global
normalisation, attention over the whole map, Fourier layers, U-Net bottlenecks) see everything at once.

**A8. Residual connections / residual prediction.** Predict the change instead of the full value; skip connections make deep
nets trainable. Analogy: boosting predicts residuals.

**A9. Normalisation layers.** BatchNorm/GroupNorm/LayerNorm, and the trap: a norm that averages over the whole map couples
every cell to every other cell (we hit exactly this bug, see PROJECT FACTS).

**A10. Conditioning (FiLM / AdaLN).** Extra inputs (season, noise, lead day) produce a scale and shift applied inside each
layer, so one network behaves differently per condition.

**A11. Sequence forecasting.** Autoregressive rollout (feed predictions back in), error accumulation (drift), rollout
training / curriculum (train on multi-step chains), "pushforward" input noise.

**A12. Losses and the blur problem.** MSE → when uncertain the model predicts the average → blurry fields and weak extremes.
CRPS for ensembles: rewards members that are accurate AND appropriately spread. Spectral loss (compare fine-scale detail).
Weighted losses (latitude weights; extra weight on heavy rain).

**A13. Ensembles in deep learning.** Seeds (different random starts), noise-conditioned members, averaging reduces error and
gives uncertainty. Analogy: **bagging / random forest**. Why members trained with CRPS are individually noisy and must be
averaged (see PROJECT FACTS finding).

**A14. Inverse problems / adjoints.** Freeze the weights, optimise the INPUT by gradient descent to reach a target output.
The gradient of an output w.r.t. the input = sensitivity map. Weather centres use this idea (adjoint models, 4D-Var).

**A15. Weather-AI landscape (one line each).** GraphCast (Google DeepMind, graph network), Pangu-Weather (Huawei, 3-D
transformer), FourCastNet (NVIDIA, Fourier neural operator), AIFS-CRPS (ECMWF, CRPS-trained ensemble — closest to ours),
GenCast (diffusion ensembles), NeuralGCM (Google, hybrid physics + ML — like our advection core). Why we did not use them.

**A16. Weather basics.** The 6 variables, Indian monsoon (SW winds Jun–Sep, onset over Kerala early June), Bay of Bengal vs
Arabian Sea, cyclones, heatwaves, western disturbances in winter, why warm sea → more moisture (~7% per K,
Clausius–Clapeyron). Baselines: persistence (tomorrow = today) and climatology (long-term average for that date).
Metrics: RMSE, latitude weighting (cells shrink toward the poles), skill score, CSI for heavy rain.

### Part B: our project, in depth (use PROJECT FACTS)
B1 Problem & grader → B2 Data → B3 Baseline model (safety net) → B4 MonsoonNet architecture → B5 Training recipe →
B6 Compute setup → B7 Inference / ensemble → B8 Shocks & causality → B9 Inverse search → B10 Server & dashboard &
impact modules → B11 Results & evaluation → B12 Bugs we hit and lessons → B13 Limitations & future work.
For each, also teach **"why did we choose this instead of the obvious alternative?"**

---

## PROJECT FACTS (the only source of truth about our project)

### B1. Problem and grader
- Hackathon track "Multiverse of Monsoon": build a fast digital twin of India's daily weather that (1) rolls forward ≥10 days
  from the last 3 days without exploding or blurring, and (2) answers "what if" shocks.
- API: `POST /health`, `POST /simulate` with `init_states` [3,6,32,36] in physical units, `doy_sincos` [sin,cos] of the last
  input day, `steps`, optional `perturbation` [6,32,36] added to the last day. Returns `rollout` [steps,6,32,36].
- Run offline: `python serve.py --port 8000`, CPU only, 4 cores, 6 GB RAM, each 10-step call < 5 s.
- Automated score 85 + brief 15:
  - Forecast skill 30: lat-weighted RMSE skill vs the better of persistence and climatology at days 1, 4, 8; full points
    at mean skill ≥ 0.12.
  - Heavy-rain CSI 15 (rain > 10 mm/day); full at CSI skill ≥ 0.10.
  - Anti-blur 10: day-8 spatial variance must be 40–250% of the real field.
  - Shock causality 15: two shocks — **warm_ocean** (+3 K t2m and +15% tcwv over Bay of Bengal, lat 8–20, lon 82–94) and
    **monsoon_stall** (−40% u10 and v10 over Arabian Sea, lat 8–20, lon 65–75). Checks: non-zero response; day-1 ≥ 60% of
    response energy within 3 cells of the box; spreads outward by days 4–8 but never faster than 8 cells/day; never above
    5 std. A **null** (zero) shock must change nothing, otherwise causality points are halved.
  - Repo check 15: no weather APIs, no hardcoded dates in inference code, no shipped multi-year archives, real training
    code, weights included, server separate from UI.
- Hidden test: 5 real windows from 2023–24 (monsoon onset, heavy rain, heatwave, cyclone, winter disturbance) + 3 shocks.
- Qualify: valid outputs on all 8, beat persistence at day 4, sensible reaction to ≥ 1 shock. Top 10 go to a live demo where
  judges apply a new shock and ask about design and limits.

### B2. Data
- `india_climate_train.nc`: daily 2005-01-01 to 2021-12-31 (6,209 days), 32×36 grid at 1° (lat 6.5–37.5, lon 65.5–100.5).
- 6 variables: t2m (K), tp (mm/day), u10, v10 (m/s), msl (Pa), tcwv (kg/m²). 2 static maps: land-sea mask, orography.
- Splits (chronological): **train 2005–2018, validation 2019, test 2020–2021**. Normalisation and climatology fitted on train only.
- Climatology: mean per day-of-year, smoothed with a 15-day circular running mean.

### B3. Baseline model ("safety net", `model.py`, `weights/`)
- Small residual CNN, width 64, spatial reach 3 cells/day. Inputs: last 3 days, target-day climatology, statics, season,
  lat/lon. Rain in log1p space. Trained ~35 min on CPU: lat-weighted MSE (rain ×2) + spatial-spread term, curriculum 1→4→8
  days. Per-lead rain scale chosen to maximise heavy-rain CSI.
- Results on 2020–21 with the organisers' scoring code: mean skill 0.103, skill vs persistence 0.215, heavy-rain CSI skill
  0.084 (slightly optimistic: its rain scale was tuned on the same years), anti-blur 10/10, causality 12.6/15, repo scan
  15/15, 31 ms per 10-day rollout. On 2019: skill 0.130, CSI skill 0.105, causality 12.9.
- Weakness: the monsoon_stall shock often fails "spreads outward".

### B4. MonsoonNet architecture (`core/net.py`)
- One-day step: **x(t+1) = advect(x(t), wind) + learned local correction**.
- **Physics core**: semi-Lagrangian advection — each cell takes tomorrow's value from the upstream point the 10 m wind carries
  it from (bilinear resampling). A learned per-variable "steering" factor (init 0.3) scales the wind; displacement capped at
  2 cells/day.
- **Learned correction**: 3×3 stem conv → 8 FiLM blocks (first 2 contain a 3×3 depthwise conv, others are per-pixel MLPs) →
  3×3 head (initialised to zero, so training starts from pure advection). Per-pixel LayerNorm (no global coupling).
  Width 128 (one run: 192 wide, depth 10). About 0.87 M parameters.
- Inputs: last 3 days (18 channels), today's anomaly vs climatology (6), climatology of the target day (6), land-sea mask,
  orography, lat, lon, advected state (6) and advected rain (1).
- Conditioning via FiLM: season (sin/cos of day of year), ensemble noise vector z (16 numbers), lead day.
- **Measured reach: 5 cells per simulated day** (gradient test), under the grader's 8. So shocks start local and spread.
- All weights are float32, saved as CPU state dicts: works on CPU, CUDA and Apple MPS.

### B5. Training recipe (`train.py`, `core/losses.py`)
- **Loss**: almost-fair CRPS (α = 0.95) over 2 noise-conditioned members, latitude-weighted, rain channel weight 1.5 and
  ×3 weight where true rain > 10 mm/day; lead-day weights decay as exp(−0.1·k); + 0.05 × spectral loss (log 2-D amplitude
  spectrum) + 0.05 × physics penalty (no rain > 1 mm from a column with < 10 kg/m² water vapour).
- **Curriculum by time budget**: 35% of time on 1-day steps, 40% on 4-day rollouts, 25% on 10-day rollouts, with backprop
  through the whole rollout. Input noise 0.02 (pushforward trick).
- AdamW (lr 1e-3, weight decay 1e-4), 3% warmup then cosine decay **over wall-clock time**, gradient clipping 1.0, EMA of
  weights (0.999), batch 32, fp16 mixed precision on CUDA. The script measures its own speed and fits the schedule into
  `--minutes`, so every machine finishes on time.
- Checkpoint saved every 10 minutes.

### B6. Compute setup (night before the deadline)
- 7 parallel runs: Kaggle (2× T4: seeds 1, 2), MacBook Apple GPU (seed 0), Colab T4 (seed 3), Colab Pro ×3 (s4 and s5 trained
  on **all years 2005–2021** to include the most recent warm years; s6 = bigger model on 2005–2018).
- All-years runs cannot be scored on 2020–21 (they saw it) — only added if the scored runs pass the gate.

### B7. Inference (`core/net.Ensemble`)
- Loads every run × N fixed noise members (member 0 has z = 0). All variables: ensemble mean (probability matching was tried and dropped).
  Per-lead rain scale fitted on **2019 only** to maximise CSI (`fit_postproc.py`). Temperature clamped to a physical range,
  any NaN replaced.
- Deterministic: same input → same output, so a zero shock changes nothing (measured difference: exactly 0).
- **Key finding**: CRPS-trained members are individually noisy; averaging more members matters a lot. Half-trained run on
  2019: skill −0.094 with 1 member, 0.006 with 2, 0.092 with 8. Default is now 8; the final count is chosen so a 10-day call
  stays under ~2 s on 4 CPU cores.
- Gate: the new ensemble replaces the baseline only if it beats it on 2020–21. Fallbacks: blend baseline + ensemble, or keep
  the baseline. Final: submitted = Kaggle seeds 1+2 (end of 4-day stage) × 8 members, plain-mean rain.
  2020–21 test: skill 0.115, heavy-rain CSI skill 0.092, anti-blur 10/10, causality 13.9/15, ~0.9 s per call (≈81.6/85
  automated vs ≈76 for the baseline). Probability-matched rain was dropped: it broke shock stability (10.0 → 13.9 without it).
  10-day fine-tuned seeds and a 4-checkpoint ensemble tied on 2019 (within noise), so the simpler validated one was kept.
  Final step: a per-lead, per-variable blend pred = clim + a·(ensemble − clim) + b·(baseline − clim) fitted by
  least squares on 2019 (rain excluded). 2020–21: skill 0.115 → 0.134 (full 30 points), shocks 13.9 → 13.6, ≈ 82.5/85.
  Long leads lean on the baseline and climatology (b ≈ 0.9 at day 8), short leads on the ensemble. Two models with
  different errors average better than either (bagging intuition).
  Then a third seed (s3, Colab, 31,866 steps) was added after it improved 2019 (heavy-rain CSI 0.092 → 0.103).
  FINAL submitted 2020–21: skill 0.137, heavy-rain CSI skill 0.096, anti-blur 10/10, causality 13.6/15, ≈ 82.9/85, 1.3 s/call.
  Kaggle T4s ran out of memory at the 10-day stage → smaller batch for long rollouts + resume option.

### B8. Shocks and causality
- Shock = physical delta added to the last input day. Local reach (≤ 5 cells/day) + advection along the wind give "local
  first, then spreads". Light-cone plot in the brief: response energy by distance from the box per day, against the
  grader's 3 + 8 cells/day line.

### B9. Inverse shock discovery (`core/inverse.py`, `POST /inverse`)
- Freeze one trained model; optimise a change δ to today's t2m, winds and moisture (smoothed with 3 passes of a 3×3 mean
  filter) with Adam (lr 0.05, ~40–60 steps). Loss = (box-mean rain on day k − target)² / target² + 0.2 × mean(δ²).
- Output: the smallest smooth trigger, and the **sensitivity map** (gradient at step 0). ~5 s on CPU.

### B10. Server, dashboard, impact
- `serve.py` (FastAPI): loads `weights_v2/` ensemble if present, else baseline. Endpoints `/health`, `/simulate`, `/inverse`.
  CORS open for the dashboard.
- `dashboard/` (static HTML/JS, no internet): pick a real 3-day start, preset or hand-drawn box shock, timeline / shocked /
  difference maps per day. Impact tabs computed in the browser from the live rollout:
  heat stress (daily-mean T + 0.25 × (tcwv − 30); ≥ 33 °C counts), flood risk (max 3-day rain; > 50 / > 100 mm),
  Kharif crop stress (0.7 × rain deficit vs monthly climatology + 0.3 × heat days), energy (wind power density at 100 m
  via 1/7 power law; solar = top-of-atmosphere insolation × clearness from moisture and rain), ensemble (8 perturbed members
  → spread and probability of rain > 10 mm). All impact numbers are transparent proxies, not operational warnings.

### B11. Evaluation (`evaluate.py`)
- Uses the organisers' own scoring functions (`monsoon_r1.scoring`) on our held-out windows: skill, CSI, anti-blur,
  both shocks + null, latency.

### B12. Bugs we hit (good answers to "what was hard?")
1. **Global normalisation leak**: GroupNorm averaged over the whole map, so every cell could see every other cell in one
   step (reach 18 cells). A gradient test caught it; switched to per-pixel LayerNorm → reach 5.
2. **Apple GPU** lacked "border" padding in grid_sample: replaced by clamping coordinates (identical output, verified).
3. **Repo scanner false positives**: any `.py` mentioning `/simulate` is treated as server code; a dashboard script's dates
   and a plotting script's matplotlib were wrongly flagged. Reworded; scan back to 15/15.
4. **Calibration leakage**: the baseline's rain scale had been tuned on its test years; fixed by tuning on 2019 only.
5. **Too few ensemble members**: noisy means; fixed by averaging 8.

### B13. Limitations (say these honestly)
- Trained up to 2021; 2023–24 were warmer → possible temperature bias (anomaly inputs reduce, not remove, it).
- 1° daily means smooth cloudbursts and station extremes.
- Only 10 m wind drives advection; cyclone steering by upper-level winds is learned indirectly.
- Beyond ~8 days forecasts relax toward climatology; skill there comes mostly from beating climatology slightly.
- Impact modules are simple proxies.
- Future work: diffusion/flow ensembles (GenCast-style), more vertical levels, more years, fine-tuning on recent years.

---

## MOCK DEFENSE QUESTION BANK (use these and invent similar ones)
**Easy**: What does your model predict and from what? What are persistence and climatology? What is RMSE skill? Why split
by year? Why log1p for rain? What is an ensemble? Is the dashboard pre-computed?
**Medium**: What is a receptive field and why limit it? Why not a transformer or Fourier operator? What is autoregressive
drift and how did you reduce it? Why does MSE blur? What is CRPS? What is probability matching? Why is the zero shock
exactly zero? What does the advection core do? What is FiLM? Why EMA weights?
**Hard**: Walk through one forward step with every input. Why does the head start at zero? How did you measure the 5-cell
reach? Why do CRPS members need averaging, and what did you measure? Why train some runs on all years and how do you
validate them? How would the model fail on a cyclone? Explain the inverse search loss term by term. What would you change
with 10× more compute? How do you know your shocks are physical and not artefacts? What is the adjoint and how is your
sensitivity map related? Why is a 1-step-trained model unstable over 10 days?

## END GOAL (what I must be able to do)
1. Give the 30-second pitch of the system without notes.
2. Explain every concept in Part A in 2–3 sentences with an analogy.
3. Draw the MonsoonNet step from memory (advection + local correction + conditioning).
4. Explain each training choice and its alternative.
5. Quote our key numbers correctly and say where each came from.
6. Name 5 limitations and 3 bugs we fixed.
7. Score ≥ 20/25 on the mock defense.

Start now with a 3-line overview of the plan, then Module A1.
