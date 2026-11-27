# Live demo script (about 4 minutes)

## Before the judges arrive (offline, 2 terminals)
```bash
python serve.py --port 8000              # terminal 1: wait for "warm 10-step rollout ... members for 10-day calls"
python dashboard/serve_dashboard.py      # terminal 2, then open http://localhost:8080 (Cmd+Shift+R)
```
Press **▶ Run simulation** once to warm up. Keep `brief/brief.pdf` open in another tab and a screen recording of this
script as a backup. The header badge must read `ready · monsoonnet-ensemble-v2` (green).

## 1. Timeline Prime (30 s)
- Card ① **Peak monsoon, active spell**, card ② **No shock**, ▶ Run. Tab **🌦 Weather**, variable **Precipitation**, ▶ Play.
- Say: "Three real days in, ten simulated days out, about two seconds on a laptop CPU. Rain bands stay sharp at day 8 —
  no blur. Arrows are the model's 10 m winds."

## 2. Multiverse branch (60 s)
- Card ② **Warmer Bay of Bengal**, keep **"Keep the shock on every day"** ticked, ▶ Run. Map ③ is the shock's effect.
- Read the KPIs: change inside the box, largest change and where, change over nearby land. ▶ Play days 1 → 10.
- Say: "Information can only move about 5 cells (~550 km) a day, so the effect starts in the box and spreads. Untick the
  box to see the official one-day kick: it relaxes in a few days because the sea surface itself is unchanged."

## 3. Judges' own shock (60 s)
- Card ② **Your own shock**: choose variable, amount, then **drag a box on any map**. ▶ Run.
- Zero shock: amount 0 → every KPI shows +0.00 (deterministic model).
- If they ask for a weaker monsoon: card **Monsoon stall** (wind −40 %, moisture −20 %, +2 hPa) → west-coast rain drops.
  Be ready to explain why a wind-only cut is not used (learned correlation: slack winds come with offshore storms in data).

## 4. Impact (45 s)
- **🔥 Heat stress**: feels-like temperature (NWS heat index), danger ≥ 41 °C. **🌊 Flood risk**: rain vs each place's
  own 1-in-100-day total. **🌾 Crops**: rain deficit vs normal + hot days. **⚡ Wind & solar**: hub-height wind, sunlight.
- **🎲 Uncertainty**: "± spread calibrated so it matches real past errors (0.89–1.05 on unseen years)".

## 5. Reverse what-if (45 s)
- Left panel **Reverse what-if: find the trigger** → choose **Your own shock** and box lat 17–21, lon 72–76 (Mumbai),
  target 40 mm, day 3 → **Find the trigger** (~5 s). Tab **🎯 Trigger** opens.
- Say: "We freeze the model and run gradient descent on today's weather instead of the weights: the smallest realistic
  change (capped at 5 K, 5 m/s) that makes it rain, plus a sensitivity map — the adjoint idea behind 4D-Var."

## If something breaks
- Red badge "server not reachable": restart terminal 1, wait for the warm-up line, press Run.
- Anything else: switch to the recording and the brief and keep talking through the same points.
