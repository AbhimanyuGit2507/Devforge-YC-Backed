# Live demo script (about 4 minutes)

## Before the judges arrive (offline, 2 terminals)
```bash
python serve.py --port 8000              # terminal 1: wait for nothing to print; check with the next line
curl -X POST localhost:8000/health       # -> {"status":"ready","model_name":"monsoonnet-ensemble-v2"}
python dashboard/serve_dashboard.py      # terminal 2, then open http://localhost:8080
```
Press **Run simulation** once to warm up. Keep `brief/brief.pdf` open in another tab. Have a screen recording of
this script as a backup.

## 1. Timeline Prime (30 s)
- Starting state: **Peak monsoon, active spell**. Field: **Precipitation**. Preset: none. Run, then **▶ Play**.
- Say: "Three real days in, ten simulated days out, about a second on a laptop CPU. No blur: rain bands stay sharp at day 8."

## 2. Multiverse branch (60 s)
- Preset **Warm Bay of Bengal (+3 K, +15 % moisture)** -> Run. Show the third map (shocked − prime), play days 1 -> 8.
- Say: "Day 1 the change stays in the box; then the monsoon winds carry it inland. It can't jump: the model can only
  move information about 5 cells (~550 km) per day. That's measured, and it's the light-cone plot in our brief."

## 3. Judges' own shock (60 s)
- Ask a judge for a region and a change. Drag a box on the map (fills the custom box), pick the channel and amount, Run.
- If they ask for "zero shock": set amount 0 -> difference map is exactly zero (deterministic ensemble).

## 4. Impact (45 s)
- **Flood**: cells above 50/100 mm in 3 days, prime vs shocked. **Crop**: Kharif water stress for the monsoon-stall preset.
- **Ensemble**: spread map and probability of > 10 mm/day — "where we are confident and where we are not".

## 5. Inverse shock discovery (45 s)
- Custom box over Mumbai (lat 17–21, lon 72–76), day 3, target 40 mm -> **Find the trigger** (~5 s).
- Say: "We freeze the model and run gradient descent on today's weather instead of the weights. It finds the smallest
  realistic change — capped at 5 K, 5 m/s — that produces that rain, plus a sensitivity map: the adjoint idea
  forecast centres use in 4D-Var."

## If something breaks
- Dashboard shows "server not reachable": restart terminal 1, wait ~3 s, press Run.
- Anything else: switch to the backup recording and the brief; keep talking through the same points.
