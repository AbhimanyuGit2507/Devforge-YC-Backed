// Runs the dashboard's own map formulas (dashboard/app.js) in Node on model rollouts, for tests/check_maps.py.
// usage: node tests/maps.js <samples.json> <rollouts.json>   -> prints JSON results to stdout
const fs = require("fs"), vm = require("vm");
const el = () => ({ value: "", checked: false, textContent: "", innerHTML: "", hidden: false, disabled: false, style: {}, children: [],
  classList: { toggle() {} }, appendChild() {}, insertAdjacentHTML() {}, addEventListener() {}, getContext: () => null });
const ctx = vm.createContext({ document: { getElementById: el, createElement: el }, location: { search: "", hostname: "localhost" },
  fetch: () => new Promise(() => {}), console, URLSearchParams, Math, Float32Array, Array, JSON, setInterval, clearInterval, performance: { now: () => 0 } });
vm.runInContext(fs.readFileSync(__dirname + "/../dashboard/app.js", "utf8"), ctx);
const samples = JSON.parse(fs.readFileSync(process.argv[2])), rolls = JSON.parse(fs.readFileSync(process.argv[3]));
const out = vm.runInContext(`(function (smp, rolls) {
  S = smp; S.landFrac = flat(S.land); S.landFlat = S.landFrac.map(v => v >= 0.5 ? 1 : 0); S.tpClim = S.tp_clim_month.map(flat); S.p95 = flat(S.flood_p95); S.p99 = flat(S.flood_p99);
  const res = {};
  for (const s of S.samples) { S.cur = s; const R = flat(rolls[s.name]), e = energy(R, 10), h = heat(R, 10);
    res[s.name] = { heat: Array.from(h.map), flood: Array.from(flood(R, 10)), crop: Array.from(crop(R, 10)), wind: Array.from(e.wind), speed: Array.from(e.speed), sun: Array.from(e.sun) }; }
  res.hi = [[32.22, 60], [37.78, 40], [26.67, 40], [35, 70]].map(([t, rh]) => heatIndexC(t, rh));
  return res; })`, ctx)(samples, rolls);
process.stdout.write(JSON.stringify(out));
