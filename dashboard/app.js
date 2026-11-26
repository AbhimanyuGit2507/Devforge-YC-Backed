// Monsoon Multiverse dashboard. Everything shown is computed in the browser from live /simulate output.
const NC = 6, H = 32, W = 36, N = H * W;
const CH = [
  { name: "2 m temperature", unit: "°C", conv: v => v - 273.15, cmap: "heat" },
  { name: "Precipitation", unit: "mm/day", conv: v => v, cmap: "rain" },
  { name: "Wind u10 (east)", unit: "m/s", conv: v => v, cmap: "div" },
  { name: "Wind v10 (north)", unit: "m/s", conv: v => v, cmap: "div" },
  { name: "Sea-level pressure", unit: "hPa", conv: v => v / 100, cmap: "seq" },
  { name: "Column water vapour", unit: "kg/m²", conv: v => v, cmap: "rain" },
];
const BOXES = { warm_ocean: [8, 20, 82, 94], monsoon_stall: [8, 20, 65, 75] };
const TABS = ["Weather", "Heat", "Flood", "Crop", "Energy", "Ensemble", "Inverse"];
const TABLABEL = { Weather: "🌦 Weather", Heat: "🔥 Heat stress", Flood: "🌊 Flood risk", Crop: "🌾 Crops", Energy: "⚡ Wind & solar", Ensemble: "🎲 Uncertainty", Inverse: "🎯 Trigger" };
const TABDESC = {
  Weather: "Day-by-day forecast. Left: real-world forecast (Timeline Prime). Middle: the what-if universe. Right: the difference — red = more, blue = less. Arrows show the 10 m wind.",
  Heat: "Heat-stress days so far: humidity-adjusted daily temperature ≥ 33 °C on land. Use the day slider or ▶ Play to watch it build up.",
  Flood: "Flood risk: the largest 3-day rain total so far in each 110 km cell. More than 100 mm in 3 days = high risk.",
  Crop: "Kharif crop water stress (0 = fine, 1 = severe): rain deficit vs the normal for this month plus heat. Try the weaker-monsoon shock.",
  Energy: "Wind power (top row) and solar resource (bottom row) averaged over the days so far — what the shock means for renewables.",
  Ensemble: "8 slightly nudged copies of the same forecast. Where they disagree (spread), the forecast is uncertain. Right: chance of heavy rain.",
  Inverse: "Reverse what-if: the smallest change to today's weather that would produce your target rain, and where today's weather matters most.",
};
const SAMPLEDESC = {
  "Monsoon onset (Kerala)": "Early June: the monsoon is arriving over Kerala.",
  "Peak monsoon, active spell": "Late July: heavy rain along the west coast and central India.",
  "Pre-monsoon heatwave": "Early June, before the rains: hot and dry across the north.",
  "Cyclone Fani approach": "Late April / May: a strong cyclone in the Bay of Bengal heading for Odisha.",
  "Winter western disturbance": "January: a winter storm bringing rain and snow to the north-west.",
  "Retreating monsoon": "October: the monsoon withdrawing, rain shifting south-east.",
};
const $ = id => document.getElementById(id);
let S = null, base = null, shock = null, ens = null, inv = null, tab = "Weather";

const CMAPS = {
  seq: [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
  heat: [[49, 54, 149], [116, 173, 209], [255, 255, 191], [244, 109, 67], [165, 0, 38]],
  rain: [[15, 20, 24], [37, 87, 140], [44, 162, 205], [120, 214, 160], [250, 240, 90]],
  div: [[33, 102, 172], [146, 197, 222], [247, 247, 247], [244, 165, 130], [178, 24, 43]],
  risk: [[26, 40, 34], [80, 150, 70], [240, 200, 60], [235, 110, 40], [190, 20, 30]],
};
function color(cm, t) {
  const s = CMAPS[cm]; t = Math.min(1, Math.max(0, isFinite(t) ? t : 0)) * (s.length - 1);
  const i = Math.min(s.length - 2, Math.floor(t)), f = t - i;
  return s[i].map((c, k) => Math.round(c + (s[i + 1][k] - c) * f));
}
const grad = cm => `linear-gradient(90deg,${CMAPS[cm].map(c => `rgb(${c})`).join(",")})`;

// ---------- data helpers ----------
const flat = a => Float32Array.from(a.flat(Infinity));
const at = (R, k, c) => R.subarray(((k * NC) + c) * N, ((k * NC) + c + 1) * N);
const latOf = i => S.lat[i], lonOf = j => S.lon[j];
const land = () => S.landFlat;
function inBox(b, i, j) { return latOf(i) >= b[0] && latOf(i) <= b[1] && lonOf(j) >= b[2] && lonOf(j) <= b[3]; }

function perturbation(last) {
  const p = $("preset").value, d = new Float32Array(NC * N);
  if (p === "none") return null;
  let box, set;
  if (p === "warm_ocean") { box = BOXES.warm_ocean; set = (q) => { d[q] = 3; d[5 * N + q] = 0.15 * last[5 * N + q]; }; }
  else if (p === "monsoon_stall") { box = BOXES.monsoon_stall; set = q => { d[2 * N + q] = -0.4 * last[2 * N + q]; d[3 * N + q] = -0.4 * last[3 * N + q]; }; }
  else {
    box = ["la0", "la1", "lo0", "lo1"].map(id => +$(id).value);
    const c = +$("cch").value, v = +$("cval").value, pct = $("cmode").value === "pct";
    set = q => { d[c * N + q] = pct ? v / 100 * last[c * N + q] : v; };
  }
  for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) if (inBox(box, i, j)) set(i * W + j);
  return d;
}
const toNested = (d, shape) => { // flat -> nested lists
  const rec = (off, dims) => dims.length === 1 ? Array.from(d.subarray(off, off + dims[0]))
    : Array.from({ length: dims[0] }, (_, k) => rec(off + k * dims.slice(1).reduce((a, b) => a * b, 1), dims.slice(1)));
  return rec(0, shape);
};

async function simulate(sample, steps, pert) {
  const body = { init_states: sample.init_states, doy_sincos: sample.doy_sincos, steps, perturbation: pert ? toNested(pert, [NC, H, W]) : null };
  const r = await fetch($("server").value.replace(/\/$/, "") + "/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error("server " + r.status);
  const j = await r.json();
  return { R: flat(j.rollout), ms: j.inference_ms };
}

// ---------- impact models (simple, transparent proxies) ----------
function heat(R, steps) { // heat-stress index: daily-mean T (°C) raised by humidity; >= 33 counts as a heat-stress day
  const days = new Float32Array(N), peak = new Float32Array(N).fill(-99);
  for (let k = 0; k < steps; k++) { const T = at(R, k, 0), Q = at(R, k, 5);
    for (let q = 0; q < N; q++) { const h = T[q] - 273.15 + 0.25 * Math.max(Q[q] - 30, 0); peak[q] = Math.max(peak[q], h); if (h >= 33 && land()[q] > 0.5) days[q]++; } }
  return { map: days, peak };
}
function flood(R, steps) { // max 3-day accumulated rain (mm)
  const m = new Float32Array(N);
  for (let k = 0; k < steps; k++) for (let q = 0; q < N; q++) {
    let s = 0; for (let o = Math.max(0, k - 2); o <= k; o++) s += at(R, o, 1)[q]; m[q] = Math.max(m[q], s); }
  return m;
}
function crop(R, steps) { // 0..1 stress on land: rain deficit vs monthly climatology (70%) + heat days (30%)
  const clim = S.tpClim[S.cur.month - 1], hd = heat(R, steps).map, out = new Float32Array(N);
  for (let q = 0; q < N; q++) { if (land()[q] < 0.5) { out[q] = NaN; continue; }
    let r = 0; for (let k = 0; k < steps; k++) r += at(R, k, 1)[q];
    const c = clim[q] * steps, def = c > steps ? Math.max(0, 1 - r / c) : 0;
    out[q] = 0.7 * def + 0.3 * Math.min(1, hd[q] / steps * 2); }
  return out;
}
function energy(R, steps) { // wind power density at 100 m (W/m²) and a solar clearness-weighted insolation index (W/m²)
  const wind = new Float32Array(N), sun = new Float32Array(N), doy0 = Math.round(Math.atan2(...S.cur.doy_sincos) / 2 / Math.PI * 365 + 365) % 365;
  for (let k = 0; k < steps; k++) { const u = at(R, k, 2), v = at(R, k, 3), tp = at(R, k, 1), tw = at(R, k, 5);
    const dec = 23.44 * Math.sin(2 * Math.PI * (284 + doy0 + k + 1) / 365) * Math.PI / 180;
    for (let q = 0; q < N; q++) { const sp = Math.hypot(u[q], v[q]) * Math.pow(10, 1 / 7);
      wind[q] += 0.5 * 1.225 * sp ** 3 / steps;
      const phi = latOf(Math.floor(q / W)) * Math.PI / 180, ws = Math.acos(Math.max(-1, Math.min(1, -Math.tan(phi) * Math.tan(dec))));
      const toa = 1361 / Math.PI * (ws * Math.sin(phi) * Math.sin(dec) + Math.cos(phi) * Math.cos(dec) * Math.sin(ws));
      const clear = Math.max(0.15, Math.min(0.8, 0.8 - 0.004 * tw[q] - 0.35 * (1 - Math.exp(-tp[q] / 5))));
      sun[q] += toa * clear / steps; } }
  return { wind, sun };
}
const mean = (a, mask) => { let s = 0, n = 0; a.forEach((v, q) => { if (isFinite(v) && (!mask || mask[q] > 0.5)) { s += v; n++; } }); return n ? s / n : NaN; };
const count = (a, f) => a.reduce((n, v) => n + (f(v) ? 1 : 0), 0);
const diff = (a, b) => a.map((v, q) => b[q] - v);

// ---------- drawing ----------
let drag = null;
const CELL = 12;                                  // canvas pixels per 1-degree cell
const CITIES = [["Delhi", 28.6, 77.2, 1], ["Mumbai", 19.1, 72.9, 1], ["Kolkata", 22.6, 88.4, 1], ["Chennai", 13.1, 80.3, 1], ["Bengaluru", 13.0, 77.6, -1], ["Ahmedabad", 23.0, 72.6, -1], ["Guwahati", 26.1, 91.7, 1]];
const px = lon => (lon - 65) * CELL, py = lat => (38 - lat) * CELL;
function coastline(g) {  // marching squares on the land fraction at cell centres -> smooth 0.5 contour
  const L = land(), v = (i, j) => L[i * W + j], cx = j => (j + 0.5) * CELL, cy = i => (H - i - 0.5) * CELL;
  const lerp = (a, b) => (0.5 - a) / (b - a);
  g.beginPath();
  for (let i = 0; i < H - 1; i++) for (let j = 0; j < W - 1; j++) {
    const a = v(i, j), b = v(i, j + 1), c = v(i + 1, j + 1), d = v(i + 1, j);       // bl, br, tr, tl (north = i+1)
    const pts = [];
    if ((a > 0.5) !== (b > 0.5)) pts.push([cx(j) + lerp(a, b) * CELL, cy(i)]);
    if ((b > 0.5) !== (c > 0.5)) pts.push([cx(j + 1), cy(i) - lerp(b, c) * CELL]);
    if ((c > 0.5) !== (d > 0.5)) pts.push([cx(j + 1) - lerp(c, d) * CELL, cy(i + 1)]);
    if ((d > 0.5) !== (a > 0.5)) pts.push([cx(j), cy(i + 1) + lerp(d, a) * CELL]);
    for (let k = 0; k + 1 < pts.length; k += 2) { g.moveTo(...pts[k]); g.lineTo(...pts[k + 1]); }
  }
  g.stroke();
}
function drawMap(title, data, cm, lo, hi, unit, box, wind) {
  const wrap = document.createElement("div"); wrap.className = "map";
  const cv = document.createElement("canvas"); cv.width = W * CELL; cv.height = H * CELL;
  const g = cv.getContext("2d"), img = g.createImageData(W, H);
  for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
    const v = data[i * W + j], o = ((H - 1 - i) * W + j) * 4, c = isFinite(v) ? color(cm, (v - lo) / (hi - lo)) : [40, 46, 52];
    img.data.set([...c, 255], o); }
  const tmp = document.createElement("canvas"); tmp.width = W; tmp.height = H; tmp.getContext("2d").putImageData(img, 0, 0);
  const cells = $("cells").checked;            // honest 1-degree cells, or a smooth (bilinear) view of the same values
  g.imageSmoothingEnabled = !cells; g.imageSmoothingQuality = "high"; g.drawImage(tmp, 0, 0, W * CELL, H * CELL);
  g.strokeStyle = "rgba(255,255,255,.10)"; g.lineWidth = 1;      // 5-degree graticule
  for (let lon = 70; lon <= 100; lon += 5) { g.beginPath(); g.moveTo(px(lon), 0); g.lineTo(px(lon), H * CELL); g.stroke(); }
  for (let lat = 10; lat <= 35; lat += 5) { g.beginPath(); g.moveTo(0, py(lat)); g.lineTo(W * CELL, py(lat)); g.stroke(); }
  g.strokeStyle = "rgba(15,20,24,.85)"; g.lineWidth = 1.6; g.lineJoin = "round"; coastline(g);
  if (wind && $("arrows").checked) {   // 10 m wind arrows every 3 cells, length scaled to the strongest wind shown
    let ref = 0; for (let q = 0; q < N; q++) ref = Math.max(ref, Math.hypot(wind.u[q], wind.v[q]));
    ref = Math.max(ref, wind.min || 0.5);
    for (let pass = 0; pass < 2; pass++) {
      g.strokeStyle = pass ? "rgba(255,255,255,.92)" : "rgba(0,0,0,.55)"; g.fillStyle = g.strokeStyle; g.lineWidth = pass ? 1.3 : 3;
      for (let i = 1; i < H; i += 3) for (let jx = 1; jx < W; jx += 3) {
        const q = i * W + jx, u = wind.u[q], v = wind.v[q], sp = Math.hypot(u, v); if (sp < ref * 0.06) continue;
        const L = Math.min(1, sp / ref) * 2.6 * CELL, x0 = (jx + 0.5) * CELL, y0 = (H - i - 0.5) * CELL, dx = u / sp * L, dy = -v / sp * L;
        g.beginPath(); g.moveTo(x0 - dx / 2, y0 - dy / 2); g.lineTo(x0 + dx / 2, y0 + dy / 2); g.stroke();
        const a = Math.atan2(dy, dx), hx = x0 + dx / 2, hy = y0 + dy / 2, hs = 4 + L * 0.12;
        g.beginPath(); g.moveTo(hx, hy); g.lineTo(hx - hs * Math.cos(a - 0.45), hy - hs * Math.sin(a - 0.45)); g.lineTo(hx - hs * Math.cos(a + 0.45), hy - hs * Math.sin(a + 0.45)); g.closePath(); g.fill();
      }
    }
    g.fillStyle = "rgba(255,255,255,.75)"; g.font = "10px system-ui, sans-serif"; g.fillText(`longest arrow = ${ref.toFixed(1)} m/s`, W * CELL - 150, 12);
  }
  g.font = "600 10px system-ui, sans-serif"; g.textBaseline = "middle";
  CITIES.forEach(([n, la, lo2, side]) => { const x = px(lo2), y = py(la), tx = side > 0 ? x + 5 : x - 5 - g.measureText(n).width;
    g.fillStyle = "#fff"; g.strokeStyle = "rgba(0,0,0,.7)"; g.lineWidth = 2.5; g.beginPath(); g.arc(x, y, 2.6, 0, 7); g.stroke(); g.fill();
    g.strokeText(n, tx, y); g.fillText(n, tx, y); });
  g.fillStyle = "rgba(255,255,255,.55)"; g.font = "9px system-ui, sans-serif";
  for (let lon = 70; lon <= 95; lon += 10) g.fillText(lon + "°E", px(lon) + 2, H * CELL - 7);
  for (let lat = 10; lat <= 30; lat += 10) g.fillText(lat + "°N", 3, py(lat) - 6);
  if (box) { g.strokeStyle = "#fff"; g.setLineDash([5, 4]); g.lineWidth = 1.8; g.strokeRect(px(box[2]), py(box[1]), px(box[3]) - px(box[2]), py(box[0]) - py(box[1])); g.setLineDash([]); }
  const toLL = e => { const r = cv.getBoundingClientRect(); return [38 - (e.clientY - r.top) / r.height * H, 65 + (e.clientX - r.left) / r.width * W]; };
  cv.onmousedown = e => { drag = toLL(e); };
  cv.onmouseup = e => { if (!drag) return; const b = toLL(e);
    $("la0").value = Math.round(Math.min(drag[0], b[0])); $("la1").value = Math.round(Math.max(drag[0], b[0]));
    $("lo0").value = Math.round(Math.min(drag[1], b[1])); $("lo1").value = Math.round(Math.max(drag[1], b[1]));
    setPreset("custom"); drag = null; render(); };
  cv.onmousemove = e => { const [la, lo2] = toLL(e), i = Math.floor(la - 6), jj = Math.floor(lo2 - 65);
    if (i >= 0 && i < H && jj >= 0 && jj < W) cv.title = `${la.toFixed(1)}°N ${lo2.toFixed(1)}°E: ${(+data[i * W + jj]).toFixed(2)} ${unit}`; };
  const f = v => Math.abs(hi - lo) >= 20 ? v.toFixed(0) : v.toFixed(1);
  wrap.innerHTML = `<div class="cap">${title}</div>`; wrap.appendChild(cv);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map(t => `<span style="left:${t * 100}%">${f(lo + t * (hi - lo))}</span>`).join("");
  wrap.insertAdjacentHTML("beforeend", `<div class="cbar" style="background:${grad(cm)}"></div><div class="ticks">${ticks}</div><div class="unit">${unit}</div>`);
  $("maps").appendChild(wrap);
}
const range = (...arrs) => { let lo = Infinity, hi = -Infinity; arrs.forEach(a => a && a.forEach(v => { if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } })); return [lo, hi === lo ? lo + 1 : hi]; };
const sym = (...arrs) => { const [lo, hi] = range(...arrs); const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-3); return [-m, m]; };
function kpis(list) { $("kpis").innerHTML = list.map(([v, l]) => `<div class="kpi"><b>${v}</b><span>${l}</span></div>`).join(""); }
function curBox() { const p = $("preset").value; return p === "custom" ? ["la0", "la1", "lo0", "lo1"].map(id => +$(id).value) : BOXES[p]; }

// pairs of maps: baseline, shocked, shocked-minus-baseline
function triple(fn, cm, unit, title, dcm = "div", wfn = null) {
  const a = fn(base.R), b = shock && fn(shock.R), [lo, hi] = range(a, b);
  const wa = wfn && wfn(base.R), wb = wfn && shock && wfn(shock.R);
  drawMap(shock ? `① Real-world forecast — ${title}` : title, a, cm, lo, hi, unit, null, wa);
  if (b) { drawMap(`② What-if universe — ${title}`, b, cm, lo, hi, unit, curBox(), wb); const d = diff(a, b), [dl, dh] = sym(d);
    drawMap(`③ Difference (② − ①): red = more, blue = less`, d, dcm, dl, dh, unit, curBox(), wb && { u: diff(wa.u, wb.u), v: diff(wa.v, wb.v), min: 0.05 }); }
  return [a, b];
}

function render() {
  $("maps").innerHTML = ""; $("kpis").innerHTML = ""; $("tabdesc").textContent = TABDESC[tab];
  if (!base && tab !== "Inverse") { $("maps").innerHTML = `<div class="empty">Choose a starting day (①) and optionally a what-if shock (②), then press <b>▶ Run simulation</b> (③).</div>`; $("explain").textContent = ""; return; }
  if (!base) { $("maps").innerHTML = inv ? "" : `<div class="empty">Open "Reverse what-if: find the trigger" on the left, set a box and a target, and press <b>Find the trigger</b>.</div>`; }
  const steps = base ? base.steps : 10, k = Math.min(+$("day").value, steps) - 1, c = +$("chan").value, ch = CH[c];
  const L = land(), kk = k + 1;   // impact tabs accumulate "through day kk", so the slider and Play animate them too
  $("chan").disabled = !(tab === "Weather" || tab === "Ensemble"); $("arrows").disabled = tab !== "Weather";
  $("dayl").textContent = tab === "Weather" || tab === "Ensemble" ? k + 1 : tab === "Inverse" ? "(not used here)" : `1–${k + 1} (totals so far)`;
  if (tab === "Weather") {
    triple(R => at(R, k, c).map(ch.conv), ch.cmap, ch.unit, `${ch.name}, day ${k + 1}`, "div", R => ({ u: at(R, k, 2), v: at(R, k, 3) }));
    const a = at(base.R, k, c).map(ch.conv);
    kpis([[mean(a).toFixed(1) + " " + ch.unit, "India-box mean, Prime"], [mean(a, L).toFixed(1) + " " + ch.unit, "land mean, Prime"],
      ...(shock ? [[(mean(at(shock.R, k, c).map(ch.conv)) - mean(a)).toFixed(2) + " " + ch.unit, "shock change (domain mean)"]] : [])]);
    $("explain").textContent = "Day-by-day rollout of the model. The shocked universe starts from the same 3 days plus your shock; the difference map shows the shock staying local, then spreading.";
  } else if (tab === "Heat") {
    const [a, b] = triple(R => heat(R, kk).map, "heat", "days", `Heat-stress days (index ≥ 33 °C), days 1–${kk}`);
    const n = x => count(x, v => v >= 3);
    kpis([[n(a), "land cells with ≥3 heat-stress days (Prime)"], ...(b ? [[n(b), "… in the shocked universe"]] : []), [Math.max(...heat(base.R, kk).peak).toFixed(1) + " °C", "peak heat-stress index"]]);
    $("explain").textContent = "Heat-stress index = daily-mean 2 m temperature + 0.25 × (column water vapour − 30), a humidity-weighted proxy; a day ≥ 33 °C daily mean corresponds roughly to IMD heatwave maxima.";
  } else if (tab === "Flood") {
    const [a, b] = triple(R => flood(R, kk), "rain", "mm", `Max 3-day rainfall, days 1–${kk}`);
    kpis([[count(a, v => v > 100), "cells > 100 mm in 3 days (high risk)"], [count(a, v => v > 50), "cells > 50 mm (moderate)"], ...(b ? [[count(b, v => v > 100) - count(a, v => v > 100), "change in high-risk cells from shock"]] : [])]);
    $("explain").textContent = "Flood risk proxy: largest 3-day accumulated rain per 1° cell. Grid-cell means are much smoother than station rain, so thresholds are lower than IMD station categories.";
  } else if (tab === "Crop") {
    const [a, b] = triple(R => crop(R, kk), "risk", "stress 0–1", `Kharif crop water-stress index, days 1–${kk}`);
    kpis([[(mean(a) * 100).toFixed(0) + "%", "mean land stress (Prime)"], [(count(a, v => v > 0.5) / count(L, v => v > 0.5) * 100).toFixed(0) + "%", "land area under high stress"],
      ...(b ? [[((mean(b) - mean(a)) * 100).toFixed(1) + " pts", "stress change from shock"]] : [])]);
    $("explain").textContent = "Stress = 0.7 × rain deficit versus the month's 2005–2021 climatology + 0.3 × fraction of heat-stress days. Try the monsoon-stall shock.";
  } else if (tab === "Energy") {
    const [wa, wb] = triple(R => energy(R, kk).wind, "seq", "W/m²", `Wind power density at 100 m, mean of days 1–${kk}`);
    const [sa, sb] = triple(R => energy(R, kk).sun, "heat", "W/m²", `Solar resource (clearness-weighted), days 1–${kk}`);
    kpis([[mean(wa).toFixed(0) + " W/m²", "mean wind power density"], [mean(sa, L).toFixed(0) + " W/m²", "mean solar on land"],
      ...(wb ? [[((mean(wb) / mean(wa) - 1) * 100).toFixed(1) + "%", "wind change from shock"], [((mean(sb, L) / mean(sa, L) - 1) * 100).toFixed(1) + "%", "solar change from shock"]] : [])]);
    $("explain").textContent = "Wind: ½ρv³ with 10 m wind lifted to 100 m by a 1/7 power law. Solar: top-of-atmosphere daily insolation × a clearness factor that drops with moisture and rain.";
  } else if (tab === "Inverse") {
    if (!inv) { $("explain").textContent = ""; return; }
    const dT = flat(inv.delta[0]), dq = flat(inv.delta[5]), sens = flat(inv.sensitivity), b = inv.box;
    drawMap("Discovered trigger: temperature change today", dT, "div", ...sym(dT), "K", b);
    drawMap("Discovered trigger: moisture change today", dq, "div", ...sym(dq), "kg/m²", b);
    drawMap("Sensitivity map (where today matters most)", sens, "seq", 0, 1, "relative", b);
    const r = inv.rain_by_iter;
    kpis([[r[0].toFixed(1) + " mm", "box rain before"], [r[r.length - 1].toFixed(1) + " mm", "box rain after search"], [inv.target + " mm", "target"], [Math.max(...dT.map(Math.abs)).toFixed(2) + " K", "largest temperature change"]]);
    $("explain").textContent = `Gradient descent through the frozen model found the smallest smooth change to today's weather that pushes day-${inv.day} rain in the box toward ${inv.target} mm.`;
  } else if (tab === "Ensemble") {
    if (!ens) { $("explain").textContent = "Ensemble members are still running in the background (8 extra simulations); this tab fills in automatically."; return; }
    const M = ens.length, mu = new Float32Array(N), sd = new Float32Array(N), pr = new Float32Array(N);
    ens.forEach(R => { const x = at(R, k, c), r = at(R, k, 1); for (let q = 0; q < N; q++) { mu[q] += ch.conv(x[q]) / M; pr[q] += (r[q] > 10) / M; } });
    ens.forEach(R => { const x = at(R, k, c); for (let q = 0; q < N; q++) sd[q] += (ch.conv(x[q]) - mu[q]) ** 2 / M; });
    sd.forEach((v, q) => sd[q] = Math.sqrt(v));
    drawMap(`${ch.name} ensemble mean, day ${k + 1}`, mu, ch.cmap, ...range(mu), ch.unit);
    drawMap(`Ensemble spread (std), day ${k + 1}`, sd, "seq", ...range(sd), ch.unit);
    drawMap(`P(rain > 10 mm/day), day ${k + 1}`, pr, "rain", 0, 1, "probability");
    kpis([[M, "members"], [mean(sd).toFixed(2) + " " + ch.unit, "mean spread"], [count(pr, v => v >= 0.5), "cells with ≥50% heavy-rain chance"]]);
    $("explain").textContent = `Each member starts from the same days with small smooth random nudges (±0.5 K, ±0.5 m/s, ±1 kg/m² moisture) plus the current shock. Spread growing with lead time shows where the forecast is uncertain.`;
  }
}

function smoothNoise(amp) {
  let a = Float32Array.from({ length: N }, () => (Math.random() * 2 - 1));
  for (let it = 0; it < 3; it++) { const b = new Float32Array(N);
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) { let s = 0, n = 0;
      for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) { const y = i + di, x = j + dj; if (y >= 0 && y < H && x >= 0 && x < W) { s += a[y * W + x]; n++; } }
      b[i * W + j] = s / n; } a = b; }
  const m = Math.max(...a.map(Math.abs)); return a.map(v => v / m * amp);
}

let runId = 0;
async function run() {
  const id = ++runId;                                       // a newer Run makes older in-flight results stale
  const steps = Math.max(1, Math.min(14, +$("steps").value)); S.cur = S.samples[+$("sample").value];
  const last = flat(S.cur.init_states[2]), pert = perturbation(last);
  $("run").disabled = true; status("simulating…"); ens = null;
  try {
    const t0 = performance.now();
    const [b, s] = await Promise.all([simulate(S.cur, steps, null), pert ? simulate(S.cur, steps, pert) : null]);
    if (id !== runId) return;
    base = { ...b, steps }; shock = s && { ...s, steps };
    $("day").max = steps; if (+$("day").value > steps) $("day").value = steps;
    $("timing").textContent = `model ${b.ms.toFixed(0)} ms per rollout · round trip ${(performance.now() - t0).toFixed(0)} ms`;
    status(`live · ${S.cur.name} · uncertainty loading…`); $("run").disabled = false; guide(4); render();
    const members = await Promise.all(Array.from({ length: 8 }, () => {   // uncertainty runs in the background
      const d = pert ? Float32Array.from(pert) : new Float32Array(NC * N);
      [[0, 0.5], [2, 0.5], [3, 0.5], [5, 1]].forEach(([c, a]) => smoothNoise(a).forEach((v, q) => d[c * N + q] += v));
      return simulate(S.cur, steps, d);
    }));
    if (id !== runId) return;
    ens = members.map(r => r.R); status(`live · ${S.cur.name}`);
    if (tab === "Ensemble") render();
  } catch (e) { if (id === runId) status("error: " + e.message + " — is serve.py running?"); }
  if (id === runId) $("run").disabled = false;
}

async function inverse() {
  S.cur = S.samples[+$("sample").value];
  const box = ["la0", "la1", "lo0", "lo1"].map(id => +$(id).value), day = Math.max(1, Math.min(10, +$("tday").value)), target = +$("tmm").value;
  $("inv").disabled = true; status("searching for the trigger…");
  try {
    const r = await fetch($("server").value.replace(/\/$/, "") + "/inverse", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ init_states: S.cur.init_states, doy_sincos: S.cur.doy_sincos, box, day, target_mm: target }) });
    const j = await r.json(); if (j.error) throw new Error(j.error);
    inv = { ...j, box, day, target }; tab = "Inverse";
    [...$("tabs").children].forEach(x => x.className = x.dataset.t === "Inverse" ? "on" : "");
    status(`live · ${S.cur.name}`); render();
  } catch (e) { status("inverse search failed: " + e.message); }
  $("inv").disabled = false;
}

let timer = null;
function play() {
  if (timer) { clearInterval(timer); timer = null; $("play").textContent = "▶ Play"; return; }
  $("play").textContent = "❚❚ Pause";
  timer = setInterval(() => { const d = $("day"); d.value = +d.value >= +d.max ? 1 : +d.value + 1; render(); }, 700);
}

function status(t) {
  const el = $("status"); el.textContent = t;
  el.className = "pill" + (/^(live|ready)/.test(t) ? " ok" : /error|failed|not reachable/.test(t) ? " err" : "");
}
function guide(n) { [1, 2, 3, 4].forEach(i => $("g" + i).className = i <= n ? "on" : ""); }
function setPreset(p) {
  $("preset").value = p; $("custom").hidden = p !== "custom";
  [...$("presets").children].forEach(b => b.classList.toggle("on", b.dataset.p === p)); guide(2);
}

async function init() {
  S = await (await fetch("samples.json")).json(); S.landFlat = flat(S.land); S.tpClim = S.tp_clim_month.map(flat);
  S.samples.forEach((s, i) => $("sample").insertAdjacentHTML("beforeend", `<option value="${i}">${s.name}</option>`));
  const sd = () => { $("sdesc").textContent = SAMPLEDESC[S.samples[+$("sample").value].name] || ""; };
  $("sample").onchange = () => { sd(); guide(1); }; sd();
  [...$("presets").children].forEach(b => b.onclick = () => setPreset(b.dataset.p));
  CH.forEach((c, i) => $("chan").insertAdjacentHTML("beforeend", `<option value="${i}">${c.name}</option>`));
  TABS.forEach(t => { const b = document.createElement("button"); b.textContent = TABLABEL[t]; b.dataset.t = t; b.className = t === tab ? "on" : "";
    b.onclick = () => { tab = t; [...$("tabs").children].forEach(x => x.className = x === b ? "on" : ""); render(); }; $("tabs").appendChild(b); });
  ["day", "chan", "cells", "arrows"].forEach(id => $(id).oninput = render);
  $("run").onclick = run; $("inv").onclick = inverse; $("play").onclick = play;
  render();   // show the guided empty state immediately, before the server answers
  try { const r = await fetch($("server").value + "/health", { method: "POST" }); const j = await r.json(); status(`${j.status} · ${j.model_name}`); }
  catch { status("server not reachable — start it with: python serve.py --port 8000"); }
  render();
}
init();
