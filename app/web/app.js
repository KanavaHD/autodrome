/* PIT WALL — control logic: scan library, build the grid, run the start-light launch, poll telemetry. */
const $ = (s) => document.querySelector(s);
const api = (p, opt) => fetch(p, opt).then(r => r.json());

const state = {
  cars: [], tracks: [],
  car: null, track: null,
  agents: 20000, steps: 20000,
  agentPresets: [], lengthPresets: [],
  running: false, busy: false,
};

/* ───────────────────────── library ───────────────────────── */
async function loadLibrary() {
  const lib = await api("/api/library");
  state.cars = lib.cars; state.tracks = lib.tracks;
  state.agentPresets = lib.agent_presets; state.lengthPresets = lib.length_presets;
  $("#acState").textContent = lib.ac_ok ? "found" : "not found";
  $("#acState").style.color = lib.ac_ok ? "var(--go)" : "var(--magenta)";
  $("#carCount").textContent = lib.cars.length;
  $("#trkCount").textContent = lib.tracks.filter(t => t.has_ai).length + "/" + lib.tracks.length;
  renderCars(); renderTracks(); renderPresets();
}

function renderCars(filter = "") {
  const f = filter.toLowerCase();
  const list = $("#carList"); list.innerHTML = "";
  state.cars.filter(c => !f || (c.name + " " + c.brand + " " + c.id).toLowerCase().includes(f))
    .forEach(c => {
      const row = document.createElement("div");
      row.className = "row"; row.setAttribute("role", "option");
      row.setAttribute("aria-selected", state.car?.id === c.id);
      row.innerHTML = `
        <img class="row-badge" loading="lazy" src="/api/preview?kind=badge&car=${encodeURIComponent(c.id)}" alt="" onerror="this.style.visibility='hidden'">
        <div class="row-txt">
          <div class="row-name">${esc(c.name)}</div>
          <div class="row-sub">${esc(c.brand || c.id)}</div>
        </div>
        ${c.encrypted ? '<span class="flag enc" title="physics packed in data.acd — uses telemetry calibration">ACD</span>' : ''}`;
      row.onclick = () => selectCar(c);
      list.appendChild(row);
    });
}

function renderTracks(filter = "") {
  const f = filter.toLowerCase();
  const list = $("#trackList"); list.innerHTML = "";
  state.tracks.filter(t => !f || (t.name + " " + t.id + " " + t.country).toLowerCase().includes(f))
    .forEach(t => {
      const row = document.createElement("div");
      row.className = "row"; row.setAttribute("role", "option");
      row.setAttribute("aria-selected", state.track?.id === t.id);
      row.innerHTML = `
        <img class="row-badge" loading="lazy" src="/api/preview?kind=outline&track=${encodeURIComponent(t.id)}" alt="" onerror="this.style.visibility='hidden'">
        <div class="row-txt">
          <div class="row-name">${esc(t.name)}</div>
          <div class="row-sub">${t.length_m ? (t.length_m/1000).toFixed(2)+' km' : ''}${t.country ? ' · '+esc(t.country) : ''}</div>
        </div>
        ${t.has_ai ? '' : '<span class="flag no" title="no AI line — not trainable yet">NO AI</span>'}`;
      row.onclick = () => t.has_ai ? selectTrack(t) : toast("That track has no AI line to build a centreline from.", true);
      if (!t.has_ai) row.style.opacity = ".55";
      list.appendChild(row);
    });
}

function selectCar(c) {
  state.car = c; renderCars($("#carSearch").value);
  $("#carName").textContent = c.name;
  $("#carSpec").textContent = [c.brand, c.power && (c.power+" bhp"), c.weight && (c.weight)]
    .filter(Boolean).join("  ·  ") || c.id;
  setImg("#carImg", `/api/preview?kind=preview&car=${encodeURIComponent(c.id)}`);
  const cb = $("#calibBtn"); cb.hidden = false; cb.classList.remove("done"); cb.textContent = "◉ Calibrate physics";
  refreshLaunchable();
}
function selectTrack(t) {
  state.track = t; renderTracks($("#trackSearch").value);
  $("#trackName").textContent = t.name;
  $("#trackSpec").textContent = [t.length_m && (t.length_m+" m"), t.width_m && (t.width_m+" m wide"),
    t.country].filter(Boolean).join("  ·  ") || t.id;
  setImg("#trackImg", `/api/preview?kind=preview&track=${encodeURIComponent(t.id)}`);
  refreshLaunchable();
  voxel.onTrack(t);
}
function setImg(sel, src) {
  const img = $(sel); img.src = src; img.classList.remove("show");
  img.onload = () => img.classList.add("show");
  img.onerror = () => img.classList.remove("show");
}

/* ───────────────────────── presets ───────────────────────── */
function renderPresets() {
  const ag = $("#agentSeg"); ag.innerHTML = "";
  state.agentPresets.forEach(p => ag.appendChild(segBtn(p.label, p.agents.toLocaleString('en-US')+" · "+p.hint,
    state.agents === p.agents, () => { state.agents = p.agents; renderPresets(); refreshLaunchable(); })));
  const ln = $("#lengthSeg"); ln.innerHTML = "";
  state.lengthPresets.forEach(p => ln.appendChild(segBtn(p.label,
    (p.steps ? p.steps.toLocaleString('en-US')+" steps" : "no limit")+" · "+p.hint,
    state.steps === p.steps, () => { state.steps = p.steps; renderPresets(); })));
}
function segBtn(label, val, on, onclick) {
  const b = document.createElement("button");
  b.setAttribute("aria-pressed", on);
  b.innerHTML = `<span class="sg-l">${esc(label)}</span><span class="sg-v">${esc(val)}</span>`;
  b.onclick = onclick; return b;
}

/* ───────────────────────── launch (signature) ───────────────────────── */
function refreshLaunchable() {
  const btn = $("#deployBtn");
  if (state.running) return;
  const ready = state.car && state.track && state.track.has_ai;
  btn.disabled = !ready || state.busy;
  if (ready) {
    $("#launchStage").textContent = `${state.car.name}  ·  ${state.track.name}`;
    $("#launchSub").textContent = `${state.agents.toLocaleString('en-US')} agents on the GPU — AC stays smooth`;
  } else {
    $("#launchStage").textContent = state.car ? "pick a trainable track" : "pick a car and a trainable track";
    $("#launchSub").textContent = "training runs on the GPU — Assetto Corsa stays smooth";
  }
}

const lights = () => [...document.querySelectorAll(".gantry .light")];
function clearLights() { lights().forEach(l => l.classList.remove("amber", "go")); }
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function deploy() {
  if (state.running) return boxStint();
  if (!(state.car && state.track)) return;
  state.busy = true; const btn = $("#deployBtn");
  btn.classList.add("arming"); btn.textContent = "ARMING…"; clearLights();
  const stages = ["VALIDATE", "ALLOCATE", "SEED", "WARM", "IGNITE"];
  // fire the request up front; light the gantry while it spins up
  const reqP = api("/api/train", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ car: state.car.id, track: state.track.id,
                           agents: state.agents, steps: state.steps }) }).catch(() => ({ ok: false, message: "server error" }));
  for (let i = 0; i < stages.length; i++) {
    lights()[i].classList.add("amber");
    $("#launchStage").textContent = stages[i];
    $("#launchSub").textContent = launchHint(stages[i]);
    await sleep(420);
  }
  const res = await reqP;
  if (!res.ok) {
    clearLights(); state.busy = false; btn.classList.remove("arming");
    btn.textContent = "DEPLOY STINT"; toast(res.message || "Could not launch.", true);
    refreshLaunchable(); return;
  }
  // lights out -> racing
  await sleep(250); clearLights(); await sleep(180);
  lights().forEach(l => l.classList.add("go"));
  $("#launchStage").textContent = "LIGHTS OUT — RACING";
  $("#launchSub").textContent = `${state.car.name} learning ${state.track.name}`;
  swarm.ignite();
  await sleep(700); lights().forEach((l, i) => setTimeout(() => l.classList.remove("go"), i * 80));
  state.busy = false; setRunning(true);
  toast("Stint launched on the GPU.");
}

function launchHint(s) {
  return { VALIDATE: "checking car physics + track AI line",
           ALLOCATE: "reserving GPU memory for the agent batch",
           SEED: "spawning agents across the grid",
           WARM: "filling the replay buffer",
           IGNITE: "starting QR-SAC updates" }[s] || "";
}

async function boxStint() {
  const btn = $("#deployBtn"); btn.disabled = true;
  const res = await api("/api/stop", { method: "POST" });
  toast(res.message || "Boxing.");
}

function setRunning(on) {
  state.running = on;
  $("#hdrStop").hidden = !on;
  $("#liveChip").dataset.on = on; $("#liveState").textContent = on ? "on track" : "idle";
  $("#telemetry").dataset.on = on;
  const btn = $("#deployBtn");
  btn.classList.toggle("box", on); btn.disabled = false;
  btn.textContent = on ? "BOX STINT" : "DEPLOY STINT";
  if (on) { track.start(); }
  else { clearLights(); swarm.cool(); track.stop(); refreshLaunchable(); }
}

/* ───────────────────────── telemetry poll ───────────────────────── */
let lastStep = -1;
async function poll() {
  try {
    const s = await api("/api/status");
    const running = !!s.running;
    if (running !== state.running) setRunning(running);
    if ($("#devState").textContent === "—")
      $("#devState").textContent = (s.live && s.live.tps) ? "CUDA" : (running ? "CUDA" : "—");
    if (running && s.live) renderTelemetry(s.live, s.meta);
  } catch (e) { /* server momentarily busy */ }
  setTimeout(poll, 600);
}

function renderTelemetry(L, meta) {
  $("#devState").textContent = "CUDA";
  $("#roStep").textContent = (L.step || 0).toLocaleString('en-US');
  $("#roReward").textContent = fmt(L.reward, 3);
  $("#roBest").textContent = L.lap_best ? L.lap_best.toFixed(1) + "s" : "—";
  $("#roAvg").textContent = L.lap_avg ? L.lap_avg.toFixed(1) + "s" : "—";
  $("#roAgents").textContent = (L.cars_total || meta.agents || 0).toLocaleString('en-US');
  $("#roCrash").textContent = (L.crashes ?? 0) + " / " + (L.cars_total || 0).toLocaleString('en-US');
  $("#roSpeed").textContent = fmt(L.avgspeed, 1) + " km/h";
  $("#roTps").textContent = L.tps ? (L.tps / 1000).toFixed(0) + "k/s" : "—";
  // explore→exploit from alpha (0.1 floor .. ~0.5 high)
  const ex = Math.max(0, Math.min(1, ((L.alpha || 0.1) - 0.1) / 0.4));
  $("#exFill").style.width = (12 + ex * 80).toFixed(0) + "%";
  // phase
  const ph = L.warmup ? "SEEDING" : (L.reward > 2.2 ? "CONVERGED" : (L.reward > 1.0 ? "REFINING" : "EXPLORING"));
  const pe = $("#phase"); pe.textContent = ph; pe.dataset.p = ph;
  drawScope(L.history || []);
  if (L.lead) drawWheel(L.lead);
  if (L.step !== lastStep) { swarm.bump(L.crashes || 0); lastStep = L.step; }
}

/* ───────────── fastest car's live steering wheel + pedals ───────────── */
let wheelAngle = 0;
function drawWheel(lead) {
  const steer = Math.max(-1, Math.min(1, lead.steer || 0));     // -1..1 policy steer
  const LOCK = 200;                                             // deg of wheel travel at full steer
  const target = steer * LOCK * Math.PI / 180;
  wheelAngle += (target - wheelAngle) * 0.35;                   // smooth toward the live value
  const c = $("#wheelCanvas"); if (!c) return;
  const ctx = c.getContext("2d"), W = c.width, H = c.height, cx = W / 2, cy = H / 2, R = 60;
  ctx.clearRect(0, 0, W, H);
  ctx.save(); ctx.translate(cx, cy); ctx.rotate(wheelAngle);
  // rim
  ctx.lineWidth = 9; ctx.strokeStyle = "#1c2330";
  ctx.beginPath(); ctx.arc(0, 0, R, 0, 2 * Math.PI); ctx.stroke();
  const grad = steer > 0 ? "#40e0ff" : "#ff568a";
  ctx.lineWidth = 5; ctx.strokeStyle = "#39455b";
  ctx.beginPath(); ctx.arc(0, 0, R, 0, 2 * Math.PI); ctx.stroke();
  // top marker (12 o'clock reference, colored by steer direction)
  ctx.fillStyle = Math.abs(steer) < 0.02 ? "#5d6b7e" : grad;
  ctx.beginPath(); ctx.arc(0, -R, 5.5, 0, 2 * Math.PI); ctx.fill();
  // spokes
  ctx.strokeStyle = "#39455b"; ctx.lineWidth = 5; ctx.lineCap = "round";
  for (const a of [Math.PI / 2, Math.PI * 7 / 6, Math.PI * 11 / 6]) {
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(Math.cos(a) * (R - 6), Math.sin(a) * (R - 6)); ctx.stroke();
  }
  ctx.fillStyle = "#222c3a"; ctx.beginPath(); ctx.arc(0, 0, 13, 0, 2 * Math.PI); ctx.fill();
  ctx.restore();
  $("#wheelDeg").textContent = Math.round(steer * LOCK) + "°";
  $("#wheelDeg").style.color = Math.abs(steer) < 0.02 ? "#8a9aad" : grad;
  $("#pedThr").style.width = Math.round((lead.thr || 0) * 100) + "%";
  $("#pedBrk").style.width = Math.round((lead.brk || 0) * 100) + "%";
  $("#wheelSpd").textContent = Math.round(lead.spd || 0) + " km/h";
}

/* ───────────────────────── reward scope ───────────────────────── */
function drawScope(hist) {
  const c = $("#scope"), ctx = c.getContext("2d");
  const w = c.width = c.clientWidth * devicePixelRatio, h = c.height = 120 * devicePixelRatio;
  ctx.clearRect(0, 0, w, h);
  // grid
  ctx.strokeStyle = "rgba(40,53,70,.5)"; ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) { const y = h * i / 4; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  if (hist.length < 2) return;
  const mn = Math.min(...hist), mx = Math.max(...hist), rng = (mx - mn) || 1;
  const x = i => i / (hist.length - 1) * w;
  const y = v => h - 8 - (v - mn) / rng * (h - 16);
  // glow fill
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(255,178,56,.22)"); grad.addColorStop(1, "rgba(255,178,56,0)");
  ctx.beginPath(); ctx.moveTo(0, h);
  hist.forEach((v, i) => ctx.lineTo(x(i), y(v))); ctx.lineTo(w, h); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();
  // line
  ctx.beginPath(); hist.forEach((v, i) => i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v)));
  ctx.strokeStyle = "#ffb238"; ctx.lineWidth = 2 * devicePixelRatio;
  ctx.shadowColor = "rgba(255,178,56,.6)"; ctx.shadowBlur = 8; ctx.stroke(); ctx.shadowBlur = 0;
  // head dot
  const lx = x(hist.length - 1), ly = y(hist[hist.length - 1]);
  ctx.beginPath(); ctx.arc(lx, ly, 3 * devicePixelRatio, 0, 7); ctx.fillStyle = "#fff"; ctx.fill();
}

/* ───────────────────────── swarm field (signature) ───────────────────────── */
const swarm = (() => {
  const c = $("#swarm"); const ctx = c.getContext("2d");
  let dots = [], on = false, crashFlash = 0, raf = null;
  function resize() { c.width = c.clientWidth * devicePixelRatio; c.height = (c.clientHeight || 150) * devicePixelRatio; }
  function seed() {
    resize(); dots = [];
    const N = 300;
    for (let i = 0; i < N; i++) dots.push({
      x: Math.random() * c.width, y: Math.random() * c.height,
      vx: (Math.random() - .5) * .6 * devicePixelRatio, vy: (Math.random() - .5) * .6 * devicePixelRatio,
      r: (Math.random() * 1.4 + .8) * devicePixelRatio, life: Math.random()
    });
  }
  function frame() {
    ctx.clearRect(0, 0, c.width, c.height);
    dots.forEach(d => {
      if (on) { d.x += d.vx; d.y += d.vy;
        if (d.x < 0 || d.x > c.width) d.vx *= -1; if (d.y < 0 || d.y > c.height) d.vy *= -1;
        d.life += .015; }
      const a = on ? .35 + .35 * Math.sin(d.life * 6) : .2;
      const crashed = crashFlash > 0 && Math.random() < .004 * crashFlash;
      ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, 7);
      ctx.fillStyle = crashed ? `rgba(255,86,138,.9)` : `rgba(64,224,255,${a})`;
      ctx.fill();
    });
    if (crashFlash > 0) crashFlash -= .04;
    raf = requestAnimationFrame(frame);
  }
  seed(); window.addEventListener("resize", seed);
  if (!raf) frame();
  return {
    ignite() { on = true; },
    cool() { on = false; },
    bump(crashes) { crashFlash = Math.min(2, crashes / 8); },
  };
})();

/* ───────────────────────── GPU usage ───────────────────────── */
async function gpuPoll() {
  try {
    const g = await api("/api/gpu");
    if (g.ok) {
      $("#gpuFill").style.width = g.util + "%";
      $("#gpuState").textContent = `${g.util}% · ${(g.mem_used/1024).toFixed(1)}/${(g.mem_total/1024).toFixed(0)}G · ${g.temp}°`;
      $("#gpuChip").title = `${g.name} — ${g.util}% util · ${g.mem_used}/${g.mem_total} MiB · ${g.temp}°C`;
    } else {
      $("#gpuState").textContent = "n/a";
    }
  } catch (e) { /* server busy */ }
  setTimeout(gpuPoll, 1500);
}

/* ───────────────────────── live track view ───────────────────────── */
const track = (() => {
  const c = $("#trackCanvas"); const ctx = c.getContext("2d");
  let geom = null, fit = null, on = false, timer = null, lastStep = -1;

  function computeFit() {
    const pts = (geom.walls && geom.walls.length ? geom.walls : geom.center);
    if (!pts || !pts.length) return null;
    let mnx = 1e9, mxx = -1e9, mnz = 1e9, mxz = -1e9;
    for (const [x, z] of pts) { mnx = Math.min(mnx, x); mxx = Math.max(mxx, x); mnz = Math.min(mnz, z); mxz = Math.max(mxz, z); }
    const w = c.width, h = c.height, pad = 26 * devicePixelRatio;
    const sx = (w - pad * 2) / ((mxx - mnx) || 1), sz = (h - pad * 2) / ((mxz - mnz) || 1);
    const s = Math.min(sx, sz);
    const ox = (w - (mxx - mnx) * s) / 2, oz = (h - (mxz - mnz) * s) / 2;
    return { mnx, mnz, s, ox, oz, h };
    }
  function X(x) { return fit.ox + (x - fit.mnx) * fit.s; }
  function Y(z) { return fit.h - (fit.oz + (z - fit.mnz) * fit.s); }  // flip z so it reads like the map

  function ensureSize() {
    // self-heal: when the panel flips from hidden->visible the canvas can briefly report 0 size.
    const w = Math.round(c.clientWidth * devicePixelRatio), h = Math.round(c.clientHeight * devicePixelRatio);
    if (w > 0 && h > 0 && (Math.abs(c.width - w) > 1 || Math.abs(c.height - h) > 1)) {
      c.width = w; c.height = h; fit = geom ? computeFit() : null;
    } else if (!fit && geom && c.width > 0) { fit = computeFit(); }
  }
  window.addEventListener("resize", ensureSize);

  async function loadGeom() {
    geom = await api("/api/track").catch(() => null);
    ensureSize();
  }

  function speedColor(t) {  // 0 slow(magenta) -> .5 amber -> 1 cyan
    t = Math.max(0, Math.min(1, t));
    const a = [255, 86, 138], b = [255, 178, 56], d = [64, 224, 255];
    const lerp = (p, q, k) => p.map((v, i) => Math.round(v + (q[i] - v) * k));
    const rgb = t < .5 ? lerp(a, b, t / .5) : lerp(b, d, (t - .5) / .5);
    return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  }

  function drawBase() {
    ctx.clearRect(0, 0, c.width, c.height);
    if (!geom || !fit) return;
    // walls as a corridor of points (the track shape)
    if (geom.walls && geom.walls.length) {
      ctx.fillStyle = "rgba(58,76,98,.8)";
      for (const [x, z] of geom.walls) { ctx.beginPath(); ctx.arc(X(x), Y(z), 1.3 * devicePixelRatio, 0, 7); ctx.fill(); }
    }
    // centreline (the racing reference)
    if (geom.center && geom.center.length) {
      ctx.beginPath();
      geom.center.forEach(([x, z], i) => i ? ctx.lineTo(X(x), Y(z)) : ctx.moveTo(X(x), Y(z)));
      ctx.closePath();
      ctx.strokeStyle = "rgba(120,150,176,.3)"; ctx.lineWidth = 1.25 * devicePixelRatio;
      ctx.setLineDash([4 * devicePixelRatio, 6 * devicePixelRatio]); ctx.stroke(); ctx.setLineDash([]);
    }
  }

  async function tick() {
    if (!on) return;
    try {
      ensureSize();
      const d = await api("/api/cars");
      drawBase();
      const vmax = d.vmax || geom?.vmax || 24.5;
      const ks = d.cars || [];
      ctx.shadowBlur = 6 * devicePixelRatio;
      for (const k of ks) {
        const [x, z, , spd, grip] = k;
        const col = speedColor(spd / vmax);
        ctx.beginPath(); ctx.arc(X(x), Y(z), 2.7 * devicePixelRatio, 0, 7);
        ctx.fillStyle = col; ctx.shadowColor = col;
        ctx.globalAlpha = grip != null && grip < .5 ? .55 : 1;   // dim the ones sliding/off-track
        ctx.fill();
      }
      ctx.shadowBlur = 0; ctx.globalAlpha = 1;
      $("#tvMeta").textContent = `${ks.length} agents shown · step ${(d.step || 0).toLocaleString('en-US')}`;
    } catch (e) { /* between writes */ }
    timer = setTimeout(tick, 220);
  }

  return {
    async start() {
      on = true; $("#trackview").dataset.on = "true";
      await loadGeom();                 // reload each stint — the track may have changed
      await sleep(60);                 // let the now-visible canvas get real dimensions
      ensureSize(); clearTimeout(timer); tick();
    },
    stop() { on = false; clearTimeout(timer); $("#trackview").dataset.on = "false"; },
  };
})();

/* ───────────────────────── views + garage ───────────────────────── */
let curView = "launch", garageTimer = null;
function setView(v) {
  curView = v;
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.view === v));
  $("#launchView").hidden = v !== "launch";
  $("#garageView").hidden = v !== "garage";
  clearTimeout(garageTimer);
  if (v === "garage") loadPolicies();
}
function fmtLap(s) {
  if (!s) return "—";
  const m = Math.floor(s / 60), r = (s % 60).toFixed(1);
  return m ? `${m}:${r.padStart(4, "0")}` : `${(+s).toFixed(1)}s`;
}
function ago(ts) {
  if (!ts) return "—";
  const d = Date.now() / 1000 - ts;
  if (d < 90) return "just now";
  if (d < 5400) return Math.round(d / 60) + "m ago";
  if (d < 172800) return Math.round(d / 3600) + "h ago";
  return Math.round(d / 86400) + "d ago";
}
const carName = (id) => (state.cars.find(c => c.id === id) || {}).name || id;
const trackName = (id) => (state.tracks.find(t => t.id === id) || {}).name || id;

async function loadPolicies() {
  const d = await api("/api/policies").catch(() => null);
  if (!d) return;
  $("#garageAgg").innerHTML = `<b>${d.count}</b> policies · <b>${(d.total_steps / 1000).toFixed(0)}k</b> steps · <b>${d.total_hours.toFixed(1)}</b> GPU-hours`;
  const grid = $("#garageGrid"); grid.innerHTML = "";
  $("#garageEmpty").hidden = d.count > 0;
  d.policies.forEach(p => grid.appendChild(policyCard(p)));
  if (curView === "garage") garageTimer = setTimeout(loadPolicies, 2500);
}

function mkbtn(label, cls, fn) {
  const b = document.createElement("button"); b.className = cls; b.textContent = label; b.onclick = fn; return b;
}
function policyCard(p) {
  const el = document.createElement("div");
  el.className = "pcard" + (p.live ? " live" : "");
  el.innerHTML = `
    <div class="pcard-top">
      <img class="pcard-badge" src="/api/preview?kind=badge&car=${encodeURIComponent(p.car)}" alt="" onerror="this.style.visibility='hidden'">
      <div class="pcard-id"><div class="pcard-car">${esc(carName(p.car))}</div><div class="pcard-track">${esc(trackName(p.track))}</div></div>
      ${p.live ? '<span class="pcard-live">LIVE</span>' : ''}
    </div>
    <div class="pcard-stats">
      <div class="ps"><span class="ps-k">BEST LAP</span><span class="ps-v">${fmtLap(p.best_lap)}</span></div>
      <div class="ps"><span class="ps-k">REWARD</span><span class="ps-v">${(p.reward ?? 0).toFixed(2)}</span></div>
      <div class="ps"><span class="ps-k">STEPS</span><span class="ps-v">${(p.sim_steps / 1000).toFixed(0)}k</span></div>
      <div class="ps"><span class="ps-k">CRASH/STEP</span><span class="ps-v">${((p.crash_rate || 0) * 100).toFixed(1)}%</span></div>
    </div>
    <div class="pcard-foot">
      <span class="pcard-when">${p.agents ? (p.agents / 1000).toFixed(0) + 'k agents · ' : ''}${ago(p.trained_at)}${p.calibrated ? ' · μ✓' : ''}</span>
    </div>`;
  const foot = el.querySelector(".pcard-foot");
  if (p.live) {
    foot.appendChild(mkbtn("■ Stop", "pbtn stop", stopFromGarage));
  } else {
    foot.appendChild(mkbtn("Resume", "pbtn", () => resumePolicy(p)));
    const del = mkbtn("Delete", "pbtn del", () => deletePolicy(p)); del.disabled = p.protected; foot.appendChild(del);
  }
  return el;
}
async function resumePolicy(p) {
  const r = await api("/api/train", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ car: p.car, track: p.track, agents: p.agents || 8000, steps: 0 }) });
  if (!r.ok) return toast(r.message, true);
  toast("Resuming " + carName(p.car) + " · " + trackName(p.track)); loadPolicies();
}
async function deletePolicy(p) {
  if (!confirm("Delete this policy and its checkpoint?\n" + carName(p.car) + " · " + trackName(p.track))) return;
  const r = await api("/api/policy/delete", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key: p.key }) });
  toast(r.message, !r.ok); loadPolicies();
}
async function stopFromGarage() { const r = await api("/api/stop", { method: "POST" }); toast(r.message); setTimeout(loadPolicies, 700); }

/* ───────────────────────── calibration ───────────────────────── */
const calib = (() => {
  let car = null, timer = null;
  function setStats(s) {
    $("#clFrames").textContent = s.frames ?? 0;
    $("#clSecs").textContent = Math.round(s.secs || 0) + "s";
    $("#clSpeed").textContent = (s.speed || 0).toFixed(0) + " km/h";
    const fe = $("#clFeed");
    if (s.feed_ok === null || s.feed_ok === undefined) { fe.textContent = "…"; fe.className = "cl-v"; }
    else { fe.textContent = s.feed_ok ? "live" : "off"; fe.className = "cl-v " + (s.feed_ok ? "good" : "bad"); }
  }
  async function pollStatus() {
    const s = await api("/api/calibrate/status").catch(() => null);
    if (s) setStats(s);
    timer = setTimeout(pollStatus, 400);
  }
  function open(c) {
    car = c; $("#calibTitle").textContent = c.name;
    $("#calibResult").hidden = true; $("#calibResult").innerHTML = "";
    $("#calibStart").hidden = false; $("#calibStop").hidden = true;
    setStats({ frames: 0, secs: 0, speed: 0, feed_ok: null });
    $("#calibModal").dataset.on = "true";
    clearTimeout(timer); pollStatus();
  }
  function close() { $("#calibModal").dataset.on = "false"; clearTimeout(timer); }
  async function start() {
    const r = await api("/api/calibrate/start", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ car: car.id }) });
    if (!r.ok) return toast(r.message, true);
    toast(r.message); $("#calibStart").hidden = true; $("#calibStop").hidden = false;
  }
  async function stop() {
    $("#calibStop").disabled = true;
    const r = await api("/api/calibrate/stop", { method: "POST" });
    $("#calibStop").disabled = false; $("#calibStop").hidden = true; $("#calibStart").hidden = false;
    if (!r.ok) return toast((r.report && r.report.error) || "Not enough data — drive longer.", true);
    renderResult(r.before, r.after, r.report || {});
    $("#calibBtn").classList.add("done"); $("#calibBtn").textContent = "◉ Calibrated";
    toast("Calibrated " + car.name + " from telemetry.");
  }
  function renderResult(b, a, rep) {
    const rows = [["Peak accel", "a_accel", " m/s²", 1, x => x],
                  ["Braking", "a_brake", " m/s²", 1, x => x],
                  ["Grip (μ)", "grip", "", 2, x => x],
                  ["Top speed", "vmax_ms", " km/h", 0, x => x * 3.6],
                  ["Steer lock", "steer_lock", " rad", 2, x => x]];
    let html = `<h4>Measured vs estimate · ${rep.n_accel || 0} accel / ${rep.n_brake || 0} brake / ${rep.n_corner || 0} corner samples</h4>`;
    for (const [lab, key, unit, dp, fn] of rows) {
      const bv = fn(b[key]), av = fn(a[key]);
      const dir = av > bv * 1.01 ? "up" : (av < bv * 0.99 ? "down" : "");
      html += `<div class="cr-row"><span class="lab">${lab}</span><span class="was">${bv.toFixed(dp)}${unit}</span><span class="now ${dir}">${av.toFixed(dp)}${unit}</span></div>`;
    }
    const el = $("#calibResult"); el.innerHTML = html; el.hidden = false;
  }
  return { open, close, start, stop };
})();

/* ───────── track surface: EXACT 3D triangle mesh from the real .kn5 (WebGL2, not voxels) ───────── */
const voxel = (() => {
  let timer = null, building = false, lastTrack = null, colors = null, loadedKey = null, loading = false;
  const cam = { yaw: 0.7, pitch: 0.62, dist: 2.7 };     // orbit camera
  let gl = null, prog = null, vao = null, nIndex = 0, center = [0, 0, 0], modelScale = 1, drag = null;

  const panel = () => $("#voxview");
  const show = (on) => { panel().dataset.on = on ? "true" : "false"; };

  function onTrack(t) {
    lastTrack = t; show(true);
    $("#voxBtn").disabled = false; $("#voxBtn").textContent = "Build 3D surface";
    $("#voxStat").textContent = `Build ${t.name}'s exact 3D surface from its mesh — road, grass, kerbs, real walls. Drag to orbit.`;
    if (loadedKey !== t.id) { loadedKey = null; nIndex = 0; if (gl) clearGL(); }
    poll();
  }

  async function build() {
    if (!lastTrack || building) return;
    const r = await api("/api/voxelize", { method: "POST",
      body: JSON.stringify({ track: lastTrack.id }) }).catch(() => ({ ok: false, message: "server error" }));
    toast(r.message, !r.ok);
    if (r.ok) { building = true; loadedKey = null; $("#voxBtn").disabled = true; $("#voxBtn").textContent = "Building…"; start(); }
  }

  function start() { stop(); timer = setInterval(poll, 500); poll(); }
  function stop() { if (timer) clearInterval(timer); timer = null; }

  async function poll() {
    const d = await api("/api/voxels").catch(() => null);
    if (!d) return;
    if (d.colors) colors = d.colors;
    if (d.legend && d.counts) renderLegend(d.legend, d.counts);
    $("#voxFill").style.width = Math.round((d.progress || 0) * 100) + "%";
    if (d.msg) $("#voxStat").textContent = d.msg;
    if (d.running) {
      building = true; $("#voxBtn").disabled = true; $("#voxBtn").textContent = "Building…";
      if (!timer) start();
    } else {
      building = false; $("#voxBtn").disabled = false;
      $("#voxBtn").textContent = "Rebuild";
      stop();
      if (lastTrack && loadedKey !== lastTrack.id && !loading) loadMesh(lastTrack.id);
    }
  }

  // ── load the exact triangle mesh and upload to the GPU ──
  async function loadMesh(trackId) {
    loading = true;
    try {
      const hdr = await api(`/api/mesh?track=${encodeURIComponent(trackId)}&part=header`).catch(() => null);
      if (!hdr || hdr.ready === false || !hdr.nverts) { loading = false; return; }
      $("#voxStat").textContent = `Loading ${(hdr.ntris / 1e6).toFixed(1)}M-triangle surface…`;
      const buf = await fetch(`/api/mesh?track=${encodeURIComponent(trackId)}&part=bin`).then(r => r.arrayBuffer());
      const pos = new Float32Array(buf, 0, hdr.nverts * 3);
      const col = new Uint8Array(buf, hdr.pos_bytes, hdr.nverts * 3);
      const io = hdr.pos_bytes + hdr.col_bytes;
      const idx = new Uint32Array(buf.slice(io, io + hdr.idx_bytes));   // slice -> 4-byte aligned
      center = [(hdr.bmin[0] + hdr.bmax[0]) / 2, (hdr.bmin[1] + hdr.bmax[1]) / 2, (hdr.bmin[2] + hdr.bmax[2]) / 2];
      const ext = Math.max(hdr.bmax[0] - hdr.bmin[0], hdr.bmax[1] - hdr.bmin[1], hdr.bmax[2] - hdr.bmin[2]);
      modelScale = 2.0 / (ext || 1);
      upload(pos, col, idx); nIndex = idx.length; loadedKey = trackId;
      $("#voxStat").textContent = `${hdr.ntris.toLocaleString('en-US')} triangles · exact mesh · drag to orbit, scroll to zoom`;
      draw();
    } catch (e) { $("#voxStat").textContent = "mesh load failed: " + e.message; }
    loading = false;
  }

  // ── WebGL2 setup ──
  function initGL() {
    const cv = $("#voxCanvas");
    gl = cv.getContext("webgl2", { antialias: true, depth: true });
    if (!gl) { $("#voxStat").textContent = "WebGL2 not available in this browser."; return false; }
    const vs = `#version 300 es
      layout(location=0) in vec3 aPos; layout(location=1) in vec3 aCol;
      uniform mat4 uMVP, uMV; out vec3 vCol; out vec3 vView;
      void main(){ vCol=aCol; vView=(uMV*vec4(aPos,1.0)).xyz; gl_Position=uMVP*vec4(aPos,1.0); }`;
    const fs = `#version 300 es
      precision highp float; in vec3 vCol; in vec3 vView; out vec4 o;
      void main(){
        vec3 n = normalize(cross(dFdx(vView), dFdy(vView)));
        vec3 L = normalize(vec3(0.35,0.75,0.55));
        float d = max(dot(n,L),0.0);
        o = vec4(vCol*(0.45+0.75*d), 1.0);
      }`;
    const sh = (t, s) => { const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o);
      if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(o)); return o; };
    prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);
    gl.enable(gl.DEPTH_TEST);
    return true;
  }

  function upload(pos, col, idx) {
    if (!gl && !initGL()) return;
    if (vao) gl.deleteVertexArray(vao);
    vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    const pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb);
    gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    const cb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, cb);
    gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.UNSIGNED_BYTE, true, 0, 0);  // normalized
    const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
  }

  function clearGL() { if (gl && vao) { gl.deleteVertexArray(vao); vao = null; nIndex = 0; draw(); } }

  // ── minimal column-major mat4 ──
  const M = {
    mul: (a, b) => { const o = new Float32Array(16);
      for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++)
        o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
      return o; },
    persp: (fov, asp, n, f) => { const t = 1 / Math.tan(fov / 2), o = new Float32Array(16);
      o[0] = t / asp; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = 2 * f * n / (n - f); return o; },
    trans: (x, y, z) => { const o = new Float32Array(16); o[0] = o[5] = o[10] = o[15] = 1; o[12] = x; o[13] = y; o[14] = z; return o; },
    scale: (s) => { const o = new Float32Array(16); o[0] = o[5] = o[10] = s; o[15] = 1; return o; },
    rotX: (a) => { const c = Math.cos(a), s = Math.sin(a), o = new Float32Array(16); o[0] = o[15] = 1; o[5] = c; o[6] = s; o[9] = -s; o[10] = c; return o; },
    rotY: (a) => { const c = Math.cos(a), s = Math.sin(a), o = new Float32Array(16); o[5] = o[15] = 1; o[0] = c; o[2] = -s; o[8] = s; o[10] = c; return o; },
  };

  function draw() {
    if (!gl) return;
    const cv = $("#voxCanvas"), wrap = cv.parentElement;
    const cw = wrap.clientWidth || 720, ch = Math.max(320, Math.round(cw * 0.54));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = cw * dpr; cv.height = ch * dpr;
    gl.viewport(0, 0, cv.width, cv.height);
    gl.clearColor(0.03, 0.04, 0.06, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!vao || !nIndex) return;
    // model: centre + scale to unit; AC is Y-up. view: orbit (yaw,pitch) + dolly back.
    const model = M.mul(M.scale(modelScale), M.trans(-center[0], -center[1], -center[2]));
    const view = M.mul(M.trans(0, 0, -cam.dist), M.mul(M.rotX(cam.pitch), M.rotY(cam.yaw)));
    const mv = M.mul(view, model);
    const mvp = M.mul(M.persp(0.82, cv.width / cv.height, 0.05, 50), mv);
    gl.useProgram(prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, "uMVP"), false, mvp);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, "uMV"), false, mv);
    gl.bindVertexArray(vao);
    gl.drawElements(gl.TRIANGLES, nIndex, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(null);
  }

  function renderLegend(legend, counts) {
    const order = ["road", "kerb", "grass", "sand", "pit", "wall"];
    const byName = {}; Object.entries(legend).forEach(([k, v]) => byName[v] = k);
    $("#voxLegend").innerHTML = order.filter(n => counts[n]).map(n => {
      const c = (colors || {})[byName[n]] || [120, 120, 120];
      return `<i class="vx" style="background:rgb(${c[0]},${c[1]},${c[2]})"></i>${n}`;
    }).join("");
  }

  // ── orbit / zoom ──
  const cv = $("#voxCanvas");
  cv.addEventListener("mousedown", e => { drag = { x: e.clientX, y: e.clientY, yaw: cam.yaw, pitch: cam.pitch }; });
  window.addEventListener("mousemove", e => {
    if (!drag) return;
    cam.yaw = drag.yaw + (e.clientX - drag.x) * 0.01;
    cam.pitch = Math.max(-0.2, Math.min(1.5, drag.pitch + (e.clientY - drag.y) * 0.008));
    draw();
  });
  window.addEventListener("mouseup", () => { drag = null; });
  cv.addEventListener("wheel", e => {
    e.preventDefault();
    cam.dist = Math.max(0.8, Math.min(8, cam.dist * (e.deltaY < 0 ? 0.9 : 1.11)));
    draw();
  }, { passive: false });
  window.addEventListener("resize", () => { if (nIndex) draw(); });

  $("#voxBtn").addEventListener("click", build);
  return { onTrack, poll };
})();

/* ───────────────────────── misc ───────────────────────── */
function esc(s) { return String(s ?? "").replace(/[&<>"]/g, m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[m])); }
function fmt(v, d) { return (v ?? 0).toFixed(d); }
let toastT;
function toast(msg, err) {
  let el = $(".toast"); if (!el) { el = document.createElement("div"); el.className = "toast"; document.body.appendChild(el); }
  el.textContent = msg; el.classList.toggle("err", !!err); el.classList.add("show");
  clearTimeout(toastT); toastT = setTimeout(() => el.classList.remove("show"), 3200);
}

$("#carSearch").addEventListener("input", e => renderCars(e.target.value));
$("#trackSearch").addEventListener("input", e => renderTracks(e.target.value));
$("#deployBtn").addEventListener("click", deploy);
$("#calibBtn").addEventListener("click", () => state.car && calib.open(state.car));
$("#calibClose").addEventListener("click", calib.close);
$("#calibStart").addEventListener("click", calib.start);
$("#calibStop").addEventListener("click", calib.stop);
$("#calibModal").addEventListener("click", (e) => { if (e.target.id === "calibModal") calib.close(); });
document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => setView(t.dataset.view)));
$("#hdrStop").addEventListener("click", async () => {
  const r = await api("/api/stop", { method: "POST" }); toast(r.message);
  if (curView === "garage") setTimeout(loadPolicies, 700);
});

loadLibrary().then(() => { poll(); if (location.hash === "#garage") setView("garage"); });
gpuPoll();
