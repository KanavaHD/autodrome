"""
PIT WALL — local control server for the AC GPU training app.

Scans your Assetto Corsa install, launches the all-GPU QR-SAC trainer (train_sim_gpu.py) on the
car + track + preset you pick, and streams live status. Training runs entirely in the GPU sim, in
its own process — it never touches AC, so AC won't lag while a stint trains.

    python app/server.py            # then open http://127.0.0.1:8077

Pure stdlib (no pip installs). One trainer process at a time.
"""
import os, sys, json, signal, subprocess, threading, mimetypes, time, functools, glob
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # project root (car-viewer)
WEB = os.path.join(HERE, "web")
ML_DIR = os.path.join(ROOT, "ml")
LIVE = os.path.join(ML_DIR, "sim_live.json")
WALLS = os.path.join(ML_DIR, "sim_walls.json")
VOXEL_LIVE = os.path.join(ML_DIR, "voxel_live.json")
TRACKS_DIR = os.path.join(ML_DIR, "tracks")
REG_DIR = os.path.join(ML_DIR, "policies")
CAR_KEY = ""                                          # no protected policy in this build
LOGDIR = os.path.join(ROOT, "data")
PORT = int(os.environ.get("PITWALL_PORT", "8077"))
PY = sys.executable

sys.path.insert(0, HERE)
import scan
sys.path.insert(0, os.path.join(ROOT, "ml"))
import calibrate_car, car_ingest

# presets: (label, agents) and (label, steps). 'Full send' = the 20k run you've been using.
AGENT_PRESETS = [
    {"id": "recon",    "label": "Recon",     "agents": 2000,  "hint": "light GPU, quick sanity"},
    {"id": "standard", "label": "Standard",  "agents": 8000,  "hint": "balanced"},
    {"id": "fullsend", "label": "Full Send", "agents": 20000, "hint": "1080 Ti sweet spot"},
    {"id": "max",      "label": "Max",       "agents": 32000, "hint": "saturate the card"},
]
LENGTH_PRESETS = [
    {"id": "sprint", "label": "Sprint", "steps": 5000,  "hint": "~few min"},
    {"id": "stint",  "label": "Stint",  "steps": 20000, "hint": "~20–40 min"},
    {"id": "enduro", "label": "Enduro", "steps": 60000, "hint": "deep convergence"},
    {"id": "open",   "label": "Open",   "steps": 0,     "hint": "until you box it"},
]

_proc = None              # the running trainer subprocess
_proc_meta = {}           # what it's training
_lock = threading.Lock()


def _trainer_alive():
    return _proc is not None and _proc.poll() is None


def _feed_running():
    """A trainer is live if sim_live.json is fresh (the trainer rewrites it ~4x/sec) and says so.
    This adopts a stint even if THIS server didn't launch it (e.g. after a server restart), so the
    UI never shows 'idle' while the GPU is clearly training."""
    try:
        if time.time() - os.path.getmtime(LIVE) > 3.0:
            return False
        with open(LIVE) as f:
            return bool(json.load(f).get("running"))
    except Exception:
        return False


def _any_trainer():
    return _trainer_alive() or _feed_running()


def _stray_trainer_pids(except_pid=None):
    """Any train_sim_gpu.py python process (not ours). Two trainers writing sim_live.json + saving
    the same checkpoint is exactly what makes the telemetry flicker — we never allow it."""
    if os.name != "nt":
        return []
    try:
        ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              "Where-Object { $_.CommandLine -like '*train_sim_gpu*' } | "
              "Select-Object -ExpandProperty ProcessId")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=8).stdout
        pids = [int(x) for x in out.split() if x.strip().isdigit()]
        return [p for p in pids if p != except_pid]
    except Exception:
        return []


def _kill_strays(except_pid=None):
    for pid in _stray_trainer_pids(except_pid):
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=8)
        except Exception:
            pass


def gpu_stats():
    """Live GPU usage via nvidia-smi (util %, memory, temp). None if no NVIDIA GPU / driver."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=6).stdout.strip()
        name, util, used, total, temp = [x.strip() for x in out.split(",")]
        return {"ok": True, "name": name, "util": int(util), "mem_used": int(used),
                "mem_total": int(total), "temp": int(temp)}
    except Exception:
        return {"ok": False}


@functools.lru_cache(maxsize=1)
def _track_geom_cached(_mtime):
    with open(WALLS) as f:
        d = json.load(f)
    return {"walls": d.get("walls", []), "center": d.get("center", []), "vmax": d.get("vmax", 24.5)}


def track_geom():
    try:
        return _track_geom_cached(os.path.getmtime(WALLS))
    except Exception:
        return {"walls": [], "center": [], "vmax": 24.5}


def cars_live():
    """The agent swarm positions for the live track view: [[x, z, heading_deg, speed, grip], ...]."""
    try:
        with open(LIVE) as f:
            d = json.load(f)
        return {"running": _any_trainer(), "cars": d.get("cars", []),
                "vmax": d.get("vmax", 24.5), "step": d.get("step", 0)}
    except Exception:
        return {"running": False, "cars": []}


# ── per-car calibration: record the CSP feed while the user drives, then fit measured physics ──
_calib = {"recording": False, "car": None, "frames": [], "thread": None, "started": 0.0, "last": None}


def _calib_loop():
    last = None
    while _calib["recording"]:
        f = calibrate_car.read_gt()
        if f and f["frame"] != last:
            last = f["frame"]; f["t"] = time.time() - _calib["started"]
            _calib["frames"].append(f); _calib["last"] = f
        time.sleep(0.015)


def calib_start(car):
    if _calib["recording"]:
        return False, "Already calibrating."
    if calibrate_car.read_gt() is None:
        return False, "No telemetry — start AC, get on track, and enable the Car GT Logger app."
    _calib.update(recording=True, car=car, frames=[], started=time.time(), last=None)
    th = threading.Thread(target=_calib_loop, daemon=True); _calib["thread"] = th; th.start()
    return True, "Recording. Drive hard: full-throttle pulls, hard braking, hard corners."


def calib_status():
    f = _calib["last"]
    return {"recording": _calib["recording"], "car": _calib["car"], "frames": len(_calib["frames"]),
            "secs": round(time.time() - _calib["started"], 1) if _calib["recording"] else 0.0,
            "speed": round(f["speed"] * 3.6, 1) if f else 0.0,
            "feed_ok": calibrate_car.read_gt() is not None}


def calib_stop():
    if not _calib["recording"]:
        return {"ok": False, "message": "Not calibrating."}
    _calib["recording"] = False
    th = _calib["thread"]
    if th:
        th.join(timeout=1.0)
    before = car_ingest.load(_calib["car"])
    fit, rep = calibrate_car.save_calibrated(_calib["car"], list(_calib["frames"]))
    return {"ok": bool(fit), "report": rep, "before": before, "after": fit, "car": _calib["car"]}


def _live_key():
    """Key of the policy training right now, if any (for the Garage 'LIVE' flag)."""
    if not _any_trainer():
        return None
    try:
        d = json.load(open(LIVE))
        return "%s__%s" % (str(d.get("track", "")).replace("/", "__"), d.get("car", ""))
    except Exception:
        m = _proc_meta
        return "%s__%s" % (str(m.get("track", "")).replace("/", "__"), m.get("car", "")) if m else None


def list_policies():
    live = _live_key()
    items = []; total_steps = 0; total_secs = 0.0
    for fp in glob.glob(os.path.join(REG_DIR, "*.json")):
        try:
            r = json.load(open(fp))
        except Exception:
            continue
        key = r.get("key") or os.path.splitext(os.path.basename(fp))[0]
        ckpt = os.path.join(ML_DIR, r.get("ckpt", "sim_sac__%s.pt" % key))
        r["has_ckpt"] = os.path.exists(ckpt)
        r["size_mb"] = round(os.path.getsize(ckpt) / 1e6, 1) if r["has_ckpt"] else 0.0
        r["live"] = (key == live)
        r["protected"] = (key == CAR_KEY)
        items.append(r)
        total_steps += int(r.get("sim_steps", 0)); total_secs += float(r.get("train_seconds", 0))
    # backfill: per-(track,car) checkpoints trained before the registry existed. ONLY the canonical
    # 'sim_sac__<track>__<car>.pt' naming is recognised.
    seen = {it["key"] for it in items}
    cands = list(glob.glob(os.path.join(ML_DIR, "sim_sac__*.pt")))
    for pt in cands:
        base = os.path.basename(pt)
        key = base[len("sim_sac__"):-3]; car = key.rsplit("__", 1)[-1]
        track = key[:-len(car) - 2].replace("__", "/")
        if key in seen:
            continue
        items.append({"key": key, "car": car, "track": track, "sim_steps": 0, "reward": 0.0,
                      "best_lap": 0.0, "crash_rate": 0.0, "agents": 0, "calibrated": False,
                      "trained_at": os.path.getmtime(pt), "has_ckpt": True,
                      "size_mb": round(os.path.getsize(pt) / 1e6, 1), "live": (key == live),
                      "protected": (key == CAR_KEY), "legacy": True})
    items.sort(key=lambda x: (x["live"], x.get("trained_at", 0)), reverse=True)
    return {"policies": items, "count": len(items), "total_steps": total_steps,
            "total_hours": round(total_secs / 3600, 2), "live_key": live}


def delete_policy(key):
    if key == _live_key():
        return False, "That policy is training now — stop it first."
    if key == CAR_KEY:
        return False, "The car's deploy policy is protected."
    reg = os.path.join(REG_DIR, key + ".json")
    ckpt = None
    try:
        ckpt = os.path.join(ML_DIR, json.load(open(reg)).get("ckpt", "sim_sac__%s.pt" % key))
    except Exception:
        ckpt = os.path.join(ML_DIR, "sim_sac__%s.pt" % key)
    removed = 0
    for p in (reg, ckpt):
        try:
            if p and os.path.exists(p) and os.path.basename(p) != "sim_sac.pt":   # never the car
                os.remove(p); removed += 1
        except Exception:
            pass
    return removed > 0, "Deleted." if removed else "Nothing to delete."


# ── voxelisation: build the surface-classified voxel grid from the track's real .kn5 mesh ──
_vox = {"proc": None, "track": None, "started": 0.0}


def voxel_status():
    """Live voxel-build feed (read from ml/voxel_live.json) + whether a cached grid already exists."""
    out = {"running": False, "track": _vox["track"]}
    p = _vox["proc"]
    if p is not None and p.poll() is None:
        out["running"] = True
    try:
        with open(VOXEL_LIVE) as f:
            out.update(json.load(f))
    except Exception:
        pass
    return out


def has_voxels(track):
    key = (track or "").replace("/", "__")
    return os.path.exists(os.path.join(TRACKS_DIR, key + "_voxels.npz"))


def start_voxelize(track):
    with _lock:
        p = _vox["proc"]
        if p is not None and p.poll() is None:
            return False, "Already voxelising %s." % _vox["track"]
        if not track:
            return False, "No track given."
        os.makedirs(LOGDIR, exist_ok=True)
        log = open(os.path.join(LOGDIR, "voxel_out.log"), "w")
        base, _, lay = track.partition("/")
        args = [PY, os.path.join("ml", "voxelize.py"), base]
        if lay:
            args.append(lay)
        _vox["proc"] = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, PYTHONUNBUFFERED="1"))
        _vox["track"] = track; _vox["started"] = time.time()
        return True, "Voxelising %s from its mesh…" % track


def start_training(car, track, agents, steps):
    global _proc, _proc_meta
    with _lock:
        if _any_trainer():
            return False, "A stint is already on track. Box it first."
        # never allow two trainers: kill any stray train_sim_gpu.py (the flicker + checkpoint-clash bug)
        _kill_strays()
        os.makedirs(LOGDIR, exist_ok=True)
        out = open(os.path.join(LOGDIR, "train_out.log"), "w")
        err = open(os.path.join(LOGDIR, "train_err.log"), "w")
        args = [PY, os.path.join("ml", "train_sim_gpu.py"),
                "--cars", str(int(agents)), "--car", car, "--track", track]
        if int(steps) > 0:
            args += ["--steps", str(int(steps))]
        # new process group so we can send CTRL_BREAK for a graceful (checkpoint-saving) stop
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        _proc = subprocess.Popen(args, cwd=ROOT, stdout=out, stderr=err,
                                 creationflags=flags, env=env)
        _proc_meta = {"car": car, "track": track, "agents": int(agents),
                      "steps": int(steps), "pid": _proc.pid, "started": time.time()}
        return True, "Stint launched."


def stop_training():
    global _proc
    with _lock:
        if _trainer_alive():
            try:
                if os.name == "nt":
                    _proc.send_signal(signal.CTRL_BREAK_EVENT)   # -> KeyboardInterrupt -> saves ckpt
                else:
                    _proc.send_signal(signal.SIGINT)
            except Exception:
                _proc.terminate()
            return True, "Boxing the stint — saving checkpoint."
        if _feed_running():            # a stray we adopted (e.g. launched before a server restart)
            _kill_strays()
            return True, "Boxing the stint."
        return False, "Nothing on track."


def read_status():
    st = {"running": _any_trainer(), "meta": _proc_meta if _trainer_alive() else {}}
    try:
        with open(LIVE, "r") as f:
            live = json.load(f)
        # only surface the light, render-relevant fields (skip the huge cars/heat arrays)
        keep = ("running", "step", "warmup", "car", "track", "avgspeed", "topspeed", "reward",
                "crashes", "n_wall", "n_off", "alpha", "buf", "tps", "cars_total",
                "laps_total", "lap_avg", "lap_best", "lap_target", "history", "chist", "shist", "ts", "lead")
        st["live"] = {k: live[k] for k in keep if k in live}
    except Exception:
        st["live"] = None
    return st


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path):
        if not path or not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/" or u.path == "/index.html":
            return self._file(os.path.join(WEB, "index.html"))
        if u.path in ("/styles.css", "/app.js"):
            return self._file(os.path.join(WEB, u.path.lstrip("/")))
        if u.path == "/api/library":
            return self._send(200, {**scan.scan_library(force=("force" in q)),
                                    "agent_presets": AGENT_PRESETS, "length_presets": LENGTH_PRESETS})
        if u.path == "/api/status":
            return self._send(200, read_status())
        if u.path == "/api/gpu":
            return self._send(200, gpu_stats())
        if u.path == "/api/track":
            return self._send(200, track_geom())
        if u.path == "/api/voxels":
            return self._send(200, voxel_status())
        if u.path == "/api/mesh":
            key = (q.get("track", [""])[0] or "").replace("/", "__")
            key = "".join(ch for ch in key if ch.isalnum() or ch in "_-")   # no path traversal
            part = q.get("part", ["header"])[0]
            ext = "_mesh.bin" if part == "bin" else "_mesh.json"
            p = os.path.join(TRACKS_DIR, key + ext)
            if part == "header" and not os.path.exists(p):
                return self._send(200, {"ready": False})
            return self._file(p)
        if u.path == "/api/cars":
            return self._send(200, cars_live())
        if u.path == "/api/calibrate/status":
            return self._send(200, calib_status())
        if u.path == "/api/policies":
            return self._send(200, list_policies())
        if u.path == "/api/preview":
            kind = (q.get("kind", ["preview"])[0])
            cid = q.get("car", [None])[0]; tid = q.get("track", [None])[0]
            p = scan.car_image(cid, kind) if cid else (scan.track_image(tid, kind) if tid else None)
            return self._file(p)
        return self._send(404, {"error": "unknown route"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if u.path == "/api/train":
            ok, msg = start_training(body.get("car", ""), body.get("track", ""),
                                     body.get("agents", 8000), body.get("steps", 0))
            return self._send(200 if ok else 409, {"ok": ok, "message": msg})
        if u.path == "/api/stop":
            ok, msg = stop_training()
            return self._send(200, {"ok": ok, "message": msg})
        if u.path == "/api/voxelize":
            ok, msg = start_voxelize(body.get("track", ""))
            return self._send(200 if ok else 409, {"ok": ok, "message": msg})
        if u.path == "/api/calibrate/start":
            ok, msg = calib_start(body.get("car", ""))
            return self._send(200 if ok else 409, {"ok": ok, "message": msg})
        if u.path == "/api/calibrate/stop":
            return self._send(200, calib_stop())
        if u.path == "/api/policy/delete":
            ok, msg = delete_policy(body.get("key", ""))
            return self._send(200 if ok else 409, {"ok": ok, "message": msg})
        return self._send(404, {"error": "unknown route"})


def main():
    lib = scan.scan_library()
    print("PIT WALL  ·  AC %s" % ("FOUND" if lib["ac_ok"] else "NOT FOUND — set AC_ROOT"))
    print("  cars: %d   trainable track layouts: %d"
          % (len(lib["cars"]), sum(t["has_ai"] for t in lib["tracks"])))
    print("  open  ->  http://127.0.0.1:%d" % PORT)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
