"""
Per-car calibration — replace the spec-ESTIMATED physics with MEASURED physics by driving a few
laps. Reads the same CSP ground-truth feed (gt_live.json) the deploy uses, so it works for any car
including the 204 whose data is encrypted.

What you drive teaches it:
  * a couple of FULL-THROTTLE pulls from low speed   -> peak acceleration + how it falls with speed
  * a few HARD braking zones down low                -> braking deceleration (tyre limit)
  * some hard CORNERS                                -> peak lateral grip (mu)
  * one long straight to top speed (optional)        -> real top speed

The app drives this (Calibrate button); fit_frames() does the maths and writes ml/cars/<car>.json.
Standalone:  python ml/calibrate_car.py <car> --secs 90
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import car_ingest

CSP_GT = r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\apps\lua\nndrive\state.json"
G = 9.81


def read_gt():
    """One CSP frame -> dict, or None if the logger isn't publishing."""
    try:
        d = json.load(open(CSP_GT))
    except Exception:
        return None
    return {"frame": d.get("frame"), "speed": float(d.get("speedKmh", 0.0)) / 3.6,
            "gas": float(d.get("gas", 0.0)), "brake": float(d.get("brake", 0.0)),
            "steer": float(d.get("steer", 0.0)), "yaw": float(d.get("yaw", 0.0))}


def _robust(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


def fit_frames(frames, base):
    """frames: list of {t, speed(m/s), gas, brake, steer, yaw}. base: estimated CarParams (for L /
    fallbacks). Returns measured params + a quality report. Anything under-sampled keeps the estimate."""
    if len(frames) < 40:
        return None, {"error": "not enough data — drive a bit longer"}
    t = np.array([f["t"] for f in frames]); v = np.array([f["speed"] for f in frames])
    gas = np.array([f["gas"] for f in frames]); brk = np.array([f["brake"] for f in frames])
    steer = np.array([f["steer"] for f in frames]); yaw = np.array([f["yaw"] for f in frames])
    # smooth speed, then tangential acceleration dv/dt (robust to per-frame jitter)
    k = np.ones(5) / 5.0
    vs = np.convolve(v, k, mode="same")
    dt = np.gradient(t)
    good = (dt > 0.004) & (dt < 0.1)
    a = np.where(good, np.gradient(vs) / np.where(good, dt, 1.0), 0.0)   # m/s^2 along travel
    out = dict(base)
    rep = {}

    # ACCELERATION: full throttle, going straight, not braking. Peak = the low-speed accel.
    acc_m = (gas > 0.85) & (brk < 0.05) & (np.abs(steer) < 0.25) & (a > 0) & good & (v > 1)
    rep["n_accel"] = int(acc_m.sum())
    if acc_m.sum() >= 15:
        va, aa = v[acc_m], a[acc_m]
        lo = va < max(8.0, np.percentile(va, 40))                       # peak lives at low speed
        peak = aa[lo] if lo.sum() >= 8 else aa
        out["a_accel"] = round(float(np.clip(np.percentile(peak, 80), 3.0, 16.0)), 3)

    # BRAKING: hard brake, off throttle. Decel magnitude (tyre/brake limit).
    brk_m = (brk > 0.6) & (gas < 0.05) & (a < 0) & good
    rep["n_brake"] = int(brk_m.sum())
    if brk_m.sum() >= 12:
        out["a_brake"] = round(float(np.clip(_robust(-a[brk_m], 85), 6.0, 22.0)), 3)

    # LATERAL GRIP: centripetal accel a_lat = v * yaw_rate in corners; peak/g = mu.
    a_lat = np.abs(v * yaw)
    cor_m = (v > 5) & (np.abs(yaw) > 0.05) & good
    rep["n_corner"] = int(cor_m.sum())
    grip_lat = _robust(a_lat[cor_m], 95) / G if cor_m.sum() >= 15 else 0.0
    grip_brk = (out["a_brake"] / G) if brk_m.sum() >= 12 else 0.0
    grip = max(grip_lat, grip_brk)
    if grip > 0.5:
        out["grip"] = round(float(np.clip(grip, 0.9, 1.7)), 3)

    # TOP SPEED: only trust it if they actually ran near the estimate (a real straight).
    vobs = _robust(v, 99.5)
    rep["top_kmh_seen"] = round(vobs * 3.6, 1)
    if vobs > 0.85 * base.get("vmax_ms", 1e9):
        out["vmax_ms"] = round(float(vobs), 2)

    # STEER LOCK: small-angle yaw model r = v*delta/L, delta = steer*lock -> lock = r*L/(v*steer).
    L = float(base.get("L", 2.6))
    sl_m = (np.abs(steer) > 0.25) & (v > 8) & (v < 32) & (np.abs(yaw) > 0.05) & good
    rep["n_steer"] = int(sl_m.sum())
    if sl_m.sum() >= 15:
        lock = np.abs(yaw[sl_m]) * L / (v[sl_m] * np.abs(steer[sl_m]))
        out["steer_lock"] = round(float(np.clip(np.median(lock), 0.1, 0.6)), 3)

    # recompute the dependent forces so the sim picks them up
    out["IZ"] = round(out["mass"] * out["LF"] * out["LR"] * 1.1, 1)
    out["C_aero"] = round(out["a_accel"] * out["mass"] / max(out["vmax_ms"] ** 2, 1.0), 4)
    out["from_data"] = "calibrated"
    rep["ok"] = rep.get("n_accel", 0) >= 15 or rep.get("n_brake", 0) >= 12 or rep.get("n_corner", 0) >= 15
    return out, rep


def save_calibrated(car_id, frames):
    base = car_ingest.load(car_id)
    fit, rep = fit_frames(frames, base)
    if fit is None:
        return None, rep
    json.dump(fit, open(os.path.join(car_ingest.OUT_DIR, car_id + ".json"), "w"), indent=1)
    return fit, rep


def record(car_id, secs=90):
    """Standalone capture: poll the CSP feed while YOU drive, then fit + save."""
    if read_gt() is None:
        print("no CSP feed (gt_live.json). Start AC, load the car, enable the Car GT Logger app."); return
    print("recording %ds — drive: hard pulls, hard braking, hard corners, one top-speed run." % secs)
    frames = []; last = None; t0 = time.time()
    while time.time() - t0 < secs:
        f = read_gt()
        if f and f["frame"] != last:
            last = f["frame"]; f["t"] = time.time() - t0; frames.append(f)
            print("\r  %4.0fs  %5d frames  v %5.1f km/h" % (time.time() - t0, len(frames), f["speed"] * 3.6),
                  end="", flush=True)
        time.sleep(0.015)
    fit, rep = save_calibrated(car_id, frames)
    print("\n", rep)
    if fit:
        print("calibrated: a_accel %.1f  a_brake %.1f  grip %.2f  top %.0f km/h  lock %.2f"
              % (fit["a_accel"], fit["a_brake"], fit["grip"], fit["vmax_ms"] * 3.6, fit["steer_lock"]))


if __name__ == "__main__":
    car = sys.argv[1] if len(sys.argv) > 1 else "bmw_m3_e92"
    secs = int(sys.argv[sys.argv.index("--secs") + 1]) if "--secs" in sys.argv else 90
    record(car, secs)
