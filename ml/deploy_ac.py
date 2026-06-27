"""
Deploy the trained car policy INTO Assetto Corsa — the bridge from sim to real.

It reads AC's live telemetry, rebuilds the EXACT 56-feature observation the sim trained on (same
formulas, same normalisation), runs the trained QR-SAC policy, and drives the car through the vJoy
virtual wheel. A safety supervisor recovers if the car ends up off-track / stuck / facing backward.

  python ml/deploy_ac.py --dry         # NO output — just print the rebuilt obs + the policy's
                                       #   intended [steer, throttle/brake]. Use this to calibrate.
  python ml/deploy_ac.py               # DRIVE for real via the virtual Xbox GAMEPAD (default).
                                       #   AC's pad steering filter gently smooths the policy's
                                       #   micro-jitter, so it steers better than the raw vJoy wheel.
  python ml/deploy_ac.py --vjoy        # use the vJoy wheel instead (linear, no pad filtering)
  python ml/deploy_ac.py --invert-steer  # flip steering if the car turns the WRONG way on the pad

CALIBRATION (one-time, in --dry mode while sitting still then driving straight):
  * HEADING_OFFSET / HEADING_SIGN — so the car's forward matches the track frame (dir feature ~[1,0]
    when driving straight along the track).
  * STEER_OUT_SIGN — so steering goes the right way (already worked out in the other drivers).
  * STEER_LOCK_DEG — AC's steering lock, so the policy's [-1,1] maps to the real wheel range.
"""
import os, sys, time, math, argparse, ctypes, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "telemetry"))
import car_sim as K
from car_sim import Track, V_MAX, RAY_MAX, FAN, A_LAT_LIMIT, G, LF, LR, MU0
from sac import SAC

# ---------------- calibration constants (tune in --dry mode) ----------------
HEADING_OFFSET = 0.0        # rad added to AC heading to align with the track xz frame
HEADING_SIGN   = 1.0        # flip if the car's forward comes out mirrored
STEER_OUT_SIGN = +1.0       # policy steer -> wheel axis. FLIPPED to +1 (it was turning the wrong way)
CSP_STEER_SIGN = +1.0       # CSP ac.overrideCarControls steer sign (flipped back to +1 — was turning
                            # the wrong way as -1; CSP cc.steer matches the policy's +steer=left here)
STEER_GAIN     = 0.50       # fraction of the wheel range the policy uses. CUT from 0.85 -> 0.50: the
                            # F1's real steering is far more responsive than the sim's, so the policy's
                            # commanded angle was oversteering (rear snapping out). Lower = calmer/less
                            # bite; raise toward 0.65 if it now UNDERsteers (won't make the corner).
STEER_LOCK_DEG = 24.75      # full policy steer (=1.0) -> front-wheel deg. MEASURED from AC (0.432 rad).
MAX_STEER_RATE = 13.0       # max steer change per second (raised to cut lag -> the wheel reaches the
                            # policy's command faster, so its anticipation isn't delayed into reaction)
STEER_SMOOTH   = 0.55       # faster tracking of the policy command = LESS lag. The policy's own
                            # decisions are already smooth (~0.15), so heavy smoothing just added lag.
CTRL_HZ        = 60.0       # wheel-update rate (smooth)
POLICY_HZ      = 14.0       # decision rate. Raised from 10 -> fresher decisions reach the wheel sooner
                            # (the obs is instantaneous, so a higher rate is just more responsive, not
                            # jittery now that the yaw/rays are clean). Cuts the 'reacts late' feel.
THR_SMOOTH     = 0.15       # smoothing for throttle/brake (gentler launch, no on/off snapping)
# ---- anti-overspeed / anti-spin supervisor (closes the sim->real grip gap) ----
SPEED_MARGIN   = 0.90       # cap corner speed to this fraction of the grip limit ahead.
                            # LOWER (e.g. 0.85) = safer/slower into corners, fewer spins.
SPIN_YAW       = 2.6        # rad/s yaw rate above this = rear stepping out -> ease power + lock
SPIN_SLIP      = 0.28       # rad rear-tyre slip above this = sliding -> ease power + lock
STEER_SPEED_FALLOFF = 0.0   # DISABLED — was cutting steering at speed and causing understeer. The
                            # policy already learned its own speed-sensitive steering in the sim.
WALL_NEAR       = 0.7       # m. only react to IMMINENT contact (was 1.3 -> it shoved the car off
                            # apexes, causing the run-wide). On a tight car track the line hugs walls.
WALL_AVOID_GAIN = 0.25      # gentle last-resort nudge only (was 0.55 -> fought the racing line)
CKPT = os.path.join(HERE, "deploy_policy.pt")
CALIB_JSON = os.path.join(HERE, "..", "data", "heading_calib.json")
DRIVE_LOG = os.path.join(HERE, "..", "data", "deploy_log.csv")
# CSP GROUND TRUTH: the 'Car GT Logger' CSP app (running inside AC) publishes AC's REAL track
# raycasts + collision flag here. When present & fresh, we use these instead of our reconstructed-
# track raycasting (which guessed the wrong stacked level -> phantom walls). The CSP app is part of
# AC (not a separate python script), so this stays a SINGLE command: python ml/deploy_ac.py.
# LIVE CAR STATE comes from the NN Drive CSP app's state.json (published every frame inside AC):
# the REAL world position (for localisation) + heading/yaw. Self-sufficient — no separate logger app,
# and a freshness check (below) means a stale file fails LOUDLY instead of driving on dead coords.
CSP_GT = r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\apps\lua\nndrive\state.json"
CSP_RAY_FLIP = False        # set True if CSP's left/right rays come out mirrored vs the sim


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# manual-drive keys (GetAsyncKeyState — works regardless of which window is focused)
VK = {"W": 0x57, "A": 0x41, "S": 0x53, "D": 0x44, "E": 0x45, "C": 0x43, "M": 0x4D,
      "UP": 0x26, "LEFT": 0x25, "DOWN": 0x28, "RIGHT": 0x27}


def key_down(vk):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def load_heading_calib():
    """Apply a saved heading calibration (HEADING_SIGN/OFFSET) if present."""
    global HEADING_SIGN, HEADING_OFFSET
    try:
        import json
        d = json.load(open(CALIB_JSON))
        HEADING_SIGN = float(d["sign"]); HEADING_OFFSET = float(d["offset"])
        print("loaded heading calibration: sign=%+.0f offset=%+.4f" % (HEADING_SIGN, HEADING_OFFSET))
    except Exception:
        pass


def ensure_gear(reader, pad, target=2, tries=10):
    """Tap Gear Up (vJoy Button 1) until the car is in FIRST gear (AC gear index: 0=R,1=N,2=1st).
    Needs 'Gear Up' bound to Button 1 in AC. (Or just enable Automatic Gearbox and it's a no-op.)"""
    for _ in range(tries):
        fr = reader.frame()
        if not fr or fr.get("status") != 2:
            time.sleep(0.1); continue
        if fr.get("gear", 1) >= target:
            return True
        pad.press_button(); pad.update(); time.sleep(0.12)
        pad.release_button(); pad.update(); time.sleep(0.25)
    fr = reader.frame()
    g = fr.get("gear", 1) if fr else 1
    if g < target:
        print("\n  car still in NEUTRAL (gear=%d). Either:" % g)
        print("    - enable Automatic Gearbox in AC (Settings -> Assists), OR")
        print("    - bind 'Gear Up' to vJoy Button 1:  python ml/bind_helper.py gearup")
    return g >= target


def calibrate(reader, pad):
    """Self-calibration: the car DRIVES ITSELF forward (gentle throttle, zero steer) via the virtual
    wheel, while we measure AC's heading vs the actual travel direction. No manual driving needed."""
    import json
    print("\nSELF-CALIBRATION: the car will roll forward gently for a few seconds. Make sure it's on")
    print("track facing roughly forward. Ctrl+C if it heads for a wall.\n")
    print("  shifting into first gear...")
    ensure_gear(reader, pad)
    cal_t = []; cal_h = []; t0 = time.time()
    try:
        while time.time() - t0 < 10.0 and len(cal_t) < 250:
            fr = reader.frame()
            if not fr or fr.get("status") != 2:
                pad.left_joystick_float(0.0); pad.right_trigger_float(0.0); pad.update()
                time.sleep(0.1); continue
            pad.left_joystick_float(x_value_float=0.0)        # straight
            pad.right_trigger_float(value_float=0.35)         # gentle throttle
            pad.left_trigger_float(value_float=0.0); pad.update()
            v = fr["speed"] / 3.6
            if v > 3.0:                                       # moving -> sample
                cal_t.append(math.atan2(fr["vel"][2], fr["vel"][0])); cal_h.append(fr["heading"])
            print("\r  collecting... speed %.1f km/h  samples %d   " % (fr["speed"], len(cal_t)),
                  end="", flush=True)
            time.sleep(1.0 / CTRL_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        pad.left_joystick_float(0.0); pad.right_trigger_float(0.0); pad.left_trigger_float(0.0); pad.update()
    if len(cal_t) < 20:
        print("\n\nnot enough motion captured — make sure AC is unpaused & on track, then retry.")
        return
    import numpy as np
    tp = np.array(cal_t); th = np.array(cal_h); best = None
    for sign in (1.0, -1.0):
        diff = tp - sign * th
        off = math.atan2(np.sin(diff).mean(), np.cos(diff).mean())
        resid = float(np.std(np.angle(np.exp(1j * (diff - off)))))
        if best is None or resid < best[2]:
            best = (sign, off, resid)
    sign, off, resid = best
    json.dump({"sign": sign, "offset": off, "spread": resid}, open(CALIB_JSON, "w"), indent=2)
    print("\n\n=== CALIBRATED (%d samples) ===" % len(cal_t))
    print("  HEADING_SIGN = %+.0f   HEADING_OFFSET = %+.4f   (fit spread %.3f rad)" % (sign, off, resid))
    print("  saved to data/heading_calib.json — deploy_ac.py will use it automatically.")
    print("  %s" % ("GOOD — now run:  python ml/deploy_ac.py --dry   to confirm dir ~+1, then drive."
                    if resid < 0.3 else "noisy fit — try again on a straighter part of the track."))


class CSPReader:
    """Reads AC's REAL raycasts + collision from the CSP app's gt_live.json (published every frame).
    Returns None when the file is missing/stale so the deploy falls back to our own raycasting."""
    def __init__(self):
        self.last_frame = None; self.last_change = time.time(); self.available = os.path.exists(CSP_GT)
        if self.available:
            print("CSP ground truth: ON  (using AC's real walls + collision from the Car GT Logger app)")
        else:
            print("CSP ground truth: off (gt_live.json not found) — using reconstructed-track rays.\n"
                  "  enable the 'Car GT Logger' app in AC for accurate walls.")

    def read(self):
        try:
            d = json.load(open(CSP_GT))
        except Exception:
            return None
        fr = d.get("frame")
        if fr != self.last_frame:
            self.last_frame = fr; self.last_change = time.time()
        if time.time() - self.last_change > 0.5:          # stale (app closed / AC paused / frozen file)
            return None
        rays = d.get("rays")                               # optional — deploy uses reconstructed rays
        ray = None
        if rays and len(rays) >= 36:
            ray = np.array(rays[:36], np.float32)
            if CSP_RAY_FLIP:                               # mirror left<->right if convention differs
                ray = ray[::-1].copy()
        return {"rays": ray, "collision": float(d.get("collision", 0.0)),
                "speed": float(d.get("speedKmh", 0.0)),
                "look": d.get("look"), "pos": d.get("pos"), "spline": d.get("spline", 0.0),
                "yaw": d.get("yaw")}      # AC's REAL yaw rate (rad/s, body frame) — clean


class ObsBuilder:
    """Rebuilds the sim's 56-feature observation from AC's live state, feature-for-feature.
    `car` (an ingested CarParams dict) makes the per-car features match training: the sim normalises
    vx by the car's top speed, derives the slip angles from the car's steer-lock + wheelbase, and
    scales grip by the car's lateral limit. With car=None it falls back to the car's constants."""
    def __init__(self, track, car=None):
        self.t = track
        # per-car normalisers (mirror GPUCarSim: vx/self.vmax, delta=action*self.lock, LF/LR, grip cap)
        self.vmax    = float(car["vmax_ms"])           if car else V_MAX
        self.lock_deg = math.degrees(float(car["steer_lock"])) if car else STEER_LOCK_DEG
        self.LF      = float(car["LF"])                if car else LF
        self.LR      = float(car["LR"])                if car else LR
        self.a_lat   = float(car["grip"]) * G          if car else A_LAT_LIMIT
        # supervisor cap scale: Track.corner_vmax is capped at the car V_MAX=32, but a real car's
        # straight/corner speeds are higher — scale the overspeed cap so it doesn't brake an F1 to
        # car speed. (The OBS keeps the raw car-scaled cvmax so it matches what the policy trained on.)
        self.cv_scale = (self.vmax / V_MAX) if car else 1.0
        ar = np.radians(np.array(FAN, float)); self.rc = np.cos(ar); self.rs = np.sin(ar)
        fan = np.array(FAN)
        self.left_idx = np.where((fan >= 25) & (fan <= 95))[0]    # +30..+90 = LEFT side rays
        self.right_idx = np.where((fan <= -25) & (fan >= -95))[0]  # -30..-90 = RIGHT side rays
        self.prog = None                          # set by a global search on the first frame
        self.prev_head = None; self.prev_t = None; self.r = 0.0
        self.n_ahead = self.t.curve_ahead.shape[1]
        self.yaw_sign = None; self._yacc = 0.0; self._yn = 0   # auto-match CSP yaw sign to our frame
        self.y_offset = None                                   # AC car-y vs our centreline-y (for floors)

    def _global_localize(self, pos, car_y=None):
        """Nearest centreline point over the WHOLE track. HEIGHT-AWARE: on the multi-level hill, two
        levels overlap in x/z, so we also match the car's real HEIGHT to pick the RIGHT floor (else it
        locks onto the wrong stacked level -> wrong direction / phantom walls -> drives off the hill)."""
        d2 = (self.t.center[:, 0] - pos[0, 0]) ** 2 + (self.t.center[:, 1] - pos[0, 1]) ** 2
        if car_y is not None and self.y_offset is not None:
            dy = self.t.center_y - (car_y - self.y_offset)
            d2 = d2 + 8.0 * dy * dy                            # strong height term -> correct floor
        return np.array([int(d2.argmin())], np.int64)

    def build(self, fr, last_steer, csp_rays=None, csp_look=None, csp_yaw=None):
        pos = np.array([[fr["pos"][0], fr["pos"][2]]], float)          # xz
        # car forward direction in the track xz frame. CSP's real 'look' vector is in AC world xz,
        # the SAME frame as our track tangents -> using it removes ALL heading-calibration error
        # (that error was driving dir_cos to -1 and making the policy steer the wrong way).
        if csp_look is not None and abs(csp_look[0]) + abs(csp_look[2]) > 1e-4:
            psi = math.atan2(csp_look[2], csp_look[0])
        else:
            psi = HEADING_SIGN * fr["heading"] + HEADING_OFFSET
        head = np.array([[math.cos(psi), math.sin(psi)]], float)
        # yaw rate from heading derivative — HEAVILY SMOOTHED (raw differentiation at high frame
        # rate is very noisy, and that noise was the main cause of the jittery steering).
        now = time.time()
        if self.prev_head is not None and self.prev_t is not None:
            dt = max(now - self.prev_t, 1e-3)
            raw_r = wrap(psi - self.prev_head) / dt
            self.r = 0.6 * self.r + 0.4 * float(np.clip(raw_r, -6, 6))
        self.prev_head = psi; self.prev_t = now
        recon_r = float(np.clip(self.r, -6, 6))             # heading-derivative yaw (noisy, sign-ref)
        # Prefer AC's REAL yaw rate from CSP (clean, no differentiation noise). Auto-learn its sign by
        # matching it to recon_r over the first turns, then use the clean value -> fixes the false-spin
        # jitter that made the policy weave/oscillate.
        if csp_yaw is not None:
            if self.yaw_sign is None:
                if abs(recon_r) > 0.3:
                    self._yacc += recon_r * float(csp_yaw); self._yn += 1
                if self._yn >= 15:
                    self.yaw_sign = 1.0 if self._yacc >= 0 else -1.0
                r = recon_r                                 # until the sign is learned
            else:
                r = float(np.clip(self.yaw_sign * float(csp_yaw), -6, 6))
        else:
            r = recon_r
        # body-frame velocity (rotate world horizontal velocity into the car frame)
        vw = np.array([fr["vel"][0], fr["vel"][2]])
        fwd = np.array([math.cos(psi), math.sin(psi)]); left = np.array([-fwd[1], fwd[0]])
        vx = float(vw @ fwd); vy = float(vw @ left)
        # G-forces (AC accG in g): [lateral, vertical, longitudinal] -> m/s^2
        latG = fr["accG"][0] * G; lonG = fr["accG"][2] * G
        grip = float(min(2.0, math.hypot(latG, lonG) / max(self.a_lat, 1e-3)))
        slip_b = math.atan2(vy, max(vx, 0.1))
        # tyre slip angles — SAME formula the sim uses (kinematics + commanded steer angle)
        delta = last_steer * math.radians(self.lock_deg)
        vxs = max(vx, 2.0)
        slip_f = math.atan2(vy + self.LF * r, vxs) - delta
        slip_r = math.atan2(vy - self.LR * r, vxs)
        # localise on the 3D centreline: global search to init / re-acquire, else floor-safe local.
        # HEIGHT-AWARE re-acquire so it can't lock onto the wrong stacked level at the hill.
        car_y = float(fr["pos"][1])
        if self.prog is None:
            self.prog = self._global_localize(pos, car_y)
        i, off = self.t.localize(pos, self.prog)
        if float(off[0]) > 9.0:                       # lost (too far) -> re-acquire globally
            i = self._global_localize(pos, car_y); _, off = self.t.localize(pos, i)
        self.prog = i
        idx = int(i[0])
        if self.y_offset is None and float(off[0]) < 3.0:     # calibrate car-y vs centreline-y once
            self.y_offset = car_y - float(self.t.center_y[idx])
        # rays (height-aware). CRITICAL: use the CENTRELINE height like the sim did in training
        # (car_sim.step uses cy = center_y[i]), NOT AC's raw car_y. This track STACKS over itself,
        # so the height selects which level's walls the rays see. AC's car_y is in a different origin
        # / sits above the surface -> the old code saw PHANTOM walls from other levels (false scrapes
        # on flat straights + under ramps), and fed those bad rays to the policy -> it really scraped.
        if csp_rays is not None:
            rays = csp_rays                            # AC's REAL walls (ground truth from CSP)
        else:
            ray_y = float(self.t.center_y[idx])        # fallback: our reconstructed-track raycast
            rays = self.t.raycast(pos, head, self.rc, self.rs, np.array([ray_y]))[0]
        # ----- assemble EXACTLY like car_sim._obs -----
        dyn = np.array([vx / self.vmax, vy / 10.0, r / 3.0, slip_b / 0.6,
                        lonG / (1.5 * G), latG / (1.5 * G), grip,
                        np.clip(1.0 - grip, 0, 1), slip_f / 0.4, slip_r / 0.4])
        ta = math.atan2(self.t.tangent[idx, 1], self.t.tangent[idx, 0])
        err = ta - psi
        direction = np.array([math.cos(err), math.sin(err)])
        ahead = self.t.curve_ahead[idx] / (math.pi / 2)
        cvmax = self.t.corner_vmax[idx]                     # car-scaled (matches the trained obs)
        excess = np.clip(vx / cvmax - 1.0, -1.5, 1.5)       # OBS feature — keep raw car-scaled
        # REAL corner-speed limit for the supervisor: lift the capped straights to the car's true top
        # speed, scale genuine curvature-limited corners only by the mechanical-grip ratio (≈1 for an
        # F1 vs the car). Without this the car-V_MAX=32 cap would brake an F1 to ~115 km/h on straights.
        grip_scale = math.sqrt(self.a_lat / A_LAT_LIMIT)
        cv_real = np.where(cvmax >= V_MAX * 0.97, self.vmax, cvmax * grip_scale)
        obs = np.concatenate([np.clip(dyn, -3, 3), direction, np.clip(ahead, -3, 3),
                              excess, np.clip(rays / RAY_MAX, 0, 1.2)]).astype(np.float32)
        info = {"vx": vx, "vy": vy, "r": r, "grip": grip, "slip_b": math.degrees(slip_b),
                "off": float(off[0]), "prog": idx, "front": float(rays[len(FAN) // 2]),
                "dir_cos": float(direction[0]), "dir_sin": float(direction[1]),
                "vx_ms": vx, "slip_r": slip_r, "cvmax": float(np.min(cv_real)),
                "left_clear": float(rays[self.left_idx].min()),
                "right_clear": float(rays[self.right_idx].min()),
                "curve": float(self.t.curve_ahead[idx, 2]),    # signed bend ~24m ahead (anticipation)
                "prog_raw": idx}
        return obs, info


class DriveLogger:
    """Records every decision to a CSV and detects/classifies wall-contact EVENTS in real time, so we
    can see WHY it crashes (overspeed vs running-wide vs sliding vs wrong line) and WHERE on the track."""
    SCRAPE = 1.2      # m. nearest wall closer than this = a scrape (edge-triggered = one event)
    CRASH = 0.6       # m. this close, or off-track, = a crash

    def __init__(self, n_prog):
        import csv
        from collections import deque
        self.f = open(DRIVE_LOG, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "prog", "speed_kmh", "vx", "cvmax", "over", "off", "left", "right",
                         "front", "nearest", "grip", "yaw_r", "slip_r", "dir_cos", "dir_sin",
                         "curve", "pol_steer", "out_steer", "pol_longi", "recovering", "event"])
        self.prev_pol = 0.0; self.prev_prog = None
        self.steer_jit = []; self.prog_jumps = 0
        self.t0 = time.time(); self.n_prog = max(n_prog, 1)
        self.near_prev = 9.9; self.off_prev = 0.0
        self.scrapes = 0; self.crashes = 0
        self.by_cause = {}                 # cause -> count
        self.by_sector = {}                # sector (0..19) -> [scrapes, crashes]
        self.over_at_event = []            # overspeed margin at each event (m/s)
        self.grip_at_event = []
        self.spd_hist = deque(maxlen=12)   # ~1.2s of speed (km/h) for impact detection
        self.stuck_t0 = None; self.last_crash_t = -99.0

    def _classify(self, info, over):
        side = "LEFT" if info["left_clear"] < info["right_clear"] else "RIGHT"
        if over > 1.5:
            return "OVERSPEED", "too fast for the corner (%.0f over the %.0f m/s limit), %s wall" % (
                over, info["cvmax"], side)
        if info["off"] > 3.5:
            return "RUNNING-WIDE", "understeer / off the line (%.1fm off, %s wall)" % (info["off"], side)
        if info["grip"] > 1.0:
            return "SLIDING", "tyres past the grip limit (grip %.2f), %s wall" % (info["grip"], side)
        return "TIGHT-LINE", "clipped a %s wall on a tight bit (off %.1fm, grip %.2f)" % (
            side, info["off"], info["grip"])

    def log(self, info, pol_steer, pol_longi, recovering, speed, collision=0.0, out_steer=0.0):
        t = time.time() - self.t0
        nearest = min(info["front"], info["left_clear"], info["right_clear"])
        over = info["vx_ms"] - info["cvmax"]
        sector = int(info["prog"] * 20 / self.n_prog) % 20
        self.spd_hist.append(speed); recent_max = max(self.spd_hist)
        # steering smoothness diagnostics: policy decision jitter + localisation jumps (= ray jitter)
        self.steer_jit.append(abs(pol_steer - self.prev_pol)); self.prev_pol = pol_steer
        if self.prev_prog is not None:
            jump = abs(((info["prog"] - self.prev_prog + self.n_prog // 2) % self.n_prog) - self.n_prog // 2)
            if jump > 6:
                self.prog_jumps += 1
        self.prev_prog = info["prog"]
        event = ""
        # ----- crash triggers. CSP collision flag (AC's OWN contact) is ground truth -> trust it first.
        ray_crash = collision > 0 or nearest < self.CRASH or info["off"] > 6.0
        # IMPACT: was moving, then suddenly stopped -> hit something the rays may have MISSED
        impact = recent_max > 10.0 and speed < 4.0 and (recent_max - speed) > 7.0
        # STUCK: commanding throttle but not moving for >1.5s -> jammed against a wall
        if speed < 3.0 and pol_longi > 0.15:
            self.stuck_t0 = self.stuck_t0 if self.stuck_t0 is not None else t
        elif speed > 6.0:
            self.stuck_t0 = None
        stuck = self.stuck_t0 is not None and (t - self.stuck_t0) > 1.5
        is_crash = (ray_crash or impact or stuck) and (t - self.last_crash_t) > 2.0
        is_scrape = (not is_crash) and (nearest < self.SCRAPE) and (self.near_prev >= self.SCRAPE)
        if is_crash or is_scrape:
            if impact and not ray_crash:
                cause = "IMPACT"; detail = ("hit a wall the RAYS MISSED (front ray %.1fm, nearest %.1fm) "
                    "-> localisation/height still off here" % (info["front"], nearest))
            elif stuck and not ray_crash:
                cause = "STUCK"; detail = ("jammed %.1fs, throttle on but not moving (rays: front %.1f "
                    "left %.1f right %.1f)" % (t - (self.stuck_t0 or t), info["front"],
                                               info["left_clear"], info["right_clear"]))
            else:
                cause, detail = self._classify(info, over)
            kind = "CRASH" if is_crash else "scrape"
            event = "%s:%s" % (kind, cause)
            if is_crash:
                self.crashes += 1; self.last_crash_t = t; self.stuck_t0 = None
            else:
                self.scrapes += 1
            self.by_cause[cause] = self.by_cause.get(cause, 0) + 1
            s = self.by_sector.setdefault(sector, [0, 0]); s[1 if is_crash else 0] += 1
            self.over_at_event.append(over); self.grip_at_event.append(info["grip"])
            print("\n  %-6s #%d  sector %2d/20  v%.0f km/h  %s : %s%s"
                  % (kind, self.crashes if is_crash else self.scrapes, sector, speed, cause, detail,
                     "  [RECOVERING]" if recovering else ""), flush=True)
        self.w.writerow([round(t, 2), info["prog"], round(speed, 1), round(info["vx_ms"], 1),
                         round(info["cvmax"], 1), round(over, 1), round(info["off"], 1),
                         round(info["left_clear"], 1), round(info["right_clear"], 1),
                         round(info["front"], 1), round(nearest, 1), round(info["grip"], 2),
                         round(info["r"], 2), round(info["slip_r"], 2), round(info["dir_cos"], 2),
                         round(info.get("dir_sin", 0), 2), round(info.get("curve", 0), 3),
                         round(pol_steer, 2), round(out_steer, 2), round(pol_longi, 2),
                         int(recovering), event])
        self.near_prev = nearest; self.off_prev = info["off"]

    def summary(self):
        try:
            self.f.close()
        except Exception:
            pass
        t = time.time() - self.t0
        print("\n" + "=" * 64)
        print("DRIVE SUMMARY  (%.0fs)   crashes %d   scrapes %d" % (t, self.crashes, self.scrapes))
        if self.steer_jit:
            import numpy as _np
            jit = float(_np.mean(self.steer_jit))
            print("  STEERING: policy-decision jitter %.3f/tick  (sim is ~0.13; >0.25 = jittery here)" % jit)
            print("            localisation jumps %d  (track index hopping -> ray/anticipation jitter)"
                  % self.prog_jumps)
            if jit > 0.25 or self.prog_jumps > 5:
                print("            -> obs is noisier in AC than the sim. likely culprit: noisy yaw/slip"
                      " or localisation hopping. (full per-tick steering log in data/deploy_log.csv)")
        if self.by_cause:
            print("  by cause:  " + "   ".join("%s=%d" % (c, n) for c, n in
                  sorted(self.by_cause.items(), key=lambda x: -x[1])))
        if self.by_sector:
            worst = sorted(self.by_sector.items(), key=lambda x: -(x[1][0] + x[1][1]))[:4]
            print("  worst spots (sector/20): " + "   ".join(
                "#%d(%dc/%ds)" % (sec, v[1], v[0]) for sec, v in worst))
        if self.over_at_event:
            import numpy as _np
            print("  at contact: avg overspeed %+.1f m/s   avg grip %.2f"
                  % (float(_np.mean(self.over_at_event)), float(_np.mean(self.grip_at_event))))
            ov = sum(1 for o in self.over_at_event if o > 1.5)
            wide = self.by_cause.get("RUNNING-WIDE", 0)
            if ov > len(self.over_at_event) * 0.4:
                print("  >> DIAGNOSIS: mostly OVERSPEED - still too hot. Lower SPEED_MARGIN (0.90->0.82).")
            elif wide > (self.crashes + self.scrapes) * 0.4:
                print("  >> DIAGNOSIS: mostly RUNNING-WIDE - understeer/line; likely steering or heading calib.")
            else:
                print("  >> DIAGNOSIS: mixed - send me data/deploy_log.csv and I will pinpoint it.")
        print("  full per-decision log -> data/deploy_log.csv")
        print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="no wheel output; print obs + action")
    ap.add_argument("--calibrate", action="store_true", help="self-calibrate heading (car drives itself)")
    ap.add_argument("--vjoy", action="store_true", help="use the vJoy wheel instead of the gamepad")
    ap.add_argument("--csp", action="store_true", help="drive via CSP ac.overrideCarControls (NN Drive app) — no vJoy/gamepad")
    ap.add_argument("--invert-steer", action="store_true", help="flip steering direction (turns wrong way)")
    ap.add_argument("--car", default=None, help="ingested car id, e.g. ks_ferrari_sf70h (default: the car)")
    ap.add_argument("--track", default=None, help="ingested track key, e.g. imola (default: the car self-map)")
    ap.add_argument("--ckpt", default=None, help="policy checkpoint (default: auto sim_sac__<track>__<car>.pt, else deploy_policy.pt)")
    args = ap.parse_args()

    global STEER_OUT_SIGN
    if args.invert_steer:
        STEER_OUT_SIGN = -STEER_OUT_SIGN
        print("steering inverted -> STEER_OUT_SIGN = %+.0f" % STEER_OUT_SIGN)

    load_heading_calib()                                 # apply saved calibration if present

    try:
        from ac_memory import ACReader
        reader = ACReader()
    except Exception as e:
        print("AC not available:", e); print("Start AC, load on the track, then re-run."); return

    # --calibrate needs only the wheel + AC (no policy/track)
    if args.calibrate:
        from pad_output import open_pad
        pad, _ = open_pad(use_vjoy=args.vjoy)
        calibrate(reader, pad)
        return

    print("loading track + policy...")
    # ANY ingested AC car/track: --track loads the fast_lane geometry, --car its physics, and the
    # matching sim_sac__<track>__<car>.pt policy. With neither, it's the car self-map + deploy_policy.pt.
    car_params = None; geom = None
    if args.track:
        from track_ingest import load_geom
        key = args.track.replace("/", "__")
        geom = load_geom(key)
        print("track: %s — %d-pt centreline, %.0f m" % (args.track, len(geom["center"]), float(geom["length"])))
    if args.car:
        from car_ingest import load as load_car
        car_params = load_car(args.car)
        print("car: %s — %.0f kg, %d bhp, top %.0f km/h, grip %.2f, %s, steer-lock %.1f deg"
              % (car_params["name"], car_params["mass"], car_params["bhp"], car_params["vmax_ms"] * 3.6,
                 car_params["grip"], car_params["drive"], math.degrees(car_params["steer_lock"])))
    track = Track(geom=geom)
    ob = ObsBuilder(track, car=car_params)
    # pick the checkpoint: explicit --ckpt, else per-(track,car) policy, else the car deploy policy
    if args.ckpt:
        ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(HERE, args.ckpt)
    elif args.track and args.car:
        ckpt = os.path.join(HERE, "sim_sac__%s__%s.pt" % (args.track.replace("/", "__"), args.car))
    else:
        ckpt = CKPT
    obs_dim = 10 + 2 + 2 * track.curve_ahead.shape[1] + len(FAN)   # dyn10 + dir2 + ahead + excess + rays
    agent = SAC(obs_dim, 2, hid=512, n_quantiles=32, device="cpu")
    if not os.path.exists(ckpt):
        print("no policy checkpoint at", ckpt); return
    agent.load(ckpt)
    print("policy loaded (deterministic drive): %s  obs_dim=%d" % (os.path.basename(ckpt), obs_dim))

    pad = None; csp_ctrl = None
    if args.csp:
        from ac_csp_control import CspController
        csp_ctrl = CspController()              # writes the NN Drive app's cmd.txt; CSP applies it in-engine
        # The NN Drive Lua app does the final steering smoothing (a critically-damped SmoothDamp at AC's
        # render framerate, no choppiness). Python adds a MODERATE rate-limit + light EMA so the policy's
        # fast 14 Hz jumps don't slam the wheel into oversteer — controlled, but not the heavy lag that
        # made it feel laggy. (Earlier I removed this entirely -> it oversteered hard.)
        global STEER_SMOOTH, MAX_STEER_RATE
        STEER_SMOOTH = 0.6                        # light EMA — calms 14 Hz jumps without much lag
        MAX_STEER_RATE = 11.0                     # cap how fast the wheel can swing (anti-oversteer)
        print("output: CSP direct control — open the 'NN Drive' app window in AC (no vJoy/gamepad).")
        print("  steering: gain %.2f + Lua SmoothDamp — controlled & smooth; tune STEER_GAIN / smoothTime." % STEER_GAIN)
    elif not args.dry:
        from pad_output import open_pad
        pad, _ = open_pad(use_vjoy=args.vjoy)   # default: virtual Xbox GAMEPAD. AC's pad steering
                                                # filter smooths the policy's micro-jitter, so it
                                                # steers better than the raw wheel. --vjoy = wheel.

    if args.dry:
        print("\nDRY RUN — DRIVE STRAIGHT along the track at moderate speed for ~5s, then Ctrl+C.")
        print("Watch TRAVEL-dir and HEADING-dir: both should read ~+1.00 when straight.\n")
    else:
        if pad is not None:
            print("\nshifting into first gear...")
            ensure_gear(reader, pad)
        print("DRIVING (AI policy). Press M to toggle MANUAL keyboard drive. Ctrl+C to stop.\n")
    last_steer = 0.0; out_longi = 0.0; period = 1.0 / CTRL_HZ
    pol_steer = 0.0; pol_longi = 0.0; last_decide = 0.0; policy_dt = 1.0 / POLICY_HZ
    info = {"off": 0.0, "front": 9.0, "dir_cos": 1.0, "dir_sin": 0.0, "prog": 0, "grip": 0.0}
    recovering = False; stuck_since = None
    mode = "ai"; man_steer = 0.0; m_was = False; e_was = False     # manual-drive toggle state
    cal_pos = []; cal_head = []          # travel angle + AC heading samples (for auto-calibration)
    dlog = DriveLogger(track.K) if not args.dry else None   # crash/scrape diagnostics
    csp = CSPReader()                    # AC's REAL walls + collision (from the CSP app), if running
    csp_coll = 0.0
    try:
        while True:
            fr = reader.frame()
            if not fr or fr.get("status") != 2:           # not live on track
                time.sleep(0.05); continue
            if fr.get("pos_stale"):                       # no live position feed -> don't drive blind
                if csp_ctrl is not None:
                    csp_ctrl.set(steer=0.0, gas=0.0, brake=0.0)   # hands off until the feed is live
                time.sleep(0.1); continue
            now = time.time()
            # ---- press M to grab/release MANUAL keyboard control (only when actually driving) ----
            if not args.dry:
                m_now = key_down(VK["M"])
                if m_now and not m_was:
                    mode = "manual" if mode == "ai" else "ai"
                    if mode == "ai":
                        stuck_since = None                 # don't trip recovery on the handover
                    print("\n>>> %s" % (
                        "MANUAL drive — W/UP gas, S/DOWN brake, A/D steer, E gear-up, C reverse. M = AI."
                        if mode == "manual" else "AI policy driving. Press M for manual."), flush=True)
                m_was = m_now
            if mode == "manual":
                # smooth keyboard steering (ramp toward held side, auto-center on release)
                if key_down(VK["A"]) or key_down(VK["LEFT"]):
                    man_steer = max(-1.0, man_steer - 3.5 * period)
                elif key_down(VK["D"]) or key_down(VK["RIGHT"]):
                    man_steer = min(1.0, man_steer + 3.5 * period)
                else:
                    man_steer = (max(0.0, man_steer - 6.0 * period) if man_steer > 0
                                 else min(0.0, man_steer + 6.0 * period))
                last_steer = man_steer
                thr = 0.6 if (key_down(VK["W"]) or key_down(VK["UP"])) else 0.0
                brk = 1.0 if (key_down(VK["S"]) or key_down(VK["DOWN"])) else 0.0
                steer_out = STEER_OUT_SIGN * STEER_GAIN * last_steer
                if csp_ctrl is not None:
                    csp_ctrl.set(steer=CSP_STEER_SIGN * steer_out, gas=thr, brake=brk)
                if pad is not None:
                    e_now = key_down(VK["E"])
                    if e_now and not e_was:                # tap gear up
                        pad.press_button(); pad.update(); time.sleep(0.03); pad.release_button()
                    e_was = e_now
                    if hasattr(pad, "gear_down"):
                        pad.gear_down(key_down(VK["C"]))   # hold C = reverse gear
                    pad.left_joystick_float(x_value_float=steer_out)
                    pad.right_trigger_float(value_float=thr)
                    pad.left_trigger_float(value_float=brk)
                    pad.update()
                out_longi = thr - brk
                time.sleep(period); continue
            # ---- DECIDE at the policy's trained rate (10 Hz), not every wheel tick ----
            if now - last_decide >= policy_dt:
                last_decide = now
                gt = csp.read() if csp.available else None     # AC's real rays + collision + heading
                csp_coll = gt["collision"] if gt else 0.0
                # IMPORTANT: CSP physics.raycastTrack only hits the ROAD surface, NOT the walls/barriers
                # -> its rays read 'all clear' and the policy drove blind into walls. Use OUR
                # reconstructed-track rays (they DO see the walls, height-correct via center_y), and take
                # only the real HEADING + YAW from CSP (those are accurate).
                obs, info = ob.build(fr, last_steer, csp_rays=None,
                                     csp_look=(gt["look"] if gt else None),
                                     csp_yaw=(gt.get("yaw") if gt else None))
                a = agent.select_action(obs, deterministic=True)
                pol_steer = float(np.clip(a[0], -1, 1)); pol_longi = float(np.clip(a[1], -1, 1))
                # ---- anti-overspeed + anti-spin supervisor (sim->real grip margin) ----
                vx = info["vx_ms"]
                v_cap = SPEED_MARGIN * info["cvmax"]
                if vx > v_cap:                              # too hot for the corner ahead -> scrub
                    over = (vx - v_cap) / max(v_cap, 1.0)
                    pol_longi = min(pol_longi, -float(np.clip(2.5 * over, 0.15, 1.0)))
                # anti-spin: a CONTINUOUS taper (not a hard switch) so it can't flip-flop and make
                # cornering inconsistent. severity 0..1 grows with yaw rate / rear slip.
                slide = min(1.0, max((abs(info["r"]) - SPIN_YAW) / 2.0,
                                     (abs(info["slip_r"]) - SPIN_SLIP) / 0.25, 0.0))
                if slide > 0:
                    pol_longi = min(pol_longi, 0.3 * (1.0 - slide))   # lift off as the slide grows
                    pol_steer *= (1.0 - 0.2 * slide)                  # only a GENTLE lock ease (was 0.5,
                                                                      # which killed steering mid-corner)
                # wall-margin guard: nudge away from a wall it's about to scratch (sign: +steer=LEFT,
                # so a near RIGHT wall -> steer left, near LEFT wall -> steer right). Grows with how
                # close it is, so it leaves clean apexes alone and only reacts near contact.
                prox_l = max(0.0, (WALL_NEAR - info["left_clear"]) / WALL_NEAR)
                prox_r = max(0.0, (WALL_NEAR - info["right_clear"]) / WALL_NEAR)
                if prox_l > 0 or prox_r > 0:
                    pol_steer = float(np.clip(pol_steer + WALL_AVOID_GAIN * (prox_r - prox_l), -1, 1))
                    pol_longi = min(pol_longi, 0.2)         # ease off near a wall (grip to steer away)
                # safety supervisor — only when GENUINELY stuck (let the policy handle near-wall driving)
                speed = fr["speed"]; off = info["off"]
                backward = info["dir_cos"] < -0.3 and speed > 5
                # only "stuck" if off-track, going backward, OR stopped AND nosed into a wall.
                # a car stopped on CLEAR track is NOT stuck — let the policy just throttle it away.
                bad = off > 9.0 or backward or (speed < 1.0 and info["front"] < 2.5)
                stuck_since = (stuck_since or now) if bad else None
                recovering = stuck_since is not None and (now - stuck_since) > 1.2
                if recovering:
                    # steer toward the open track and push; reverse out only if nosed into a wall
                    pol_steer = float(np.clip(info["dir_sin"] * 1.8, -1, 1))
                    pol_longi = 0.6 if info["front"] > 1.8 else -0.6
                elif speed < 1.0 and not bad:
                    pol_longi = max(pol_longi, 0.5)             # kick a stopped car off the line
                if dlog is not None and mode == "ai":
                    dlog.log(info, pol_steer, pol_longi, recovering, speed, collision=csp_coll,
                             out_steer=last_steer)
            # ---- SMOOTH the wheel toward the decision every tick (60 Hz) -> human-like motion ----
            max_step = MAX_STEER_RATE * period
            target = last_steer + float(np.clip(pol_steer - last_steer, -max_step, max_step))
            last_steer = (1 - STEER_SMOOTH) * last_steer + STEER_SMOOTH * target
            out_longi = (1 - THR_SMOOTH) * out_longi + THR_SMOOTH * pol_longi
            longi = out_longi

            if args.dry:
                # AUTO-CALIBRATE heading: while DRIVING STRAIGHT (>5 m/s), compare AC's heading to the
                # actual travel direction. travel_dir is convention-FREE (the car goes where it goes),
                # so it tells us the true HEADING_OFFSET/SIGN to make the policy's frame match AC.
                ta = math.atan2(ob.t.tangent[info["prog"], 1], ob.t.tangent[info["prog"], 0])
                travel = math.atan2(fr["vel"][2], fr["vel"][0])
                travel_dircos = math.cos(ta - travel)
                if fr["speed"] / 3.6 > 5.0:                  # driving -> collect calibration samples
                    cal_pos.append(travel); cal_head.append(fr["heading"])
                    cal_pos[:] = cal_pos[-200:]; cal_head[:] = cal_head[-200:]
                if int(now * 5) % 3 == 0:
                    msg = ("\rprog %3d off %4.1f v %5.1f | TRAVEL-dir %+.2f  HEADING-dir %+.2f  "
                           "(both should be ~+1 driving straight) | grip %.2f front %4.1f %s   "
                           % (info["prog"], info["off"], fr["speed"], travel_dircos, info["dir_cos"],
                              info["grip"], info["front"], "RECOVER" if recovering else ""))
                    print(msg, end="", flush=True)
            else:
                # trim steering authority as speed rises -> kills high-speed snap-oversteer
                gain = STEER_GAIN * (1.0 - STEER_SPEED_FALLOFF * min(fr["speed"] / 3.6 / V_MAX, 1.0))
                steer_out = STEER_OUT_SIGN * gain * last_steer
                if csp_ctrl is not None:        # CSP in-engine control (Lua app slews steering smooth)
                    csp_ctrl.set(steer=CSP_STEER_SIGN * steer_out, gas=max(0.0, longi), brake=max(0.0, -longi))
                if pad is not None:
                    pad.left_joystick_float(x_value_float=steer_out)
                    pad.right_trigger_float(value_float=max(0.0, longi))
                    pad.left_trigger_float(value_float=max(0.0, -longi))
                    if hasattr(pad, "gear_down"):
                        pad.gear_down(recovering and longi < 0)
                    pad.update()
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        if dlog is not None:
            dlog.summary()
        if args.dry and len(cal_pos) > 20:
            # find HEADING_OFFSET/SIGN so HEADING_SIGN*heading + OFFSET == travel direction
            tp = np.array(cal_pos); th = np.array(cal_head)
            best = None
            for sign in (1.0, -1.0):
                diff = tp - sign * th
                off = math.atan2(np.sin(diff).mean(), np.cos(diff).mean())     # circular mean
                resid = np.std(np.angle(np.exp(1j * (diff - off))))           # circular spread
                if best is None or resid < best[2]:
                    best = (sign, off, resid)
            sign, off, resid = best
            print("\n\n=== HEADING CALIBRATION (from %d driving samples) ===" % len(cal_pos))
            print("  suggested:  HEADING_SIGN = %+.0f   HEADING_OFFSET = %+.4f   (fit spread %.3f rad)"
                  % (sign, off, resid))
            print("  %s" % ("GOOD fit — tell me these two numbers and I'll lock them in."
                            if resid < 0.25 else "noisy — drive a longer straight and retry."))
        if csp_ctrl is not None:
            csp_ctrl.release()                   # hand control back to the driver
        if pad is not None:
            pad.left_joystick_float(0.0); pad.right_trigger_float(0.0); pad.left_trigger_float(0.0)
            pad.update()


if __name__ == "__main__":
    main()
