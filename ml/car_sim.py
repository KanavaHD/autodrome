"""
Vectorised car simulator — hundreds of cars on the SAME real track, in parallel, faster than
real time. This is what unblocks Sophy-style training on one machine: instead of one real-time AC
instance, we run N independent cars (they ghost through each other) on the actual track geometry,
reset instantly, and pour millions of transitions into SAC in minutes.

  * Track   = your real AC track: the wall-occupancy grid from wall_sensor (rays cast against it),
              plus a centreline from one recorded lap (for progress reward + respawn points).
  * Physics = a REAL dynamic bicycle model: longitudinal drive/brake, per-axle tyre SLIP ANGLES,
              load transfer, and a FRICTION CIRCLE (lateral + longitudinal grip share one budget).
              This is the GT-Sophy-grade model: it can understeer, oversteer, scrub, and DRIFT, and
              it produces every dynamics value (slip, yaw rate, G-G, grip used) the policy + the
              viz consume. Domain-randomised per car (grip/power vary) for the sim-to-AC jump.
  * Reset   = a crashed car instantly respawns at a RANDOM point around the track (auto curriculum:
              it learns every corner, not just the start). No hotkey, no teleport, no AC.

  python ml/car_sim.py        # self-test: random actions, prints steps/sec + stability

obs = [vx, vy, yaw, slip, lonG, latG, grip, slipF, slipR, rays(16)]  (all reproducible from AC
telemetry via vehicle_state -> the policy transfers).  action = [steer, longi(throttle/brake)].
"""
import os, sys, glob, json, time, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wall_sensor import WallSensor, H_BELOW, H_ABOVE

RAY_MAX = 16.0
# DENSE, FULL-SURROUND LIDAR: 36 rays every 10 deg, all the way around (not just a 180 forward arc).
# The old 16-ray forward fan had 3.4 m gaps at range (walls hid BETWEEN beams) and was blind to the
# sides/rear — so cars slid into outer walls they couldn't see. This sees everything, much tighter.
FAN = list(range(-180, 180, 10))               # 36 rays, 360 deg coverage
DT = 0.1                                        # 10 Hz control step
NSUB = 5                                         # physics substeps (dt=0.02s) for a stable tyre model
V_MAX = 32.0                                    # m/s reference top speed (~115 km/h)
MAX_OFF = 7.0                                   # m from centreline -> off track (terminal)
# ---- CAR FOOTPRINT: the car is NOT a point. It's a real ~1.3x1.8m body, so its corners stick out
# ~0.9m from centre. Modelling it as a point let the policy learn lines that clip walls with the body.
# We give each ray the car's half-extent IN THAT DIRECTION (rectangle), so 'clearance = ray - extent'
# is the real gap from the BODY to the wall -> the policy keeps its actual body off the walls. ----
CAR_HALF_W = 0.62                              # car half-width (m)
CAR_HALF_L = 0.88                              # car half-length (m)
CRASH_MARGIN = 0.05                            # body-to-wall gap (m) under which it's a real contact
_fa = np.radians(np.array(FAN, float))
CAR_EXTENT = np.minimum(CAR_HALF_L / np.maximum(np.abs(np.cos(_fa)), 1e-3),
                         CAR_HALF_W / np.maximum(np.abs(np.sin(_fa)), 1e-3))   # (R,) half-extent per ray dir
CRASH_RAY = 0.5                                 # (legacy; superseded by the footprint model above)
SPAWN_LAT = 1.5                                 # lateral spread when respawning (m)
LAP_TARGET_S = 60.0                             # reference lap time (your ~1:00 best) — completing a
                                                # lap faster than this pays more, so it chases your pace

# ---- REWARD SHAPING (tuned to fix the AC symptoms: wall-scratching, late turn-in, needless steering).
#      Defined here so the CPU sim and the GPU sim (which imports these) stay identical. ----
RW_FWD          = 2.0      # progress along the track — the core objective (speed, right direction)
RW_WALL         = 2.6      # WALL-MARGIN penalty weight. Raised so it learns a SAFER, wider line (it was
                           # hugging walls -> 'tight line' clips in AC where geometry differs slightly).
RW_WALL_KEEP    = 1.3      # m of BODY clearance to keep (footprint-aware now). Start penalising when the
                           # car's BODY is within this of a wall -> it leaves real room, no body clips.
RW_SMOOTH       = 0.5      # steering-RATE penalty (was 0.04). Squared -> punishes jerks, allows fine trims
RW_STRAIGHT_STR = 0.4      # penalty for steering while the track is STRAIGHT (steer only when needed)
RW_ALIGN        = 0.6      # reward facing where the track GOES next -> anticipate, turn in ON TIME
RW_ALIGN_LOOK   = 5        # centreline points ahead to aim the heading at (~10 m) for the align reward
RW_GRIP         = 0.5      # ride near the grip limit (use the available grip)
RW_OVERGRIP     = 0.8      # penalty for exceeding the limit (sliding / scrubbing = slow)
RW_STRAIGHT_SPD = 0.5      # get on the power where the track is straight (corner exits + straights)
RW_CRASH        = 3.0      # terminal penalty for a wall hit / going off track

# ---- car dynamics parameters (a light, high-grip car) ----
M    = 180.0          # mass (kg, car + driver)
IZ   = 110.0          # yaw inertia (kg m^2)
LF   = 0.55           # CG -> front axle (m)
LR   = 0.55           # CG -> rear axle (m)
L    = LF + LR
HCG  = 0.30           # CG height (m) -> sets load transfer magnitude
G    = 9.81
MU0  = 1.45           # base tyre friction coefficient (cars grip hard)
F_DRIVE = 2600.0      # max drive force (N) -> ~14 m/s^2 launch
F_BRAKE = 4200.0      # max brake force (N)
STIFF   = 7.0         # tyre cornering stiffness shape (higher = sharper turn-in, saturates sooner)
C_ROLL  = 6.0         # rolling resistance (N per m/s)
C_AERO  = 2.55        # aero drag (N per (m/s)^2) -> balances drive near V_MAX
V_EPS   = 2.0         # low-speed slip-angle softening (avoids standstill singularity)
STEER_LOCK = 0.45     # rad of front-wheel angle at full steer input (=1.0). MEASURED from AC.
A_LAT_LIMIT = MU0 * G # peak lateral (cornering) accel — sets the max corner speed lookahead
# DOMAIN RANDOMISATION ranges (per-car, each episode) — so the policy is ROBUST to the residual
# sim-vs-AC mismatch instead of overfitting one exact physics. Centred on the measured values.
DR_GRIP = (0.82, 1.18)   # tyre grip  x this  (±18%)
DR_PWR  = (0.85, 1.15)   # drive power x this (±15%)
DR_LOCK = (0.85, 1.15)   # steering lock x this (±15%)
DR_BAL  = 0.09           # front/rear grip BALANCE spread: mu_f=mu*(1+b), mu_r=mu*(1-b), b in ±this.
                         # >0 = front grippier (oversteer-prone), <0 = understeer. Training across the
                         # range makes the policy ROBUST to AC's exact balance -> no more running-wide.

# ---- FAITHFUL AC TYRE MODEL (from the AC car's tyres.ini). This replaces the
# flat tanh (which saturated at 100% grip and NEVER fell off -> the policy learned it could over-slip
# for free, then spun in real AC). Per-axle peak grip + load sensitivity + a peak-then-falloff slip
# curve = AC's actual tyre behaviour, so the sim-trained policy transfers (the RocketSim approach). ----
TYRE_DY0_F = 1.630; TYRE_DY1_F = -0.043; TYRE_FZ0_F = 230.0   # front peak grip, load sens., ref load
TYRE_DY0_R = 1.660; TYRE_DY1_R = -0.043; TYRE_FZ0_R = 310.0   # rear  (rear grips harder -> stable)
TYRE_LSEXPY = 0.9                                             # load-sensitivity exponent (mu drops w/ load)
ALPHA_PEAK = math.radians(7.0)                               # FRICTION_LIMIT_ANGLE: grip peaks here
FALLOFF_LEVEL = 0.92                                        # grip retained far past the peak (sliding)
FALLOFF_SPEED = 4.0                                        # how fast grip drops past the peak
# ---- LONGITUDINAL response (measured from AC, calibrate_longi.py) ----
THR_GAMMA = 1.0          # partial-throttle softness: applied throttle = input^THR_GAMMA (AC ~1.6)
BRK_GAMMA = 1.0          # partial-brake shape
AERO_SCALE = 1.0         # scales aero drag so high-speed accel matches AC (<1 = less drag, more top-end)

# ---- CALIBRATION: override the above with YOUR real car, measured from the recorded laps
#      (data/car_calib.json, written by calibrate_from_laps.py). This is what makes the sim
#      behave like your actual car so the policy's "how steering/throttle/G affect the car"
#      knowledge is real and transfers to AC. Falls back to the defaults if no calib file. ----
_CALIB = os.path.join(HERE, "..", "data", "car_calib.json")
try:
    _c = json.load(open(_CALIB))
    # exact chassis from the AC car files (mass, yaw inertia, axle geometry, CG height)
    M = float(_c.get("mass_kg", M))
    IZ = float(_c.get("iz", IZ))
    LF = float(_c.get("lf", LF)); LR = float(_c.get("lr", LR)); L = LF + LR
    HCG = float(_c.get("hcg", HCG))
    MU0 = float(_c["mu"])
    V_MAX = float(_c["v_max_ms"])
    F_DRIVE = float(_c["a_accel_ms2"]) * M          # F = m*a  (real engine/gearing -> drive force)
    F_BRAKE = float(_c["a_brake_ms2"]) * M
    # rebalance aero drag so top speed still lands at the calibrated V_MAX (drag = drive at vmax)
    C_AERO = max(0.2, (F_DRIVE - C_ROLL * V_MAX) / (V_MAX * V_MAX))
    _l = _c.get("longi", {})                              # measured AC longitudinal response
    THR_GAMMA = float(_l.get("thr_gamma", THR_GAMMA))
    BRK_GAMMA = float(_l.get("brk_gamma", BRK_GAMMA))
    AERO_SCALE = float(_l.get("aero_scale", AERO_SCALE))
    C_AERO *= AERO_SCALE                                  # less drag -> sim holds accel at speed like AC's gears
    A_LAT_LIMIT = float(_c.get("a_lat_ms2", MU0 * G))    # calibrated cornering-grip limit
    STEER_LOCK = float(_c.get("steer_lock_rad", STEER_LOCK))   # measured AC steering lock
    _t = _c.get("tyre", {})                               # faithful AC tyre params (real ini values)
    TYRE_DY0_F = float(_t.get("dy0_f", TYRE_DY0_F)); TYRE_DY1_F = float(_t.get("dy1_f", TYRE_DY1_F))
    TYRE_FZ0_F = float(_t.get("fz0_f", TYRE_FZ0_F))
    TYRE_DY0_R = float(_t.get("dy0_r", TYRE_DY0_R)); TYRE_DY1_R = float(_t.get("dy1_r", TYRE_DY1_R))
    TYRE_FZ0_R = float(_t.get("fz0_r", TYRE_FZ0_R))
    TYRE_LSEXPY = float(_t.get("ls_expy", TYRE_LSEXPY))
    ALPHA_PEAK = math.radians(float(_t.get("friction_limit_angle_deg", 7.0)))
    FALLOFF_LEVEL = float(_t.get("falloff_level", FALLOFF_LEVEL))
    FALLOFF_SPEED = float(_t.get("falloff_speed", FALLOFF_SPEED))
    print("[car_sim] AC-calibrated: mu=%.2f vmax=%.1f m/s accel=%.1f brake=%.1f lock=%.3frad | mass=%.0fkg Iz=%.0f LF/LR=%.2f/%.2f"
          % (MU0, V_MAX, F_DRIVE / M, F_BRAKE / M, STEER_LOCK, M, IZ, LF, LR))
except Exception:
    pass


def tyre_force_frac(alpha):
    """AC-faithful lateral-grip fraction (0..1) vs slip angle: rises to the PEAK at ALPHA_PEAK, then
    FALLS OFF toward FALLOFF_LEVEL (the tyre is sliding). Signed. This is what the flat tanh lacked —
    past the limit a real tyre gives LESS grip, which teaches the policy not to over-slip."""
    x = np.abs(alpha) / ALPHA_PEAK
    rising = np.sin(0.5 * np.pi * np.clip(x, 0.0, 1.0))               # 0 -> 1 up to the peak
    falloff = FALLOFF_LEVEL + (1.0 - FALLOFF_LEVEL) * np.exp(-np.maximum(x - 1.0, 0.0) * FALLOFF_SPEED)
    return np.sign(alpha) * np.where(x <= 1.0, rising, falloff)


class Track:
    """Height-aware (multi-level) track. Some AC tracks pass OVER themselves (a top floor above
    a lower section), so a flat 2D grid would merge the two levels — phantom walls, floor-jumping.
    We keep each wall cell's HEIGHT SPAN (ymin..ymax) and a 3D centreline carrying height, so a car
    on the lower floor never sees the upper floor's wall (and vice-versa)."""

    def __init__(self, geom=None):
        if geom is not None:
            self._init_from_geom(geom)                       # an ingested AC track (track_ingest.py)
        else:
            ws = WallSensor()
            if not hasattr(ws, "_ymin"):
                ws._build_grid()
            self.cell = ws.cell; self.gmin = ws._gmin; self.H0, self.H1 = ws._H0, ws._H1
            # KEEP the height grids (not a flat boolean): ymax<-1e29 means "no wall here".
            self.ymin = ws._ymin.copy(); self.ymax = ws._ymax.copy()
            # remove PHANTOM walls the car demonstrably drove through — but HEIGHT-AWARE, so driving
            # the lower floor doesn't delete the upper floor's wall sitting above it.
            self._pruned = self._prune_phantom_walls()
            # a clean, wall-CENTRED, closed, 3D racing line
            self.center, self.center_y, self.tangent, self.halfwidth, self.cumlen = self._centreline()
        self.K = len(self.center)
        self.spawn_idx = np.where(self.halfwidth > 1.2)[0]
        if len(self.spawn_idx) < 10:
            self.spawn_idx = np.arange(self.K)
        # LOOKAHEAD: the signed turn angle of the track over the next few horizons (this is what lets
        # a policy BRAKE BEFORE a corner instead of reacting — Sophy's key 'sees the course ahead').
        self.ahead_h = [3, 6, 12, 20]                      # horizons in centreline points (~6..40 m)
        ta = np.arctan2(self.tangent[:, 1], self.tangent[:, 0])
        self.curve_ahead = np.zeros((self.K, len(self.ahead_h)))
        for hi, H in enumerate(self.ahead_h):
            d = ta[(np.arange(self.K) + H) % self.K] - ta          # heading change over the horizon
            self.curve_ahead[:, hi] = np.arctan2(np.sin(d), np.cos(d))   # wrapped signed angle
        # total lap length (m) — for detecting a completed lap + measuring lap time
        self.length = float(self.cumlen[-1] + np.linalg.norm(self.center[0] - self.center[-1]))
        # MAX CORNER SPEED ahead (the turn-timing signal): the fastest the car could be going and
        # still make the tightest bit coming up, at the grip limit (v = sqrt(a_lat * R)). The policy
        # compares its speed to this -> learns WHEN to brake and HOW fast to carry corners.
        # NOTE: we use LOCAL curvature over a short window, NOT the net heading change over the whole
        # horizon. In a chicane the right- and left-hand halves cancel, making the net angle ~0 and
        # the old formula spike the cap to vmax mid-esse (the car then enters flat-out and washes
        # into the wall). Local curvature sees each tight part; a forward rolling-min over the horizon
        # then gives 'the slowest speed required anywhere in the next Hh points'.
        a_lat = A_LAT_LIMIT
        spacing = self.length / self.K
        w = 3                                                      # local-curvature window (points)
        dloc = ta[(np.arange(self.K) + w) % self.K] - ta
        dloc = np.arctan2(np.sin(dloc), np.cos(dloc))              # local heading change (no cancel)
        arc_w = max(w * spacing, 1.0)
        R_loc = arc_w / (np.abs(dloc) + 1e-3)                      # local corner radius (m)
        v_loc = np.clip(np.sqrt(a_lat * R_loc), 3.0, V_MAX)        # local grip-limited speed
        self.corner_vmax = np.full((self.K, len(self.ahead_h)), V_MAX)
        for hi, Hh in enumerate(self.ahead_h):
            for k in range(Hh + 1):                                # forward rolling-min over horizon
                self.corner_vmax[:, hi] = np.minimum(
                    self.corner_vmax[:, hi], v_loc[(np.arange(self.K) + k) % self.K])

    def _init_from_geom(self, geom):
        """Build the track from an ingested AC fast_lane.ai (see ml/track_ingest.py): take the
        centreline arrays directly, and rasterise the height-aware occupancy grid the LIDAR raycast
        needs from the compact wall-edge points (centre ± side). AC tracks are single-level, so a
        uniform wall height span over each edge cell is all the height test needs."""
        g = lambda k: np.asarray(geom[k])
        self.center = g("center").astype(float)
        self.center_y = g("center_y").astype(float)
        self.tangent = g("tangent").astype(float)
        self.halfwidth = g("halfwidth").astype(float)
        self.cumlen = g("cumlen").astype(float)
        self.cell = float(geom["cell"])
        self._pruned = 0
        edges = g("edges").astype(float)                     # (M, 3): x, z, y
        c = self.cell
        gx = np.round(edges[:, 0] / c).astype(np.int64)
        gz = np.round(edges[:, 1] / c).astype(np.int64)
        self.gmin = np.array([gx.min() - 1, gz.min() - 1])
        self.H0 = int(gx.max() - self.gmin[0] + 2); self.H1 = int(gz.max() - self.gmin[1] + 2)
        self.ymin = np.full((self.H0, self.H1), 1e30)
        self.ymax = np.full((self.H0, self.H1), -1e30)
        ix = gx - self.gmin[0]; iz = gz - self.gmin[1]
        lo = edges[:, 2] - 0.2; hi = edges[:, 2] + 2.0       # a 2 m-tall wall at each edge cell
        for dx in (-1, 0, 1):                                # 1-cell dilation -> no diagonal gaps
            for dz in (-1, 0, 1):
                jx = np.clip(ix + dx, 0, self.H0 - 1); jz = np.clip(iz + dz, 0, self.H1 - 1)
                np.minimum.at(self.ymin, (jx, jz), lo)
                np.maximum.at(self.ymax, (jx, jz), hi)
        # REAL WALLS: if this track has been voxelised from its .kn5 mesh, replace the fast_lane
        # "side-distance" walls (which put phantom barriers along open straights -> oversteer) with the
        # true 1WALL geometry, and attach the surface-class grid so grass/sand become a grip penalty
        # rather than a ray obstacle. Falls back silently to the edge walls if no voxels exist.
        self.surf = None
        try:
            self._apply_voxels(str(geom.get("name", "")))
        except Exception as ex:
            print("[car_sim] voxel walls unavailable (%s) — using fast_lane edges" % type(ex).__name__)

    def _apply_voxels(self, key):
        """Override the ray occupancy grid with the real WALL voxels and store the surface-class grid."""
        key = key.replace("/", "__")
        if not key:
            return
        import voxelize as VX
        path = os.path.join(HERE, "tracks", key + "_voxels.npz")
        if not os.path.exists(path):
            return
        v = VX.load_voxels(key)
        cls = v["cls"]; wmin = v["wall_min"]; wmax = v["wall_max"]
        vox_cell = float(v["cell"]); ox, oz = [float(x) for x in v["origin"]]
        # rebuild ymin/ymax in the sim's OWN cell-rounded world indexing (so raycast() is unchanged)
        self.cell = vox_cell; c = self.cell
        wx, wz = np.where(cls == VX.CLS_WALL)
        worldx = ox + (wx + 0.5) * c; worldz = oz + (wz + 0.5) * c
        wy_lo = wmin[wx, wz]; wy_hi = wmax[wx, wz]
        gx = np.round(worldx / c).astype(np.int64); gz = np.round(worldz / c).astype(np.int64)
        self.gmin = np.array([gx.min() - 1, gz.min() - 1])
        self.H0 = int(gx.max() - self.gmin[0] + 2); self.H1 = int(gz.max() - self.gmin[1] + 2)
        self.ymin = np.full((self.H0, self.H1), 1e30); self.ymax = np.full((self.H0, self.H1), -1e30)
        ix = gx - self.gmin[0]; iz = gz - self.gmin[1]
        for dx in (-1, 0, 1):                                # dilate 1 cell -> solid barrier, no gaps
            for dz in (-1, 0, 1):
                jx = np.clip(ix + dx, 0, self.H0 - 1); jz = np.clip(iz + dz, 0, self.H1 - 1)
                np.minimum.at(self.ymin, (jx, jz), wy_lo - 0.2)
                np.maximum.at(self.ymax, (jx, jz), wy_hi + 0.3)
        # surface-class grid (for the off-track grip penalty): drivable = road/kerb/pit, off = grass/sand
        DRIVABLE = {VX.CLS_ROAD, VX.CLS_KERB, VX.CLS_PIT}
        OFF = {VX.CLS_GRASS: 0.62, VX.CLS_SAND: 0.42}        # grip multiplier off the racing surface
        gripmul = np.ones(cls.shape, np.float32)
        for c_id, mul in OFF.items():
            gripmul[cls == c_id] = mul
        onroad = np.isin(cls, list(DRIVABLE))
        self.surf = {"gripmul": gripmul, "onroad": onroad, "origin": np.array([ox, oz], np.float32),
                     "cell": np.float32(vox_cell), "shape": np.array(cls.shape, np.int32)}
        nwall = int((cls == VX.CLS_WALL).sum())
        print("[car_sim] REAL walls from voxels: %d wall cells, surface grid %dx%d (grass/sand = grip penalty)"
              % (nwall, cls.shape[0], cls.shape[1]))

    @staticmethod
    def _tangents(P):
        TG = np.roll(P, -1, 0) - P
        return TG / (np.linalg.norm(TG, axis=1, keepdims=True) + 1e-9)

    def _driven_points(self):
        """All (x, z, y) the car drove (speed>3) across every lap, telemetry GLITCHES filtered out
        (position jumps >3 m). Height is kept so pruning can be height-aware."""
        pts = []
        for f in sorted(glob.glob(os.path.join(HERE, "..", "data", "laps", "lap_*.json"))):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            prev = None
            for p in d.get("points", []):
                q = p.get("p")
                if not q or p.get("speed", 0) <= 3:
                    prev = None
                    continue
                xz = (q[0], q[2])
                if prev is not None and (xz[0] - prev[0]) ** 2 + (xz[1] - prev[1]) ** 2 > 9.0:
                    continue
                pts.append((q[0], q[2], q[1])); prev = xz
        return np.array(pts, float) if pts else np.zeros((0, 3))

    def _prune_phantom_walls(self, radius=0.4):
        """Clear a wall cell only if its height span lies ENTIRELY within the band you drove through
        at that (x,z) — i.e. it's a false wall at your level. A tall wall that also reaches another
        floor is left intact (the height-aware raycast separates the floors)."""
        P = self._driven_points()
        if len(P) == 0:
            return 0
        c = self.cell
        gx = np.clip(np.round(P[:, 0] / c).astype(np.int64) - self.gmin[0] + 1, 0, self.H0 - 1)
        gz = np.clip(np.round(P[:, 1] / c).astype(np.int64) - self.gmin[1] + 1, 0, self.H1 - 1)
        y = P[:, 2]
        rad = int(np.ceil(radius / c))
        offs = [(dx, dz) for dx in range(-rad, rad + 1) for dz in range(-rad, rad + 1)
                if (dx * c) ** 2 + (dz * c) ** 2 <= radius * radius]
        before = int((self.ymax > -1e29).sum())
        lo = y - H_BELOW; hi = y + H_ABOVE
        for dx, dz in offs:
            ix = np.clip(gx + dx, 0, self.H0 - 1); iz = np.clip(gz + dz, 0, self.H1 - 1)
            cell_lo = self.ymin[ix, iz]; cell_hi = self.ymax[ix, iz]
            # phantom = the cell's whole wall span is inside the height you drove through here
            phantom = (cell_hi > -1e29) & (cell_lo >= lo) & (cell_hi <= hi)
            kill = ix[phantom], iz[phantom]
            self.ymax[kill] = -1e30; self.ymin[kill] = 1e30
        return before - int((self.ymax > -1e29).sum())

    def _recorded_path(self):
        """One clean recorded lap, resampled + smoothed, in 3D (x, z, y) — gives topology, direction
        AND height. We use its shape; the geometry below re-centres the xz between the walls."""
        for f in sorted(glob.glob(os.path.join(HERE, "..", "data", "laps", "lap_*.json"))):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if not (25 <= d.get("timeMs", 0) / 1000 <= 150):
                continue
            pts = [p["p"] for p in d.get("points", []) if p.get("p") and p.get("speed", 0) > 3]
            if len(pts) < 200:
                continue
            P = np.array([[p[0], p[2], p[1]] for p in pts], float)        # x, z, y
            seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
            t = np.linspace(0, seg[-1], 300, endpoint=False)
            C = np.stack([np.interp(t, seg, P[:, 0]), np.interp(t, seg, P[:, 1]),
                          np.interp(t, seg, P[:, 2])], 1)
            for _ in range(3):
                C = 0.25 * np.roll(C, 1, 0) + 0.5 * C + 0.25 * np.roll(C, -1, 0)
            return C
        raise RuntimeError("no usable recorded lap to seed the centreline")

    def _centreline(self):
        raw = self._recorded_path()                          # (300,3) x,z,y
        closed = np.vstack([raw, raw[:1]])
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(closed[:, :2], axis=0), axis=1))]
        K = 360
        t = np.linspace(0, seg[-1], K, endpoint=False)
        P = np.stack([np.interp(t, seg, closed[:, 0]), np.interp(t, seg, closed[:, 1])], 1)
        Y = np.interp(t, seg, closed[:, 2])                  # height carried along the line
        P0 = P.copy()                                        # the REAL driven line — ground truth
        # RE-CENTRE xz to the MIDDLE of the corridor — HEIGHT-AWARE. The recorded line hugs the
        # inside railing on the narrow ramp (only ~0.6 m clearance there -> every car crashes), but
        # the corridor is actually ~3 m wide. Centre it properly: iterate enough to converge, and
        # cap drift only loosely (3 m) so it can reach the middle without wandering onto a railing.
        MAX_DRIFT = 3.0
        ar = np.radians([90.0, -90.0]); rc = np.cos(ar); rs = np.sin(ar)
        TARGET = 2.2                                          # desired clearance when only one wall is seen
        for _ in range(10):
            TG = self._tangents(P)
            clr = self.raycast(P, TG, rc, rs, Y)             # (K,2): clearance left, right at height Y
            Ld, Rd = clr[:, 0], clr[:, 1]
            both = (Ld < 14) & (Rd < 14)
            only_r = (Ld >= 14) & (Rd < 14)                  # open LEFT, wall RIGHT -> move left (+)
            only_l = (Ld < 14) & (Rd >= 14)                  # open RIGHT, wall LEFT -> move right (-)
            shift = np.zeros(len(P))
            shift[both] = ((Ld - Rd) * 0.5)[both]            # both walls -> centre between them
            shift[only_r] = np.clip(TARGET - Rd[only_r], 0, 3.0)     # push off the close right wall
            shift[only_l] = -np.clip(TARGET - Ld[only_l], 0, 3.0)    # push off the close left wall
            shift = np.clip(shift, -3.0, 3.0)
            N = np.stack([-TG[:, 1], TG[:, 0]], 1)
            P = P + N * shift[:, None]
            P = 0.25 * np.roll(P, 1, 0) + 0.5 * P + 0.25 * np.roll(P, -1, 0)   # light smooth
            d = P - P0
            dist = np.linalg.norm(d, axis=1, keepdims=True)
            P = P0 + d * np.clip(MAX_DRIFT / (dist + 1e-9), 0, 1)
        TG = self._tangents(P)
        clr = self.raycast(P, TG, rc, rs, Y)
        halfwidth = np.minimum(clr[:, 0], clr[:, 1])
        seglen = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        cumlen = np.r_[0, np.cumsum(seglen)][:K]
        return P, Y, TG, halfwidth, cumlen

    def raycast(self, pos, head, rc, rs, cy):
        """Batched HEIGHT-AWARE rays. pos (N,2), head (N,2), cy (N,) = each caster's height. A cell
        blocks only if its wall height-span overlaps [cy-H_BELOW, cy+H_ABOVE] — so a lower-floor
        car's rays pass UNDER the upper floor, and an upper-floor car's rays ignore walls below."""
        c = self.cell; START = 0.45; step = c * 0.6
        steps = np.arange(START, RAY_MAX, step)            # (S,)
        dirx = head[:, 0:1] * rc[None, :] - head[:, 1:2] * rs[None, :]   # (N,R)
        dirz = head[:, 0:1] * rs[None, :] + head[:, 1:2] * rc[None, :]
        px = pos[:, 0:1, None] + dirx[:, :, None] * steps[None, None, :]  # (N,R,S)
        pz = pos[:, 1:2, None] + dirz[:, :, None] * steps[None, None, :]
        gx = np.clip(np.round(px / c).astype(np.int64) - self.gmin[0] + 1, 0, self.H0 - 1)
        gz = np.clip(np.round(pz / c).astype(np.int64) - self.gmin[1] + 1, 0, self.H1 - 1)
        wmax = self.ymax[gx, gz]; wmin = self.ymin[gx, gz]               # (N,R,S)
        lo = (cy - H_BELOW)[:, None, None]; hi = (cy + H_ABOVE)[:, None, None]
        blocked = (wmax >= lo) & (wmin <= hi)                            # height-aware
        has = blocked.any(2); first = np.argmax(blocked, 2)
        return np.where(has, steps[first], RAY_MAX)

    def localize(self, pos, prog, window=18):
        """FLOOR-SAFE progress update: re-find each car's position along the centreline by searching
        only NEAR its current progress (±window points), so at a stacked spot it stays on its own
        floor instead of snapping to the level above/below. Returns (new_prog_index, lateral_off)."""
        n = len(pos)
        offs = np.arange(-window, window + 1)
        idx = (np.round(prog).astype(np.int64)[:, None] + offs[None, :]) % self.K   # (n, W)
        cx = self.center[idx, 0]; cz = self.center[idx, 1]                          # (n, W)
        d2 = (cx - pos[:, 0:1]) ** 2 + (cz - pos[:, 1:2]) ** 2
        j = d2.argmin(1)
        new_idx = idx[np.arange(n), j]
        off = np.sqrt(d2[np.arange(n), j])
        return new_idx, off

    def wall_points(self, maxpts=2500):
        """World-space (x,z) of wall cells, downsampled — for drawing real walls in the viz."""
        gx, gz = np.where(self.ymax > -1e29)
        x = (gx - 1 + self.gmin[0]) * self.cell
        z = (gz - 1 + self.gmin[1]) * self.cell
        if len(x) > maxpts:
            sel = np.linspace(0, len(x) - 1, maxpts).astype(int)
            x, z = x[sel], z[sel]
        return np.stack([x, z], 1)


class VecCarSim:
    def __init__(self, n=256, seed=0):
        self.n = n
        self.track = Track()
        self.rng = np.random.default_rng(seed)
        ar = np.radians(np.array(FAN, float)); self.rc = np.cos(ar); self.rs = np.sin(ar)
        self.n_rays = len(FAN)
        self.n_dyn = 10                                    # dynamics + grip-used + grip-margin
        self.n_dir = 2                                     # heading-vs-track (cos,sin of error)
        self.n_ahead = len(self.track.ahead_h)            # track curvature lookahead
        self.n_pred = len(self.track.ahead_h)             # speed-vs-corner-limit (turn timing)
        self.obs_dim = self.n_dyn + self.n_dir + self.n_ahead + self.n_pred + self.n_rays
        self.act_dim = 2
        self.prev_steer = np.zeros(n)                      # for the smoothness reward
        # kinematic state
        self.pos = np.zeros((n, 2)); self.psi = np.zeros(n)          # heading angle (rad)
        self.vx = np.zeros(n); self.vy = np.zeros(n); self.r = np.zeros(n)  # body-frame vels + yaw rate
        # last-step dynamics (exposed for obs + viz)
        self.lonG = np.zeros(n); self.latG = np.zeros(n); self.grip = np.zeros(n)
        self.slip_f = np.zeros(n); self.slip_r = np.zeros(n); self.slip_b = np.zeros(n)
        # per-car domain randomisation
        self.mu = np.zeros(n); self.pwr = np.zeros(n); self.steer_lock = np.zeros(n)
        self.gbal = np.zeros(n)                            # front/rear grip balance (domain rand.)
        # MULTI-LEVEL state: progress along the 3D centreline (which FLOOR/where on the lap) + height
        self.prog = np.zeros(n, np.int64)      # nearest centreline index (tracked locally -> floor-safe)
        self.cy = np.zeros(n)                  # current height (m) -> drives the height-aware rays
        # LAP tracking: arc length covered this lap + steps, so we can detect completion & time it
        self.lap_dist = np.zeros(n); self.lap_steps = np.zeros(n, np.int64)
        self.last_lap_times = np.zeros(0)      # lap times (s) of cars that finished THIS step
        # crash bookkeeping for the viz
        self.n_wall = 0; self.n_off = 0; self.crash_pos = np.zeros((0, 2))
        self.reset(np.arange(n))

    @property
    def head(self):
        return np.stack([np.cos(self.psi), np.sin(self.psi)], 1)

    @property
    def spd(self):                                          # ground speed (for viz/reward)
        return np.sqrt(self.vx ** 2 + self.vy ** 2)

    def reset(self, idx):
        m = len(idx)
        # spawn ONLY where the corridor is wide enough (no more spawning inside a wall and dying)
        k = self.track.spawn_idx[self.rng.integers(0, len(self.track.spawn_idx), m)]
        C = self.track.center[k]; T = self.track.tangent[k]
        nrm = np.stack([-T[:, 1], T[:, 0]], 1)
        hw = self.track.halfwidth[k]                       # local clearance -> keep spawns on track
        lat = self.rng.normal(0, 1, (m, 1)) * np.clip(hw[:, None] * 0.4, 0.2, SPAWN_LAT)
        self.pos[idx] = C + nrm * lat
        self.psi[idx] = np.arctan2(T[:, 1], T[:, 0])       # face along the track
        self.prog[idx] = k                                 # spawned AT centreline index k...
        self.cy[idx] = self.track.center_y[k]              # ...so its floor/height is known
        self.vx[idx] = self.rng.uniform(0, 6, m); self.vy[idx] = 0.0; self.r[idx] = 0.0
        self.lonG[idx] = 0; self.latG[idx] = 0; self.grip[idx] = 0
        self.slip_f[idx] = 0; self.slip_r[idx] = 0; self.slip_b[idx] = 0
        self.mu[idx] = self.rng.uniform(*DR_GRIP, m)              # per-car grip SCALE (base mu from tyre model)
        self.pwr[idx] = self.rng.uniform(*DR_PWR, m)              # power
        self.steer_lock[idx] = STEER_LOCK * self.rng.uniform(*DR_LOCK, m)   # steering lock
        self.gbal[idx] = self.rng.uniform(-DR_BAL, DR_BAL, m)     # front/rear grip balance
        self.prev_steer[idx] = 0.0
        self.lap_dist[idx] = 0.0; self.lap_steps[idx] = 0

    # ---- observation: dynamics + GRIP + DIRECTION + LOOKAHEAD + TURN-TIMING + rays ----
    def _obs(self, rays):
        dyn = np.stack([
            self.vx / V_MAX,                # forward speed
            self.vy / 10.0,                 # lateral velocity (sideslip / drift)
            self.r / 3.0,                   # yaw rate
            self.slip_b / 0.6,              # body slip angle
            self.lonG / (1.5 * G),          # longitudinal G (friction circle x)
            self.latG / (1.5 * G),          # lateral G (friction circle y)
            self.grip,                      # fraction of grip USED (0..1+)
            np.clip(1.0 - self.grip, 0, 1), # grip MARGIN left (how much it can still push)
            self.slip_f / 0.4,              # front tyre slip angle
            self.slip_r / 0.4,              # rear tyre slip angle
        ], 1)
        # DIRECTION: where the track points vs where the car points. cos>0 = roughly forward,
        # cos<0 = facing BACKWARD (kills wrong-way wandering); sin = which way to steer to align.
        tang = self.track.tangent[self.prog]
        ta = np.arctan2(tang[:, 1], tang[:, 0]); err = ta - self.psi
        direction = np.stack([np.cos(err), np.sin(err)], 1)
        # LOOKAHEAD: how the track bends over the next horizons (brake-before-the-corner signal)
        ahead = self.track.curve_ahead[self.prog] / (np.pi / 2)        # normalise (~±1 for a 90deg bend)
        # TURN TIMING: speed vs the max safe speed for the bend ahead. >0 = TOO FAST, brake now;
        # <0 = room to GO FASTER. This is the explicit 'when to turn / when to floor it' signal.
        cvmax = self.track.corner_vmax[self.prog]                      # (n, nH)
        excess = np.clip(self.vx[:, None] / cvmax - 1.0, -1.5, 1.5)
        return np.concatenate([np.clip(dyn, -3, 3), direction, np.clip(ahead, -3, 3),
                               excess, np.clip(rays / RAY_MAX, 0, 1.2)], 1)

    def reset_all(self):
        self.reset(np.arange(self.n))
        rays = self.track.raycast(self.pos, self.head, self.rc, self.rs, self.cy)
        return self._obs(rays)

    # ---- the dynamic vehicle model (vectorised, substepped) ----
    def _dynamics(self, steer, longi):
        """Integrate the bicycle model NSUB substeps. Updates pos/psi/vx/vy/r and the exposed
        dynamics (lonG, latG, grip, slip_*). steer,longi in [-1,1]."""
        h = DT / NSUB
        delta = steer * self.steer_lock                     # per-car steering lock (AC-calibrated + DR)
        thr = np.clip(longi, 0, 1) ** THR_GAMMA * self.pwr  # AC throttle gamma (softer partial throttle)
        brk = np.clip(-longi, 0, 1) ** BRK_GAMMA
        for _ in range(NSUB):
            vxs = np.maximum(self.vx, V_EPS)                # soften low-speed slip singularity
            # tyre slip angles (front/rear), body slip
            af = np.arctan2(self.vy + LF * self.r, vxs) - delta
            ar_ = np.arctan2(self.vy - LR * self.r, vxs)
            # longitudinal forces: rear drive, brakes both axles; drag opposes motion
            fx_drive = thr * F_DRIVE
            fx_brk = brk * F_BRAKE * np.sign(self.vx + 1e-6)
            fx_f = -0.4 * fx_brk
            fx_r = fx_drive - 0.6 * fx_brk
            drag = (C_ROLL * self.vx + C_AERO * self.vx * np.abs(self.vx))
            ax_est = (fx_f + fx_r - drag) / M               # for load transfer
            # vertical load with longitudinal transfer (brake->front, accel->rear)
            fzf = np.maximum(M * (G * LR - ax_est * HCG) / L, 50.0)
            fzr = np.maximum(M * (G * LF + ax_est * HCG) / L, 50.0)
            # LOAD-SENSITIVE peak grip per axle (AC tyres.ini): mu DROPS as vertical load rises.
            nf_f = np.clip(fzf / TYRE_FZ0_F, 0.1, 6.0)
            nf_r = np.clip(fzr / TYRE_FZ0_R, 0.1, 6.0)
            mu_f = (TYRE_DY0_F + TYRE_DY1_F * (nf_f - 1.0)) * nf_f ** (TYRE_LSEXPY - 1.0)
            mu_r = (TYRE_DY0_R + TYRE_DY1_R * (nf_r - 1.0)) * nf_r ** (TYRE_LSEXPY - 1.0)
            cap_f = mu_f * self.mu * (1.0 + self.gbal) * fzf      # self.mu = per-car grip SCALE (DR)
            cap_r = mu_r * self.mu * (1.0 - self.gbal) * fzr
            # friction circle: longitudinal use eats into lateral budget
            fxf_c = np.clip(fx_f, -cap_f, cap_f)
            fxr_c = np.clip(fx_r, -cap_r, cap_r)
            fy_cap_f = np.sqrt(np.maximum(cap_f ** 2 - fxf_c ** 2, 0.0))
            fy_cap_r = np.sqrt(np.maximum(cap_r ** 2 - fxr_c ** 2, 0.0))
            # lateral force with the AC peak-then-falloff slip curve (replaces tanh)
            fyf = -fy_cap_f * tyre_force_frac(af)
            fyr = -fy_cap_r * tyre_force_frac(ar_)
            cd, sd = np.cos(delta), np.sin(delta)
            # body-frame equations of motion
            Fx = fxr_c + fxf_c * cd - fyf * sd - drag
            Fy = fyr + fyf * cd + fxf_c * sd
            ax = Fx / M; ay = Fy / M
            vx_dot = ax + self.vy * self.r
            vy_dot = ay - self.vx * self.r
            r_dot = (LF * (fyf * cd + fxf_c * sd) - LR * fyr) / IZ
            self.vx = np.clip(self.vx + vx_dot * h, -4.0, V_MAX)
            self.vy = self.vy + vy_dot * h
            self.r = self.r + r_dot * h
            self.psi = self.psi + self.r * h
            # global-frame position update
            c, s = np.cos(self.psi), np.sin(self.psi)
            self.pos[:, 0] += (self.vx * c - self.vy * s) * h
            self.pos[:, 1] += (self.vx * s + self.vy * c) * h
        # expose last-substep dynamics for obs + viz
        self.lonG = ax; self.latG = ay
        self.grip = np.maximum(np.sqrt(fxf_c ** 2 + fyf ** 2) / (cap_f + 1e-6),
                               np.sqrt(fxr_c ** 2 + fyr ** 2) / (cap_r + 1e-6))
        self.slip_f = af; self.slip_r = ar_
        self.slip_b = np.arctan2(self.vy, np.maximum(self.vx, 0.1))

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        steer = a[:, 0]; longi = a[:, 1]
        prev = self.pos.copy()
        self._dynamics(steer, longi)

        # FLOOR-SAFE localisation: update each car's progress along the 3D centreline by searching
        # only near its previous progress, so a car on the lower floor can't snap to the upper floor
        # stacked above it. Its height (cy) then comes from the centreline -> height-aware rays.
        i, off = self.track.localize(self.pos, self.prog)
        self.prog = i; self.cy = self.track.center_y[i]
        rays = self.track.raycast(self.pos, self.head, self.rc, self.rs, self.cy)
        # crash = a wall within reach in ANY direction (a real touch, not just straight ahead)
        nearest_wall = (rays - CAR_EXTENT[None, :]).min(1)   # BODY-to-wall clearance (footprint-aware)
        tang = self.track.tangent[i]
        fwd = ((self.pos - prev) * tang).sum(1)
        spd = self.spd
        # SMART REWARD — the sweet spot, not pure recklessness:
        #  + progress along the track  (rewards speed; wrong-way is automatically negative)
        #  - WALL-MARGIN penalty, scaled by speed: continuously discourages hugging/charging walls
        #    BEFORE the crash, which (with the lookahead obs) is what teaches braking for corners
        #  - small SMOOTHNESS penalty on steering change: clean inputs, not twitchy (helps AC transfer)
        reward = RW_FWD * fwd
        # WALL MARGIN: squared so it's gentle far out but firm near contact; speed-scaled (a fast
        # scrape is worse). Wider keep-clear distance -> the policy leaves a real buffer (no scraping).
        near = np.clip((RW_WALL_KEEP - nearest_wall) / RW_WALL_KEEP, 0.0, 1.0)
        reward -= RW_WALL * near * near * (0.4 + 0.6 * spd / V_MAX)
        # SMOOTHNESS: penalise CHANGING the wheel (squared) -> deliberate steering, no twitch/saw
        reward -= RW_SMOOTH * (steer - self.prev_steer) ** 2
        self.prev_steer = steer.copy()
        # STRAIGHT pre-compute (small bend ahead = straight): used by 3 terms below
        straight = 1.0 - np.clip(np.abs(self.track.curve_ahead[i, 1]) / 0.4, 0, 1)
        # ANTICIPATION + NO NEEDLESS STEER: reward heading aligned with the track a few points AHEAD.
        # Turning LATE -> misaligned through the corner -> penalty (pulls turn-in earlier). Steering on
        # a straight -> heading diverges from the straight track ahead -> penalty (only steer when needed).
        look_ang = np.arctan2(self.track.tangent[(i + RW_ALIGN_LOOK) % self.track.K, 1],
                              self.track.tangent[(i + RW_ALIGN_LOOK) % self.track.K, 0])
        reward += RW_ALIGN * (np.cos(look_ang - self.psi) - 1.0)
        reward -= RW_STRAIGHT_STR * straight * steer ** 2       # don't saw the wheel on the straight
        # DRIVE-THE-LIMIT: use the grip, don't exceed it (sliding = slow)
        reward += RW_GRIP * np.clip(self.grip, 0, 1.0)
        reward -= RW_OVERGRIP * np.clip(self.grip - 1.05, 0, 1.0)
        # get on the power where it's straight (corner exits + straights)
        reward += RW_STRAIGHT_SPD * straight * (self.vx / V_MAX)
        # LAP COMPLETION — the real racing objective. Accumulate distance along the lap; when a car
        # covers a full lap length, reward it BIG and scaled by speed (a faster lap pays more), so it
        # learns to optimise the WHOLE lap (line, braking, corners), not just instantaneous progress.
        self.lap_steps += 1
        self.lap_dist += fwd
        lapped = self.lap_dist >= self.track.length
        if lapped.any():
            lt = self.lap_steps[lapped] * DT                  # lap time (s)
            self.last_lap_times = lt.copy()
            reward[lapped] += np.clip(8.0 * (LAP_TARGET_S / np.maximum(lt, 1.0)), 2.0, 20.0)
            self.lap_dist[lapped] -= self.track.length        # carry remainder into the next lap
            self.lap_steps[lapped] = 0
        else:
            self.last_lap_times = np.zeros(0)
        wall_term = nearest_wall < CRASH_MARGIN               # the car BODY contacts a wall
        off_term = off > MAX_OFF
        done = wall_term | off_term
        reward[done] -= RW_CRASH                           # a crash genuinely costs (vs the old -0.3)
        self.n_wall = int(wall_term.sum()); self.n_off = int(off_term.sum())
        dmask = np.where(done)[0]
        self.crash_pos = prev[dmask][:120].copy() if len(dmask) else np.zeros((0, 2))
        obs = self._obs(rays)
        if done.any():
            self.reset(dmask)
            r2 = self.track.raycast(self.pos, self.head, self.rc, self.rs, self.cy)
            obs = self._obs(r2)                            # fresh obs for respawned cars
        return obs, reward, done


if __name__ == "__main__":
    print("building sim...")
    sim = VecCarSim(n=256)
    print("track: %d centreline pts | obs_dim=%d (dyn=%d rays=%d) | %d cars"
          % (sim.track.K, sim.obs_dim, sim.n_dyn, sim.n_rays, sim.n))
    obs = sim.reset_all()
    t0 = time.time(); steps = 400
    for _ in range(steps):
        act = np.random.uniform(-1, 1, (sim.n, 2))
        obs, r, done = sim.step(act)
    dt = time.time() - t0
    print("ran %d steps x %d cars = %d transitions in %.2fs  ->  %d transitions/sec"
          % (steps, sim.n, steps * sim.n, dt, int(steps * sim.n / dt)))
    print("obs shape:", obs.shape, "finite:", bool(np.isfinite(obs).all()))
    print("speed  mean %.1f max %.1f m/s" % (sim.spd.mean(), sim.spd.max()))
    print("latG   mean %.1f max %.1f m/s^2" % (np.abs(sim.latG).mean(), np.abs(sim.latG).max()))
    print("grip   mean %.2f max %.2f" % (sim.grip.mean(), sim.grip.max()))
    print("slip_b mean %.3f max %.3f rad" % (np.abs(sim.slip_b).mean(), np.abs(sim.slip_b).max()))
