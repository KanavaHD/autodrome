"""
Virtual LIDAR for the AC self-driver — wall sensing from the track mesh.

Why this exists: the controller's only sense of danger was cross-track error to the
racing line (a proxy). That's why it drove off the elevated section and into pit
walls — it literally couldn't see walls. This builds a wall map from track.glb and
casts rays from the car each frame to measure real clearance in every direction.

How walls are found — BY GEOMETRY, NOT BY NAME. A wall is any near-vertical, tall-
enough surface in ANY mesh. So barriers that aren't named '*wall*' are still caught,
while flat surfaces (road top, kerbs, runoff) are skipped automatically because their
faces point up, not sideways. This was the explicit requirement: watch every mesh.

It is 2.5D: each wall cell remembers its height span, and a ray only counts a cell as
blocking if that span overlaps the CAR's current height — so bridge supports below an
elevated road, or overhead gantries, don't create phantom walls.

  python ml/wall_sensor.py            # build + print stats + save a top-down preview PNG
"""
import os, json, math
import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLB = os.path.join(ROOT, "data", "track.glb")
CALIB = os.path.join(ROOT, "data", "calib.json")

# a face counts as WALL if it is near-vertical AND tall enough to be a real barrier
VERT_NY = 0.35          # |face-normal.y| below this = a vertical (side-facing) surface
MIN_WALL_H = 0.30       # the face must span at least this much in Y (skips kerb lips / road seams)
GRID = 0.30             # occupancy cell size (m)
# height band (relative to the car) within which a wall actually blocks the car
H_BELOW = 1.2           # walls reaching from this far below the car...
H_ABOVE = 1.4           # ...to this far above it, are relevant (car is low). Kept tight so
                        # overhead structure (bridge undersides, gantries) on a CLIMB isn't
                        # mistaken for a wall ahead — that froze the driver on every hill.
# meshes that never collide with the car — excluded by name (everything else is fair game)
SKIP_NAMES = ("sky", "cloud", "light", "lamp", "sun", "flare", "fog")


CACHE = os.path.join(ROOT, "data", "wall_cache.npz")


class WallSensor:
    def __init__(self, glb=GLB, calib=CALIB, cell=GRID):
        self.cell = cell
        self.cells = {}                 # (gx,gz) -> [ymin, ymax] of wall geometry there
        if not self._load_cache(glb, calib):
            self._build(glb, calib)
            self._save_cache(glb, calib)

    def _sig(self, glb, calib):         # cache is valid only if inputs/params unchanged
        return "%.0f|%.0f|%.3f|%.2f|%.2f" % (os.path.getmtime(glb), os.path.getmtime(calib),
                                             self.cell, VERT_NY, MIN_WALL_H)

    def _load_cache(self, glb, calib):
        try:
            d = np.load(CACHE, allow_pickle=False)
            if str(d["sig"]) != self._sig(glb, calib):
                return False
            keys = d["keys"]; spans = d["spans"]
            self.cells = {(int(k[0]), int(k[1])): [float(s[0]), float(s[1])]
                          for k, s in zip(keys, spans)}
            self.n_meshes_used = int(d["n_meshes"])
            return True
        except Exception:
            return False

    def _save_cache(self, glb, calib):
        try:
            keys = np.array(list(self.cells.keys()), dtype=np.int32)
            spans = np.array(list(self.cells.values()), dtype=np.float32)
            np.savez(CACHE, sig=self._sig(glb, calib), keys=keys, spans=spans,
                     n_meshes=getattr(self, "n_meshes_used", 0))
        except Exception:
            pass

    # ---------- build the wall map (once, at startup) ----------
    def _build(self, glb, calib):
        scene = trimesh.load(glb, force="scene")
        xf = json.load(open(calib))["xf"]
        a, b, tx, tz = xf["a"], xf["b"], xf["tx"], xf["tz"]
        det = a * a + b * b

        def to_ac(x, z):                # invert the locked calibration -> AC xz frame
            dx, dz = x - tx, z - tz
            return (a * dx + b * dz) / det, (-b * dx + a * dz) / det

        self.n_meshes_used = 0
        for node in scene.graph.nodes_geometry:
            T, gname = scene.graph[node]
            if gname is None:
                continue
            nm = gname.lower()
            if any(k in nm for k in SKIP_NAMES):
                continue
            g = scene.geometry[gname]
            if not hasattr(g, "faces") or len(g.faces) == 0:
                continue
            V = trimesh.transformations.transform_points(g.vertices, T)   # world space
            tri = V[g.faces]                                              # (M,3,3)
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            ln = np.linalg.norm(n, axis=1) + 1e-12
            ny = np.abs(n[:, 1] / ln)
            ymin = tri[:, :, 1].min(axis=1)
            ymax = tri[:, :, 1].max(axis=1)
            mask = (ny < VERT_NY) & ((ymax - ymin) > MIN_WALL_H)
            if not mask.any():
                continue
            self.n_meshes_used += 1
            for t, y0, y1 in zip(tri[mask], ymin[mask], ymax[mask]):
                self._raster_tri(t, float(y0), float(y1), to_ac)

    def _mark(self, x, z, y0, y1):
        key = (int(round(x / self.cell)), int(round(z / self.cell)))
        c = self.cells.get(key)
        if c is None:
            self.cells[key] = [y0, y1]
        else:
            if y0 < c[0]: c[0] = y0
            if y1 > c[1]: c[1] = y1

    def _raster_tri(self, t, y0, y1, to_ac):
        """Mark the triangle's xz footprint. A vertical wall projects to ~a segment, so
        sampling the three edges fills its footprint line densely enough."""
        pts = [to_ac(t[i, 0], t[i, 2]) for i in range(3)]
        for i in range(3):
            p, q = pts[i], pts[(i + 1) % 3]
            d = math.hypot(q[0] - p[0], q[1] - p[1])
            steps = max(1, int(d / (self.cell * 0.5)))
            for k in range(steps + 1):
                f = k / steps
                self._mark(p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f, y0, y1)

    # ---------- query (each frame) ----------
    def _blocked(self, x, z, car_y):
        gx = int(round(x / self.cell)); gz = int(round(z / self.cell))
        lo, hi = car_y - H_BELOW, car_y + H_ABOVE
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = self.cells.get((gx + dx, gz + dz))
                if c is not None and c[1] >= lo and c[0] <= hi:
                    return True
        return False

    def ray(self, x, z, dx, dz, car_y, max_d=18.0):
        """Distance until a wall is hit along (dx,dz), or max_d if clear."""
        step = self.cell * 0.6
        d = 0.45                 # start at the car's edge (see scan_fast) — avoids self-collapse
        while d < max_d:
            if self._blocked(x + dx * d, z + dz * d, car_y):
                return d
            d += step
        return max_d

    def scan(self, pos, head, car_y, angles_deg, max_d=18.0):
        """Cast a fan of rays. angles in degrees, 0 = straight ahead, + = right, - = left.
        Returns clearance (m) per angle."""
        hx, hz = head
        out = []
        for ad in angles_deg:
            r = math.radians(ad); c, s = math.cos(r), math.sin(r)
            dx = hx * c - hz * s
            dz = hx * s + hz * c
            out.append(self.ray(pos[0], pos[1], dx, dz, car_y, max_d))
        return out

    # ---------- FAST height-aware scan (vectorised; for the 60Hz control loop) ----------
    def _build_grid(self):
        """One-time: numpy ymin/ymax grids for O(1) vectorised occupancy + a 3x3 dilation
        baked in (matches _blocked's neighbourhood), so the fast scan == the slow one."""
        keys = np.array(list(self.cells.keys()), dtype=np.int64)
        vals = np.array(list(self.cells.values()), dtype=np.float32)     # (M,2) ymin,ymax
        self._gmin = keys.min(0); g = keys.max(0) - self._gmin + 1
        self._H0, self._H1 = int(g[0]) + 2, int(g[1]) + 2                # +pad for dilation
        ymin = np.full((self._H0, self._H1), np.inf, np.float32)
        ymax = np.full((self._H0, self._H1), -np.inf, np.float32)
        gx = keys[:, 0] - self._gmin[0] + 1; gz = keys[:, 1] - self._gmin[1] + 1
        for dx in (-1, 0, 1):                                            # 3x3 dilation
            for dz in (-1, 0, 1):
                np.minimum.at(ymin, (gx + dx, gz + dz), vals[:, 0])
                np.maximum.at(ymax, (gx + dx, gz + dz), vals[:, 1])
        self._ymin = ymin; self._ymax = ymax

    def scan_fast(self, pos, head, car_y, ray_cos, ray_sin, max_d=18.0, pitch=0.0):
        """Fully-vectorised scan(): all rays AND all distance steps in one numpy indexing op.
        ray_cos/ray_sin are precomputed cos/sin of the fan angles (radians). ~50x faster.

        pitch>0 tilts the rays UPWARD: the test height rises with distance (y = car_y + pitch*d),
        so the ray climbs with a hill/ramp instead of slamming into the rising ground. A real
        (vertical) wall still blocks it; a slope does not — that's how we tell a hill from a wall."""
        if not hasattr(self, "_ymin"):
            self._build_grid()
            self._steps = None
        c = self.cell; step = c * 0.6
        # START rays at the car's EDGE (~0.45m), not its centre. Marching from the centre
        # meant that whenever the car was near/against a wall, the FIRST sample was already
        # inside that wall's dilated footprint, so EVERY ray collapsed to one step (~0.18m) —
        # even rays pointing at open track. That zeroed the model's inputs and froze the car
        # ("rays go invisible, car halts"). Starting at the edge lets away-from-wall rays
        # escape the footprint and read true clearance.
        START = 0.45
        if self._steps is None or self._steps[-1] < max_d - step:
            self._steps = np.arange(START, max_d, step)            # cached distance samples
        steps = self._steps; S = len(steps)
        hx, hz = head
        dirx = hx * ray_cos - hz * ray_sin                          # (n,)
        dirz = hx * ray_sin + hz * ray_cos
        px = pos[0] + dirx[:, None] * steps[None, :]                # (n,S)
        pz = pos[1] + dirz[:, None] * steps[None, :]
        gx = np.clip(np.round(px / c).astype(np.int64) - self._gmin[0] + 1, 0, self._H0 - 1)
        gz = np.clip(np.round(pz / c).astype(np.int64) - self._gmin[1] + 1, 0, self._H1 - 1)
        if pitch:
            yc = car_y + pitch * steps                              # (S,) ray rises with distance
            lo = (yc - H_BELOW)[None, :]; hi = (yc + H_ABOVE)[None, :]
        else:
            lo, hi = car_y - H_BELOW, car_y + H_ABOVE
        blocked = (self._ymax[gx, gz] >= lo) & (self._ymin[gx, gz] <= hi)   # (n,S)
        has = blocked.any(axis=1)
        first = np.argmax(blocked, axis=1)                          # first hit step per ray
        return np.where(has, steps[first], max_d)


# default ray fan used by the driver: dense ahead, wide to the sides
FAN = [-90, -60, -40, -25, -12, 0, 12, 25, 40, 60, 90]


def _preview():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("building wall map from %s ..." % os.path.basename(GLB))
    ws = WallSensor()
    cells = np.array(list(ws.cells.keys())) * ws.cell
    print("  meshes contributing walls: %d" % ws.n_meshes_used)
    print("  wall cells: %d  (cell=%.2fm)" % (len(ws.cells), ws.cell))

    # compare to the old name-only ("wall") capture, to prove we catch the unnamed ones
    try:
        from road_boundary import wall_cells
        old, _ = wall_cells()
        print("  name-only ('wall') cells: %d  ->  geometry test adds %d more"
              % (len(old), max(0, len(ws.cells) - len(old))))
    except Exception as e:
        print("  (name-only comparison skipped:", e, ")")

    rl = json.load(open(os.path.join(os.path.dirname(__file__), "racingline.json")))
    P = np.array(rl["points"])                      # x, y, z
    Pxz = P[:, [0, 2]]; Py = P[:, 1]

    # measure wall-to-wall width along the line (sanity: should be a sane car-track width)
    tg = np.roll(Pxz, -1, 0) - Pxz
    tg /= (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9)
    leftn = np.stack([-tg[:, 1], tg[:, 0]], axis=1)
    widths = []
    for i in range(0, len(Pxz), 4):
        l = ws.ray(Pxz[i, 0], Pxz[i, 1], leftn[i, 0], leftn[i, 1], Py[i], 12)
        r = ws.ray(Pxz[i, 0], Pxz[i, 1], -leftn[i, 0], -leftn[i, 1], Py[i], 12)
        widths.append(l + r)
    widths = np.array(widths)
    print("  wall-to-wall width along line: mean %.1fm  min %.1fm  max %.1fm"
          % (widths.mean(), widths.min(), widths.max()))

    # top-down image: walls (grey), racing line (blue), a few ray fans (orange)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(cells[:, 0], cells[:, 1], s=1, c="0.55", marker="s", linewidths=0)
    ax.plot(Pxz[:, 0], Pxz[:, 1], "-", color="royalblue", lw=1.6, label="racing line")
    for i in range(0, len(Pxz), max(1, len(Pxz) // 8)):
        head = tg[i]
        for d in ws.scan((Pxz[i, 0], Pxz[i, 1]), (head[0], head[1]), Py[i], FAN, 18):
            pass
        for ad in FAN:
            r = math.radians(ad); c, s = math.cos(r), math.sin(r)
            dx = head[0] * c - head[1] * s; dz = head[0] * s + head[1] * c
            dist = ws.ray(Pxz[i, 0], Pxz[i, 1], dx, dz, Py[i], 18)
            ax.plot([Pxz[i, 0], Pxz[i, 0] + dx * dist],
                    [Pxz[i, 1], Pxz[i, 1] + dz * dist], "-", color="darkorange", lw=0.7, alpha=0.8)
    ax.set_aspect("equal"); ax.legend(loc="upper right")
    ax.set_title("Wall sensor (geometry-based, all meshes) — grey=walls, orange=rays")
    out = os.path.join(ROOT, "data", "wall_preview.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("  saved preview -> %s" % out)
    print("Open that image and check the grey walls line up with the track edges.")


if __name__ == "__main__":
    _preview()
