"""
Extract the real drivable road boundary from track.glb, in AC coordinates.

Used to clamp the optimal racing line so it can never leave the tarmac (no walls).
The GLB road mesh is in track space; we invert the locked 2-point calibration
(data/calib.json) to bring it into the same AC frame as the laps / centre-line.
"""
import os, json
import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLB = os.path.join(ROOT, "data", "track.glb")
CALIB = os.path.join(ROOT, "data", "calib.json")
CELL = 1.0


WALL_CELL = 0.5


def _mesh_points_ac(match):
    """World-space vertices of meshes matching `match`, in AC coordinates (xz)."""
    s = trimesh.load(GLB, force="scene")
    xf = json.load(open(CALIB))["xf"]
    a, b, tx, tz = xf["a"], xf["b"], xf["tx"], xf["tz"]
    det = a * a + b * b
    out = []
    for node in s.graph.nodes_geometry:
        T, gname = s.graph[node]
        if gname is None:
            continue
        if match(gname.lower()):
            g = s.geometry[gname]
            V = trimesh.transformations.transform_points(g.vertices, T)
            dx, dz = V[:, 0] - tx, V[:, 2] - tz
            out.append(np.stack([(a*dx + b*dz)/det, (-b*dx + a*dz)/det], axis=1))
    return np.vstack(out) if out else np.zeros((0, 2))


def wall_cells(cell=WALL_CELL):
    """Occupied cells (AC space) containing wall geometry — the hard track limit."""
    pts = _mesh_points_ac(lambda nm: "wall" in nm)
    occ = set()
    for p in pts:
        occ.add((int(round(p[0] / cell)), int(round(p[1] / cell))))
    return occ, cell


def road_edges(center, left, margin=0.6, march_max=7.0):
    """
    For each station, march outward until a WALL is hit (capped short, so it can't
    jump to a far section across the infield). Where no wall is found within
    march_max, returns ±inf (no constraint there — the driven spread governs).
    Returns (lo, hi): signed lateral wall offsets (left +), minus a margin.
    """
    occ, cell = wall_cells()

    def is_wall(p):
        gx, gz = int(round(p[0] / cell)), int(round(p[1] / cell))
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (gx + dx, gz + dz) in occ:
                    return True
        return False

    K = len(center)
    lo = np.full(K, -np.inf); hi = np.full(K, np.inf)
    step = cell
    for i in range(K):
        c, n = center[i], left[i]
        d = 0.0
        while d < march_max and not is_wall(c + (d + step) * n):
            d += step
        if d < march_max:                       # wall actually found this side
            hi[i] = max(0.3, d - margin)
        d = 0.0
        while d < march_max and not is_wall(c - (d + step) * n):
            d += step
        if d < march_max:
            lo[i] = -max(0.3, d - margin)
    return lo, hi


if __name__ == "__main__":
    c = json.load(open(os.path.join(os.path.dirname(__file__), "centerline.json")))
    center = np.array(c["center"]); tang = np.array(c["tangent"])
    left = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    lo, hi = road_edges(center, left)
    w = hi - lo
    print("wall-to-wall width along lap: mean %.1f m  min %.1f  max %.1f"
          % (w.mean(), w.min(), w.max()))
