"""
TrackIngestor — turn ANY Assetto Corsa track into a sim-ready centreline + walls, with NO mesh
(.kn5) parsing. AC ships every track's racing line in `ai/fast_lane.ai`, and crucially each point
carries `sideLeft`/`sideRight` — the distance to the left/right track edge. So:

    walls = centreline  ±  side  (along the lateral normal)

That's the whole drivable corridor, which is all the LIDAR sim needs — we never touch 1WALL/1GRASS
meshes in the kn5. (surfaces.ini could later distinguish wall-vs-grass penalties; not needed for v1.)

    python ml/track_ingest.py imola              # -> ml/tracks/imola.npz
    python ml/track_ingest.py ks_brands_hatch gp # a layout

The .npz holds the resampled centreline, per-point clearance, and compact wall EDGE points (with
height); Track rasterises the occupancy grid from those edges at load. car_sim.Track(geom=...)
consumes it.
"""
import os, sys, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AC_ROOT = os.environ.get("AC_ROOT", r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa")
OUT_DIR = os.path.join(HERE, "tracks")


def parse_fast_lane(track, layout=""):
    """Read fast_lane.ai (v7): 16-byte header, count×20-byte points [posX,posY,posZ,len,id],
    then a detail block of count×18 floats where [5]=sideLeft, [6]=sideRight. Returns world XZ
    centre (the AI line), height Y, and the side distances."""
    base = os.path.join(AC_ROOT, "content", "tracks", track)
    f = os.path.join(base, layout, "ai", "fast_lane.ai") if layout else os.path.join(base, "ai", "fast_lane.ai")
    raw = open(f, "rb").read()
    ver, count = struct.unpack_from("<ii", raw, 0)
    if ver != 7:
        # other versions exist but are rare; the point block still starts at 16 in practice
        pass
    P = np.frombuffer(raw, dtype="<f4", count=count * 5, offset=16).reshape(count, 5)
    pos = P[:, 0:3].astype(np.float64)                       # X, Y(up), Z
    det_off = 16 + count * 20
    det_count = struct.unpack_from("<i", raw, det_off)[0]
    D = np.frombuffer(raw, dtype="<f4", count=det_count * 18, offset=det_off + 4).reshape(det_count, 18)
    sideL = D[:, 5].astype(np.float64); sideR = D[:, 6].astype(np.float64)
    if det_count != count:
        raise ValueError("point/detail count mismatch (%d vs %d)" % (count, det_count))
    return pos[:, 0], pos[:, 2], pos[:, 1], sideL, sideR     # x, z, y, sideL, sideR


def _resample_closed(x, z, y, sl, sr, spacing):
    """Resample the closed loop to ~uniform `spacing` (m). Returns K and the resampled arrays."""
    cx = np.r_[x, x[:1]]; cz = np.r_[z, z[:1]]
    seg = np.r_[0, np.cumsum(np.hypot(np.diff(cx), np.diff(cz)))]
    total = seg[-1]
    K = max(200, int(round(total / spacing)))
    t = np.linspace(0, total, K, endpoint=False)
    wrap = lambda a: np.interp(t, seg, np.r_[a, a[:1]])
    return K, wrap(x), wrap(z), wrap(y), wrap(sl), wrap(sr), float(total)


def _densify(points, step):
    """Insert points so consecutive samples are <= step apart (closed loop) — a solid wall for raster."""
    out = []
    n = len(points)
    for i in range(n):
        a = points[i]; b = points[(i + 1) % n]
        d = np.hypot(*(b[:2] - a[:2]))
        m = max(1, int(np.ceil(d / step)))
        for k in range(m):
            out.append(a + (b - a) * (k / m))
    return np.array(out)


def build(track, layout="", spacing=2.0, cell=0.5, clamp_side=14.0):
    x, z, y, sl, sr = parse_fast_lane(track, layout)
    sl = np.clip(sl, 0.5, clamp_side); sr = np.clip(sr, 0.5, clamp_side)
    K, x, z, y, sl, sr, length = _resample_closed(x, z, y, sl, sr, spacing)
    center = np.stack([x, z], 1)
    # unit tangent + left normal
    TG = np.roll(center, -1, 0) - center
    TG /= (np.linalg.norm(TG, axis=1, keepdims=True) + 1e-9)
    N = np.stack([-TG[:, 1], TG[:, 0]], 1)                   # left-hand normal in XZ
    # recentre the AI racing line to the geometric middle of the corridor; symmetric clearance
    half = (sl + sr) * 0.5
    center = center + N * ((sl - sr) * 0.5)[:, None]
    left = center + N * half[:, None]
    right = center - N * half[:, None]
    cumlen = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.r_[center, center[:1]], axis=0), axis=1))][:K]
    # compact wall edges with height (densified) for the occupancy grid
    le = _densify(np.c_[left, y], cell * 0.5)
    re = _densify(np.c_[right, y], cell * 0.5)
    edges = np.vstack([le, re]).astype(np.float32)           # (M, 3): x, z, y
    os.makedirs(OUT_DIR, exist_ok=True)
    key = track if not layout else "%s__%s" % (track, layout)
    out = os.path.join(OUT_DIR, key + ".npz")
    np.savez_compressed(out, center=center.astype(np.float32), center_y=y.astype(np.float32),
                        tangent=TG.astype(np.float32), halfwidth=half.astype(np.float32),
                        cumlen=cumlen.astype(np.float32), edges=edges, cell=np.float32(cell),
                        length=np.float32(length), name=np.array(key))
    return out, dict(K=K, length=length, edges=len(edges), half_med=float(np.median(half)))


def load_geom(key):
    """Load an ingested track into the dict shape car_sim.Track(geom=...) expects."""
    p = key if key.endswith(".npz") else os.path.join(OUT_DIR, key + ".npz")
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python ml/track_ingest.py <track> [layout]"); raise SystemExit(1)
    track = sys.argv[1]; layout = sys.argv[2] if len(sys.argv) > 2 else ""
    out, info = build(track, layout)
    print("ingested %s -> %s" % (track + ("/" + layout if layout else ""), out))
    print("  K=%d centreline pts | %.0f m | %d wall-edge pts | median half-width %.1f m"
          % (info["K"], info["length"], info["edges"], info["half_med"]))
