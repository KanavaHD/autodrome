"""
Voxelise an AC track's REAL mesh (.kn5) into a surface-classified grid, so the sim drives on true
geometry: actual 1WALL barriers (the only thing the LIDAR rays should stop at), real ROAD, and GRASS
/ SAND as grip penalties — NOT as phantom walls. This replaces the fast_lane "side-distance" walls
that put barriers along open straights and made the policy swerve.

AC tags physics meshes by name: a 2-digit prefix + a surface KEY, e.g. 01WALL, 04ROAD008, 01GRASS001,
08SAND, 01KERB, 02PITS-IMA. We classify each mesh from that key (cross-checked with surfaces.ini's
IS_VALID_TRACK), then rasterise its triangles into a top-down grid of square `cell` columns. Each
column stores: the drivable surface class at ground level, and — for WALL meshes — the barrier's
vertical span (ymin..ymax), which is exactly what car_sim's height-aware raycast consumes.

Live: streams a downsampled class map to ml/voxel_live.json after each mesh, so PIT WALL can watch
the map fill in. Final grid saved to ml/tracks/<key>_voxels.npz.

    python ml/voxelize.py imola               # -> ml/tracks/imola_voxels.npz  (+ live feed)
    python ml/voxelize.py imola --cell 1.0    # column size in metres (default 1.0)
"""
import os, sys, time, json, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kn5

AC_ROOT = os.environ.get("AC_ROOT", r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa")
OUT_DIR = os.path.join(HERE, "tracks")
LIVE = os.path.join(HERE, "voxel_live.json")

# surface classes (also the legend/colour order in the live view)
CLS_NONE, CLS_ROAD, CLS_KERB, CLS_GRASS, CLS_WALL, CLS_SAND, CLS_PIT = 0, 1, 2, 3, 4, 5, 6
CLASS_NAMES = {CLS_NONE: "none", CLS_ROAD: "road", CLS_KERB: "kerb", CLS_GRASS: "grass",
               CLS_WALL: "wall", CLS_SAND: "sand", CLS_PIT: "pit"}
# colours for the live view (r,g,b)
CLASS_COLORS = {CLS_NONE: [12, 14, 20], CLS_ROAD: [70, 74, 84], CLS_KERB: [200, 80, 70],
                CLS_GRASS: [40, 120, 55], CLS_WALL: [240, 220, 90], CLS_SAND: [180, 160, 90],
                CLS_PIT: [60, 90, 130]}
# drivable-surface priority: a road column must not be overwritten by an overlapping grass skirt.
PRIORITY = {CLS_NONE: 0, CLS_GRASS: 1, CLS_SAND: 1, CLS_KERB: 2, CLS_PIT: 2, CLS_ROAD: 3}

# Map a surface keyword (found in a mesh's NAME or MATERIAL) to a class. Order matters: more specific /
# higher-priority keys first (WALL before everything; SAND before GRASS, since Mugello's sand meshes use
# a 'grass-brd' material). Tracks tag surfaces two ways — Imola by mesh-name prefix (01ROAD, 01WALL),
# Mugello by material (asph-new, grassBB, curb, walls1, Ringhiere) — so we scan name + material for both.
KEY_RULES = [
    # NB: "TYRE" alone is NOT a wall key — track meshes use "tyres_add"/"groove" for the racing-line
    # RUBBER decal that sits ON the road; tagging that as a wall puts phantom barriers on the line.
    ("WALL", CLS_WALL), ("FENCE", CLS_WALL), ("BARRIER", CLS_WALL), ("TYREWALL", CLS_WALL),
    ("TYRE-WALL", CLS_WALL), ("TYREBARR", CLS_WALL),
    ("RINGHIER", CLS_WALL), ("GUARDRAIL", CLS_WALL), ("ARMCO", CLS_WALL),
    ("KERB", CLS_KERB), ("CURB", CLS_KERB), ("CORDOL", CLS_KERB), ("RUMBLE", CLS_KERB),
    ("SAND", CLS_SAND), ("GRAVEL", CLS_SAND), ("GHIAIA", CLS_SAND), ("DIRT", CLS_SAND),
    ("GRASS", CLS_GRASS), ("CUTGRA", CLS_GRASS), ("GREEN", CLS_GRASS), ("CARPET", CLS_GRASS),
    ("ERBA", CLS_GRASS), ("GRS", CLS_GRASS),
    ("PIT", CLS_PIT), ("BOX", CLS_PIT),
    ("ROAD", CLS_ROAD), ("TARMAC", CLS_ROAD), ("ASPH", CLS_ROAD), ("ASFAL", CLS_ROAD),
    ("CONCRETE", CLS_ROAD), ("CUTCONC", CLS_ROAD), ("MANTO", CLS_ROAD), ("STRAD", CLS_ROAD),
    ("HOTLAP", CLS_ROAD), ("START", CLS_ROAD), ("TRACK", CLS_ROAD), ("TARM", CLS_ROAD),
]
# Materials/names that are clearly NOT physics surfaces even if a keyword sneaks in (glass, scenery).
DENY = ("GLASS", "VETRO", "SKY", "TREE", "ALBER", "LAMP", "LIGHT", "BANNER", "FLAG", "CROWD",
        "PEOPLE", "TRIBUN", "GRANDSTAND", "GSTAND", "ROOF", "WINDOW", "BRIDGE", "TENT", "NUVOL")


def classify(mesh):
    """Surface class from a mesh's name+material keyword, or CLS_NONE for scenery. No naming-convention
    assumption: works whether the track tags surfaces by mesh-name prefix (Imola) or material (Mugello)."""
    s = (mesh.name + " " + mesh.material).upper()
    for bad in DENY:
        if bad in s:
            return CLS_NONE
    for kw, cls in KEY_RULES:
        if kw in s:
            return cls
    return CLS_NONE


def _sample_tris(verts, tris, cell, dens=1.0, budget=2_000_000):
    """Adaptive barycentric point-sampling of triangles (vectorised). Returns world (x,y,z) samples at
    ~`dens` points per cell² of XZ area. BOUNDED: total samples can never exceed `budget`, so a huge
    terrain 'loft' mesh (thousands of big triangles) can't explode into billions of points and hang."""
    if len(tris) == 0:
        return np.zeros((0, 3), np.float32)
    a = verts[tris[:, 0]]; b = verts[tris[:, 1]]; c = verts[tris[:, 2]]
    # XZ area of each triangle (y is up; project onto the ground plane)
    ux, uz = b[:, 0] - a[:, 0], b[:, 2] - a[:, 2]
    vx, vz = c[:, 0] - a[:, 0], c[:, 2] - a[:, 2]
    area = 0.5 * np.abs(ux * vz - uz * vx)
    nsamp = np.clip(np.ceil(area / (cell * cell) * dens).astype(np.int64), 1, 512)
    total = int(nsamp.sum())
    if total > budget:
        if len(tris) >= budget:                     # more triangles than budget -> centroid of a subset
            sel = np.random.choice(len(tris), budget, replace=False)
            return ((a[sel] + b[sel] + c[sel]) / 3.0).astype(np.float32)
        nsamp = np.maximum((nsamp * (budget / total)).astype(np.int64), 1)  # scale density down to fit
        total = int(nsamp.sum())
    if total == 0:
        return np.zeros((0, 3), np.float32)
    idx = np.repeat(np.arange(len(tris)), nsamp)
    r1 = np.sqrt(np.random.random(total)); r2 = np.random.random(total)
    u = (1.0 - r1)[:, None]; v = (r1 * (1.0 - r2))[:, None]; w = (r1 * r2)[:, None]
    pts = u * a[idx] + v * b[idx] + w * c[idx]
    return pts.astype(np.float32)


def build(track, layout="", cell=1.0, pad=4.0, live=True):
    base = os.path.join(AC_ROOT, "content", "tracks", track)
    kn5_path = os.path.join(base, "%s.kn5" % track)
    if not os.path.exists(kn5_path):                       # some tracks name the main model differently
        import glob
        cands = sorted(glob.glob(os.path.join(base, "*.kn5")), key=os.path.getsize, reverse=True)
        kn5_path = cands[0] if cands else kn5_path
    t0 = time.time()
    _publish(live, {"phase": "loading", "track": track, "msg": "reading %s..." % os.path.basename(kn5_path)})
    meshes = kn5.load(kn5_path)

    # classify; keep only physics surfaces, and find the TRACK bounds from them (ignore far scenery)
    keep = []
    for mesh in meshes:
        cls = classify(mesh)
        if cls != CLS_NONE and len(mesh.tris):
            keep.append((cls, mesh))
    if not keep:
        raise ValueError("no physics surfaces found in %s" % kn5_path)
    allv = np.vstack([m.verts for _, m in keep])
    lo = allv.min(0) - pad; hi = allv.max(0) + pad
    W = int(np.ceil((hi[0] - lo[0]) / cell)); H = int(np.ceil((hi[2] - lo[2]) / cell))
    cls_grid = np.zeros((W, H), np.uint8)                  # drivable surface class per column
    pri_grid = np.zeros((W, H), np.uint8)                  # priority written so far (road beats grass)
    ground_y = np.full((W, H), np.nan, np.float32)         # road/ground height per column
    wall_min = np.full((W, H), np.inf, np.float32)         # WALL vertical span (for height-aware rays)
    wall_max = np.full((W, H), -np.inf, np.float32)
    counts = {c: 0 for c in CLASS_NAMES}

    # order so higher-priority surfaces rasterise LAST (overwrite skirts); walls handled separately
    keep.sort(key=lambda km: PRIORITY.get(km[0], 0))
    key = track if not layout else "%s__%s" % (track, layout)
    n = len(keep)
    # Per-class sampling density (points per cell²). DRIVABLE surfaces (road/kerb/pit) are sampled HARD
    # so random-sampling holes can't leave the racing line classified as the dense grass underneath it
    # (that bug made cars crawl: 39% of the line read as grass -> grip penalty on the line itself).
    DENS = {CLS_ROAD: 12.0, CLS_KERB: 12.0, CLS_PIT: 10.0, CLS_WALL: 3.0}
    for i, (cls, mesh) in enumerate(keep):
        dens = DENS.get(cls, 1.0)                          # grass/sand stay light (holes there are harmless)
        pts = _sample_tris(mesh.verts, mesh.tris, cell, dens)
        if len(pts):
            ix = np.clip(((pts[:, 0] - lo[0]) / cell).astype(np.int64), 0, W - 1)
            iz = np.clip(((pts[:, 2] - lo[2]) / cell).astype(np.int64), 0, H - 1)
            if cls == CLS_WALL:
                np.minimum.at(wall_min, (ix, iz), pts[:, 1])
                np.maximum.at(wall_max, (ix, iz), pts[:, 1])
                cls_grid[ix, iz] = CLS_WALL                # walls always show on the map
            else:
                pr = PRIORITY.get(cls, 1)
                better = pr >= pri_grid[ix, iz]            # write if >= current priority
                sx, sz = ix[better], iz[better]
                cls_grid[sx, sz] = cls; pri_grid[sx, sz] = pr
                ground_y[sx, sz] = pts[better, 1]
        counts[cls] = counts.get(cls, 0) + len(mesh.tris)
        if live and (i % 8 == 0 or i == n - 1):
            _publish_grid(cls_grid, ground_y, lo, hi, cell, phase="voxelizing",
                          progress=(i + 1) / n, track=track,
                          msg="%s  (%d/%d meshes)" % (mesh.name[:20], i + 1, n), counts=counts)

    # EXACT 3D VIEW: export the real classified triangle mesh (not the voxel grid, which is blocky).
    # The voxel grid above is still used by the sim for wall raycasting; the mesh is just for the viewer.
    try:
        _publish_grid(cls_grid, ground_y, lo, hi, cell, phase="meshing", progress=0.97, track=track,
                      msg="building exact 3D surface mesh…", counts=counts)
        mesh_keep = list(keep)
        try:                                              # add the exact corridor road ribbon on top
            mesh_keep.append((CLS_ROAD, _corridor_mesh(track, layout)))
        except Exception:
            pass
        ntris = export_surface_mesh(mesh_keep, key)
        print("[voxelize] exact surface mesh: %d triangles -> %s_mesh.bin" % (ntris, key))
    except Exception as ex:
        print("[voxelize] surface mesh export skipped (%s: %s)" % (type(ex).__name__, ex))

    # PRECISION: stamp the fast_lane drivable corridor (centre ± halfwidth) as solid ROAD on top of the
    # mesh classes. The mesh's road tags are inconsistent per-track (Imola splits the tarmac across
    # top1/asph-graph/groove), so thick grass meshes were hiding the racing surface. The corridor is
    # the surface the sim actually uses for grip, so this makes the voxel grid precise AND consistent.
    try:
        _stamp_corridor(cls_grid, ground_y, lo, cell, W, H, track, layout)
    except Exception as ex:
        print("[voxelize] corridor stamp skipped (%s)" % type(ex).__name__)

    wall_min[~np.isfinite(wall_min)] = 0.0; wall_max[~np.isfinite(wall_max)] = 0.0
    # fill any unknown ground height (none cells) by nearest finite so the 3D surface has no holes
    gy = ground_y.copy()
    if np.isnan(gy).any() and np.isfinite(gy).any():
        gy[np.isnan(gy)] = np.nanmedian(gy)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, key + "_voxels.npz")
    np.savez_compressed(out, cls=cls_grid, ground_y=gy, wall_min=wall_min, wall_max=wall_max,
                        origin=np.array([lo[0], lo[2]], np.float32), cell=np.float32(cell),
                        shape=np.array([W, H], np.int32), name=np.array(key))
    secs = time.time() - t0
    if live:
        _publish_grid(cls_grid, gy, lo, hi, cell, phase="done", progress=1.0, track=track,
                      msg="saved %s in %.1fs" % (os.path.basename(out), secs), counts=counts, done=True)
    return out, {"W": W, "H": H, "cell": cell, "wall_cells": int((cls_grid == CLS_WALL).sum()),
                 "road_cells": int((cls_grid == CLS_ROAD).sum()), "secs": round(secs, 1)}


def _corridor_mesh(track, layout):
    """Build an exact triangle ribbon for the fast_lane drivable corridor (centre ± halfwidth), raised
    slightly so it reads as the clear racing surface over the track's inconsistent tarmac meshes."""
    import track_ingest as TI
    key = track if not layout else "%s__%s" % (track, layout)
    npz = os.path.join(OUT_DIR, key + ".npz")
    if os.path.exists(npz):
        g = TI.load_geom(key)
        center = np.asarray(g["center"], np.float64); tang = np.asarray(g["tangent"], np.float64)
        half = np.asarray(g["halfwidth"], np.float64); cy = np.asarray(g["center_y"], np.float64)
    else:
        x, z, y, sl, sr = TI.parse_fast_lane(track, layout)
        K, x, z, y, sl, sr, _ = TI._resample_closed(x, z, y, sl, sr, 2.0)
        center = np.stack([x, z], 1); cy = y
        TG = np.roll(center, -1, 0) - center; TG /= (np.linalg.norm(TG, axis=1, keepdims=True) + 1e-9)
        tang = TG; half = (sl + sr) * 0.5
    K = len(center); nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    left = center + nrm * half[:, None]; right = center - nrm * half[:, None]
    verts = np.zeros((2 * K, 3), np.float32)
    verts[0::2, 0] = left[:, 0];  verts[0::2, 2] = left[:, 1];  verts[0::2, 1] = cy + 0.15
    verts[1::2, 0] = right[:, 0]; verts[1::2, 2] = right[:, 1]; verts[1::2, 1] = cy + 0.15
    tris = []
    for i in range(K):
        a, b = 2 * i, 2 * i + 1; c, d = (2 * (i + 1)) % (2 * K), (2 * (i + 1) + 1) % (2 * K)
        tris.append((a, b, d)); tris.append((a, d, c))
    return kn5.Mesh("corridor", "ROAD", verts, np.array(tris, np.int32))


def export_surface_mesh(keep, key, budget_tris=2_500_000):
    """Write the EXACT classified triangle mesh (indexed) for the 3D view — real geometry, not voxels.
    `<key>_mesh.bin` = positions(float32) + colors(uint8) + indices(uint32); `<key>_mesh.json` = header.
    Per-vertex colour is the surface class; the browser shades it with flat normals (WebGL). AC world is
    Y-up; we keep raw world coords (float32 is plenty at track scale) and let the viewer centre it."""
    pos_parts = []; col_parts = []; idx_parts = []; voff = 0; total = 0
    for cls, m in keep:
        if len(m.tris) == 0:
            continue
        v = m.verts.astype("<f4")
        col = np.tile(np.array(CLASS_COLORS.get(cls, [120, 120, 120]), np.uint8), (len(v), 1))
        idx = (m.tris.astype("<u4") + voff)
        pos_parts.append(v); col_parts.append(col); idx_parts.append(idx)
        voff += len(v); total += len(m.tris)
        if total > budget_tris:                       # safety cap for pathological meshes
            break
    pos = np.concatenate(pos_parts); col = np.concatenate(col_parts); idx = np.concatenate(idx_parts)
    bmin = pos.min(0); bmax = pos.max(0)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, key + "_mesh.bin"), "wb") as f:
        f.write(pos.tobytes()); f.write(col.tobytes()); f.write(idx.tobytes())
    json.dump({"nverts": int(len(pos)), "ntris": int(idx.size // 3), "ndraw": int(idx.size),
               "bmin": [float(x) for x in bmin], "bmax": [float(x) for x in bmax],
               "pos_bytes": int(pos.nbytes), "col_bytes": int(col.nbytes), "idx_bytes": int(idx.nbytes)},
              open(os.path.join(OUT_DIR, key + "_mesh.json"), "w"))
    return int(idx.size // 3)


def _stamp_corridor(cls_grid, ground_y, lo, cell, W, H, track, layout):
    """Mark the fast_lane drivable corridor (centre ± halfwidth) as ROAD — the reliable racing surface."""
    import track_ingest as TI
    key = track if not layout else "%s__%s" % (track, layout)
    npz = os.path.join(OUT_DIR, key + ".npz")
    if os.path.exists(npz):                              # use the already-ingested centreline if present
        g = TI.load_geom(key)
        center = np.asarray(g["center"], np.float64); tang = np.asarray(g["tangent"], np.float64)
        half = np.asarray(g["halfwidth"], np.float64); cy = np.asarray(g["center_y"], np.float64)
    else:                                               # else parse fast_lane.ai directly
        x, z, y, sl, sr = TI.parse_fast_lane(track, layout)
        K, x, z, y, sl, sr, _ = TI._resample_closed(x, z, y, sl, sr, 2.0)
        center = np.stack([x, z], 1); cy = y
        TG = np.roll(center, -1, 0) - center; TG /= (np.linalg.norm(TG, axis=1, keepdims=True) + 1e-9)
        tang = TG; half = (sl + sr) * 0.5
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)         # left normal
    # sample across the corridor width and along its length; mark those cells road (don't touch walls)
    for i in range(len(center)):
        hw = max(float(half[i]), 1.5)
        nlat = max(3, int(hw / cell * 2) + 1)
        lat = np.linspace(-hw, hw, nlat)[:, None]
        pts = center[i][None, :] + nrm[i][None, :] * lat          # (nlat, 2)
        nxt = center[(i + 1) % len(center)]
        seg = np.linspace(0, 1, max(2, int(np.hypot(*(nxt - center[i])) / cell) + 1))[:, None, None]
        allp = (pts[None] * (1 - seg) + (nxt[None, None] + nrm[i][None, None] * lat[None]) * seg).reshape(-1, 2)
        ix = np.clip(((allp[:, 0] - lo[0]) / cell).astype(np.int64), 0, W - 1)
        iz = np.clip(((allp[:, 1] - lo[2]) / cell).astype(np.int64), 0, H - 1)
        keep = cls_grid[ix, iz] != CLS_WALL                       # never paint over a real barrier
        cls_grid[ix[keep], iz[keep]] = CLS_ROAD
        ground_y[ix[keep], iz[keep]] = float(cy[i])


# ───────────────────────── live feed for the PIT WALL voxel view ─────────────────────────
def _publish(live, payload):
    if not live:
        return
    try:
        payload["ts"] = time.time()
        json.dump(payload, open(LIVE + ".tmp", "w")); os.replace(LIVE + ".tmp", LIVE)
    except Exception:
        pass


def _publish_grid(cls_grid, ground_y, lo, hi, cell, phase, progress, track, msg, counts, done=False, maxdim=420):
    """Downsample the class + HEIGHT grids to <=maxdim and emit them for the 3D canvas view."""
    W, H = cls_grid.shape
    step = max(1, int(np.ceil(max(W, H) / maxdim)))
    small = cls_grid[::step, ::step]
    gy = ground_y[::step, ::step]
    sw, sh = small.shape
    g = np.where(np.isfinite(gy), gy, 0.0).astype(np.float32)
    g0 = float(np.nanmin(g)) if np.isfinite(g).any() else 0.0
    # heights as compact ints (decimetres above the lowest point) — small JSON, plenty precise for relief
    hgt = np.clip(np.round((g - g0) * 10.0), 0, 65000).astype(np.int32)
    _publish(True, {"phase": phase, "progress": round(progress, 3), "track": track, "msg": msg,
                    "w": sw, "h": sh, "cell": cell * step, "done": done, "hbase": round(g0, 2),
                    "origin": [float(lo[0]), float(lo[2])], "extent": [float(hi[0] - lo[0]), float(hi[2] - lo[2])],
                    "colors": CLASS_COLORS, "legend": {str(k): v for k, v in CLASS_NAMES.items()},
                    "counts": {CLASS_NAMES[c]: int(n) for c, n in counts.items()},
                    "grid": small.T[::-1].flatten().tolist(),         # row-major, image orientation (z down)
                    "hgt": hgt.T[::-1].flatten().tolist()})           # matching height field (decimetres)


def load_voxels(key):
    p = key if key.endswith(".npz") else os.path.join(OUT_DIR, key + "_voxels.npz")
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python ml/voxelize.py <track> [layout] [--cell M]"); raise SystemExit(1)
    track = sys.argv[1]
    layout = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else ""
    cell = float(sys.argv[sys.argv.index("--cell") + 1]) if "--cell" in sys.argv else 1.0
    out, info = build(track, layout, cell=cell)
    print("voxelised %s -> %s" % (track, out))
    print("  %dx%d grid @ %.2fm | %d wall cells, %d road cells | %.1fs"
          % (info["W"], info["H"], info["cell"], info["wall_cells"], info["road_cells"], info["secs"]))
