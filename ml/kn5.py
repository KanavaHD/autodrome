"""
kn5 reader — parse Assetto Corsa's binary track/car mesh format to get GEOMETRY + MESH NAMES, so we
can voxelise the REAL track (road, grass, kerbs, and the actual 1WALL barriers) instead of faking
walls from the AI line's side-distance (which put phantom walls along open straights).

We only need per-mesh: name, material name, world-space triangle positions. We SKIP texture blobs
(most of the file's bytes) and shader-property values — we just walk past them to reach the node tree.

kn5 layout (versions 5/6, the AC stock format):
  magic "sc6969" | int version | [int extra if v>=6]
  textures:  int N; N×( int type, str name, int size, byte[size] )            # sizes skipped
  materials: int N; N×( str name, str shader, byte blend, byte alphaTested,    # v>4
                        int depth, int props; props×(str name, float v, float[4] padding),
                        int texSlots; slots×(str sampler, int slot, str texName) )
  nodes (recursive from root):
     int type(1=dummy,2=mesh,3=skinnedmesh) | str name | int childCount | bool active
     type1: float[16] localMatrix; then children
     type2: bool castShadows,bool visible,bool transparent; int verts;
            verts×(float3 pos,float3 normal,float2 uv,float3 tangent);
            int idx; idx×uint16; int materialID; ... bbox/lod tail ; then children
     type3: like 2 but with a bone table before the vertices

    python ml/kn5.py <file.kn5>          # summary: meshes, triangles, name samples, world bounds
"""
import os, sys, struct
import numpy as np


class Reader:
    def __init__(self, buf):
        self.b = buf; self.p = 0; self.n = len(buf)

    def u8(self):
        v = self.b[self.p]; self.p += 1; return v

    def i32(self):
        v = struct.unpack_from("<i", self.b, self.p)[0]; self.p += 4; return v

    def u32(self):
        v = struct.unpack_from("<I", self.b, self.p)[0]; self.p += 4; return v

    def f32(self):
        v = struct.unpack_from("<f", self.b, self.p)[0]; self.p += 4; return v

    def skip(self, k):
        self.p += k

    def string(self):
        ln = self.u32()
        if ln > 1_000_000 or self.p + ln > self.n:
            raise ValueError("bad string length %d at %d" % (ln, self.p - 4))
        s = self.b[self.p:self.p + ln].decode("utf-8", "replace"); self.p += ln
        return s


class Mesh:
    __slots__ = ("name", "material", "verts", "tris")

    def __init__(self, name, material, verts, tris):
        self.name = name          # node name (often carries the surface tag, e.g. "1ROAD_...")
        self.material = material   # material name (the other place the tag lives)
        self.verts = verts        # (V,3) float32 world XYZ  (AC: x, y=up, z)
        self.tris = tris          # (T,3) int32 indices into verts


def _read_header(r):
    magic = r.b[:6]
    if magic != b"sc6969":
        raise ValueError("not a kn5 file (magic=%r)" % magic)
    r.p = 6
    ver = r.i32()
    if ver >= 6:
        r.i32()                                  # extra header int present from v6
    return ver


def _skip_textures(r):
    n = r.u32()
    for _ in range(n):
        r.i32()                                  # type
        r.string()                               # name
        size = r.u32()
        r.skip(size)                             # blob — skipped


def _skip_materials(r, ver):
    n = r.u32()
    names = []
    for _ in range(n):
        names.append(r.string())                 # material name
        r.string()                               # shader name
        if ver > 4:
            r.u8()                               # alpha blend mode
            r.u8()                               # alpha tested (bool)
            r.i32()                              # depth mode
        props = r.u32()
        for _ in range(props):
            r.string()                           # prop name
            r.f32()                              # scalar valueA
            r.skip(9 * 4)                         # valueB(vec2)+valueC(vec3)+valueD(vec4) — unused
        slots = r.u32()
        for _ in range(slots):
            r.string()                           # sampler name
            r.u32()                              # slot
            r.string()                           # texture name
    return names


def _read_node(r, ver, matnames, out, mats):
    ntype = r.i32()
    name = r.string()
    children = r.i32()
    r.u8()                                       # active flag
    if ntype == 1:                               # dummy / transform node
        r.skip(16 * 4)                           # local 4x4 matrix
    elif ntype in (2, 3):                        # mesh / skinned mesh
        r.u8(); r.u8(); r.u8()                    # castShadows, visible, transparent
        if ntype == 3:                           # skinned: bone table precedes the vertices
            nb = r.u32()
            for _ in range(nb):
                r.string()                       # bone name
                r.skip(16 * 4)                   # inverse-bind matrix
        vcount = r.u32()
        # vertex = pos3 + normal3 + uv2 + tangent3 = 11 floats (44 bytes)
        stride = 11
        raw = np.frombuffer(r.b, dtype="<f4", count=vcount * stride, offset=r.p).reshape(vcount, stride)
        r.skip(vcount * stride * 4)
        verts = raw[:, 0:3].astype(np.float32)
        icount = r.u32()
        idx = np.frombuffer(r.b, dtype="<u2", count=icount, offset=r.p).astype(np.int32)
        r.skip(icount * 2)
        matid = r.u32()
        r.skip(4)                                # layer
        r.f32(); r.f32()                          # lodIn, lodOut
        r.skip(3 * 4)                             # bounding-sphere centre
        r.f32()                                  # bounding-sphere radius
        r.u8()                                   # isRenderable (1-byte flag) — end of mesh node
        tris = idx[:(len(idx) // 3) * 3].reshape(-1, 3)
        material = matnames[matid] if 0 <= matid < len(matnames) else ""
        out.append(Mesh(name, material, verts, tris))
        mats.append(material)
    else:
        raise ValueError("unknown node type %d for %r at %d" % (ntype, name, r.p))
    for _ in range(children):
        _read_node(r, ver, matnames, out, mats)


def load(path, verbose=False):
    """Parse a .kn5 -> list[Mesh] with world-space triangle geometry and names."""
    buf = open(path, "rb").read()
    r = Reader(buf)
    ver = _read_header(r)
    _skip_textures(r)
    matnames = _skip_materials(r, ver)
    meshes = []; mats = []
    _read_node(r, ver, matnames, meshes, mats)
    if verbose:
        print("kn5 v%d  %.0f MB  -> %d materials, %d meshes" % (ver, len(buf) / 1e6, len(matnames), len(meshes)))
    return meshes


def _summary(path):
    meshes = load(path, verbose=True)
    tot_t = sum(len(m.tris) for m in meshes); tot_v = sum(len(m.verts) for m in meshes)
    allv = np.vstack([m.verts for m in meshes]) if meshes else np.zeros((1, 3), np.float32)
    lo = allv.min(0); hi = allv.max(0)
    print("  %d triangles, %d vertices" % (tot_t, tot_v))
    print("  world bounds X[%.1f..%.1f] Y[%.1f..%.1f] Z[%.1f..%.1f]" %
          (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    # sample some mesh / material names so we can see the surface tags (1ROAD, 1WALL, grass, kerb...)
    print("  --- sample mesh names ---")
    for m in meshes[:25]:
        print("    %-34s mat=%-24s tris=%d" % (m.name[:34], m.material[:24], len(m.tris)))
    # tally tag-ish keywords
    import re
    tags = {}
    for m in meshes:
        key = (m.name + " " + m.material).upper()
        for kw in ("ROAD", "WALL", "GRASS", "KERB", "CURB", "SAND", "GRAVEL", "CONCRETE", "PIT", "ASPH"):
            if kw in key:
                tags[kw] = tags.get(kw, 0) + 1
    print("  tag keyword counts:", tags)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python ml/kn5.py <file.kn5>"); raise SystemExit(1)
    _summary(sys.argv[1])
