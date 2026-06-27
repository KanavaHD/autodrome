"""
AC process memory — read/write the car's control inputs DIRECTLY (no vJoy, no virtual gamepad),
the way the built-in AI does. This locates the writable gas/brake/steer floats in acs.exe by
correlating a value scan against the shared-memory ground truth, then writes our policy's outputs.

Pure ctypes (no pymem). acs.exe is 64-bit; we scan the private read-write heap (~2.6 GB) and pin the
car's physics struct by narrowing a value scan against the shared-memory ground truth.

    python ml/ac_memory.py find 40        # AI driving: narrows the car-struct anchor (speed in m/s)
    python ml/ac_memory.py read 0x<addr>  # read a float
    python ml/ac_memory.py write 0x<addr> <val>   # write a 4-byte float

SAFETY: single-player, your own car, offline research — the standard AC modding/telemetry use case.
"""
import ctypes as C
import ctypes.wintypes as W
import sys, time, struct
import numpy as np

k32 = C.WinDLL("kernel32", use_last_error=True)

PROCESS_VM_READ = 0x0010
PROCESS_VM_WRITE = 0x0020
PROCESS_VM_OPERATION = 0x0008
PROCESS_QUERY_INFORMATION = 0x0400
MEM_COMMIT = 0x1000
PAGE_READWRITE = 0x04
PAGE_WRITECOPY = 0x08
PAGE_EXECUTE_READWRITE = 0x40
PAGE_GUARD = 0x100
WRITABLE = (PAGE_READWRITE,)   # private RW heap — where the game's car physics state lives (64-bit acs.exe)


class MBI(C.Structure):
    _fields_ = [("BaseAddress", C.c_ulonglong), ("AllocationBase", C.c_ulonglong),
                ("AllocationProtect", W.DWORD), ("__a1", W.DWORD),
                ("RegionSize", C.c_ulonglong), ("State", W.DWORD),
                ("Protect", W.DWORD), ("Type", W.DWORD), ("__a2", W.DWORD)]


k32.OpenProcess.restype = W.HANDLE
k32.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
k32.VirtualQueryEx.restype = C.c_size_t
k32.VirtualQueryEx.argtypes = [W.HANDLE, C.c_ulonglong, C.POINTER(MBI), C.c_size_t]
k32.ReadProcessMemory.argtypes = [W.HANDLE, C.c_ulonglong, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
k32.WriteProcessMemory.argtypes = [W.HANDLE, C.c_ulonglong, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]


def find_pid(name="acs.exe"):
    import subprocess
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq %s" % name, "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if name.lower() in line.lower():
            return int(line.split(",")[1].strip('"'))
    return None


class AcMemory:
    def __init__(self):
        self.pid = find_pid()
        self.h = None
        if self.pid:
            self.h = k32.OpenProcess(PROCESS_VM_READ | PROCESS_VM_WRITE | PROCESS_VM_OPERATION
                                     | PROCESS_QUERY_INFORMATION, False, self.pid)

    @property
    def ok(self):
        return bool(self.h)

    def regions(self, max_region=64 * 1024 * 1024):
        """Writable committed regions across the full 64-bit space (acs.exe is 64-bit). The control
        input lives in writable heap; we skip read-only, guard pages, and huge (texture) regions."""
        addr = 0; mbi = MBI(); out = []
        while k32.VirtualQueryEx(self.h, addr, C.byref(mbi), C.sizeof(mbi)):
            nxt = mbi.BaseAddress + mbi.RegionSize
            if nxt <= addr:                # no progress -> done
                break
            if (mbi.State == MEM_COMMIT and mbi.Protect in WRITABLE
                    and not (mbi.Protect & PAGE_GUARD) and mbi.RegionSize <= max_region):
                out.append((mbi.BaseAddress, mbi.RegionSize))
            addr = nxt
        return out

    def read(self, addr, size):
        buf = (C.c_char * size)(); n = C.c_size_t(0)
        if k32.ReadProcessMemory(self.h, addr, buf, size, C.byref(n)):
            return buf.raw[:n.value]
        return b""

    def read_float(self, addr):
        b = self.read(addr, 4)
        return struct.unpack("<f", b)[0] if len(b) == 4 else None

    def write_float(self, addr, val):
        data = struct.pack("<f", float(val)); n = C.c_size_t(0)
        return bool(k32.WriteProcessMemory(self.h, addr, data, 4, C.byref(n)))

    def scan_float(self, target, eps=1e-4, max_hits=400000, regions=None):
        """All 4-byte-aligned addresses in writable memory whose float ≈ target. Use a DISTINCTIVE
        target (speed/position), never a common value like 0 or 1 (millions of matches)."""
        hits = []
        for base, size in (regions or self.regions()):
            data = self.read(base, size)
            if len(data) < 4:
                continue
            arr = np.frombuffer(data[:(len(data) // 4) * 4], dtype="<f4")
            idx = np.nonzero(np.abs(arr - target) <= eps)[0]
            for i in idx:
                hits.append(base + int(i) * 4)
            if len(hits) > max_hits:           # target too common — bail, caller should pick a rarer anchor
                return None
        return hits

    def narrow(self, addrs, target, eps=1e-4):
        return [a for a in addrs if (lambda v: v is not None and abs(v - target) <= eps)(self.read_float(a))]


def find_controls(seconds=30):
    """Locate the writable steer/gas/brake input floats: anchor on the car's SPEED (distinctive value),
    narrow it as the AI drives through corners, then search the bytes around that car-struct for the
    control values. Returns {name: address}. Needs the AI driving so speed + inputs vary."""
    import sys as _s; _s.path.insert(0, __import__("os").path.dirname(__file__))
    from ac_shared_memory import open_sm
    m = AcMemory(); sm = open_sm()
    if not (m.ok and sm.available):
        print("attach failed (AC running? on track?)"); return {}
    regs = m.regions()
    spd = sm.physics()["speedKmh"] / 3.6                     # heap stores velocity in m/s, not km/h
    print("anchoring on speed = %.2f m/s ... (scanning %.0f MB heap)" % (spd, sum(s for _, s in regs) / 1e6))
    cand = m.scan_float(spd, eps=0.05, regions=regs)
    if cand is None:
        print("speed too common right now — try when moving fast"); return {}
    print("  %d candidates; narrowing as the car's speed changes (drive through corners)..." % len(cand))
    t0 = time.time(); last = spd
    while time.time() - t0 < seconds and len(cand) > 3:
        time.sleep(0.2)
        v = sm.physics()["speedKmh"] / 3.6
        if abs(v - last) < 0.4:
            continue                       # only filter on a real change (more discriminating)
        last = v
        cand = [a for a in cand if (lambda x: x is not None and abs(x - v) <= 0.4)(m.read_float(a))]
        print("\r  speed=%.1f m/s -> %4d candidates  " % (v, len(cand)), end="", flush=True)
    print("\n  car speed anchor(s):", [hex(a) for a in cand[:6]])
    if not cand:
        return {}
    # the control inputs (gas/brake/steer) sit in the same car physics struct — search a window around
    # the anchor for the live SHM values of all three at once.
    p = sm.physics(); found = {}
    base = min(cand); WIN = 0x8000
    blob = m.read(base - WIN, WIN * 2)
    arr = np.frombuffer(blob[:(len(blob) // 4) * 4], dtype="<f4")
    for name, val, eps in (("gas", p["gas"], 0.01), ("brake", p["brake"], 0.01),
                           ("steer", p["steerAngle"], 0.01)):
        idx = np.nonzero(np.abs(arr - val) <= eps)[0]
        addrs = [base - WIN + int(i) * 4 for i in idx]
        found[name] = addrs
        print("  %-6s = %.3f -> %d nearby candidate(s)" % (name, val, len(addrs)))
    found["_anchor"] = cand
    return found


def read_around(anchor, span=0x400):
    """Page-safe dump of floats around a pinned address (skips unmapped pages). Used to locate the
    control inputs relative to the car struct once an anchor (velocity) is found."""
    m = AcMemory(); out = {}
    for off in range(-span, span, 4):
        v = m.read_float(anchor + off)
        if v is not None:
            out[off] = v
    return out


def find_control(get_live, label, samples=10, settle=0.25):
    """Locate the address(es) holding a control by correlating a value scan with the live shared-memory
    value over several samples while it changes (AI driving). Returns the surviving addresses."""
    m = AcMemory()
    if not m.ok:
        print("could not attach to acs.exe"); return m, []
    v0 = get_live()
    print("  scanning for %s ≈ %.4f ..." % (label, v0))
    cands = m.scan_float(v0, eps=0.003)
    print("  %d initial candidates" % len(cands))
    seen = set()
    for i in range(samples):
        time.sleep(settle)
        v = get_live()
        cands = m.narrow(cands, v, eps=0.01)
        key = round(v, 3)
        if key not in seen:
            seen.add(key)
        print("\r  sample %2d: %s=%.3f -> %5d candidates" % (i + 1, label, v, len(cands)), end="", flush=True)
        if len(cands) <= 6 and len(seen) >= 4:
            break
    print()
    return m, cands


class ACReader:
    """AC shared-memory telemetry for deploy_ac.py: combines the physics page (speed/velocity/heading/
    accG/gear) with the graphics page (status + world carCoordinates) into one frame dict.
    frame() -> {status, gear, speed(km/h), vel[3], heading(rad), pos[3], accG[3]} or None."""
    def __init__(self):
        import mmap
        try:
            from ac_shared_memory import SharedMemory
            self._phys = SharedMemory()
        except Exception:
            self._phys = None
        self._gfx = None
        for size in (4096, 2048, 1024):
            try:
                self._gfx = mmap.mmap(-1, size, "Local\\acpmf_graphics", access=mmap.ACCESS_READ)
                break
            except Exception:
                self._gfx = None
        self._gt_frame = None; self._gt_t = 0.0           # track the CSP feed's frame counter (freshness)
        self._warned = False

    # LIVE world position is published every frame by the NN Drive CSP app (car.position). The
    # graphics-page carCoordinates offset isn't stable under CSP, so we use this file instead.
    _GT = r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\apps\lua\nndrive\state.json"

    def frame(self):
        p = self._phys.physics() if self._phys else None
        if not p:
            return None
        status = 2                                        # status 2 = LIVE
        if self._gfx:
            try:
                self._gfx.seek(0); status = struct.unpack_from("<i", self._gfx.read(8), 4)[0]
            except Exception:
                pass
        # world position + FRESHNESS: if the CSP feed's frame counter stops advancing the file is
        # stale (app not running / AC paused) -> DON'T drive on dead coordinates (that makes the policy
        # localise to one frozen spot and steer blindly). Report pos_stale so the loop idles + warns.
        pos = [0.0, 0.0, 0.0]; pos_stale = True
        try:
            import json
            d = json.load(open(self._GT))
            pos = d.get("pos", pos)
            f = d.get("frame")
            now = time.time()
            if f != self._gt_frame:
                self._gt_frame = f; self._gt_t = now
            pos_stale = (now - self._gt_t) > 1.0
        except Exception:
            pos_stale = True
        if pos_stale and not self._warned:
            print("\n  [!] no live position from the NN Drive app (state.json missing/frozen).")
            print("      open the 'NN Drive' app window in AC so it publishes the car's position.\n")
            self._warned = True
        elif not pos_stale:
            self._warned = False
        return {"status": status, "gear": p.get("gear", 1), "speed": p.get("speedKmh", 0.0),
                "vel": p.get("velocity", [0.0, 0.0, 0.0]), "heading": p.get("heading", 0.0),
                "pos": pos, "accG": p.get("accG", [0.0, 0.0, 0.0]), "pos_stale": pos_stale}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "find"
    if cmd == "find":
        secs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
        res = find_controls(seconds=secs)
        print("\ncar-struct anchor(s):", [hex(a) for a in res.get("_anchor", [])][:6])
    elif cmd == "read":
        print(AcMemory().read_float(int(sys.argv[2], 16)))
    elif cmd == "write":
        ok = AcMemory().write_float(int(sys.argv[2], 16), float(sys.argv[3]))
        print("wrote" if ok else "write failed")
    else:
        print("usage: python ml/ac_memory.py [find <secs> | read <hexaddr> | write <hexaddr> <val>]")
