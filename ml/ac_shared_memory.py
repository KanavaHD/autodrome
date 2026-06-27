"""
Assetto Corsa shared-memory reader — the REAL per-wheel ground truth (slip, load, tyre temps,
suspension travel, G-forces, slip angles) that the CSP raycast feed doesn't expose. We log this
while the BUILT-IN AI drives, then fit our sim's physics to match it (system identification).

AC publishes three memory-mapped pages on Windows: acpmf_physics (dynamics, all 4-byte fields, no
strings -> easy + reliable), acpmf_graphics (lap/spline, has wchar strings), acpmf_static (car/track,
maxRpm, tyre radius). We read physics + static; lap/spline we already get from the CSP feed.

    python ml/ac_shared_memory.py        # live dump (AC must be on track)
"""
import mmap, struct, sys, time

# physics page: ordered (name, type, count). Every field is 4 bytes (i/f) so there is no padding —
# the layout is just the concatenation. Matches AC's SPageFilePhysics through localVelocity.
_PHYS = [
    ("packetId", "i", 1), ("gas", "f", 1), ("brake", "f", 1), ("fuel", "f", 1),
    ("gear", "i", 1), ("rpms", "i", 1), ("steerAngle", "f", 1), ("speedKmh", "f", 1),
    ("velocity", "f", 3), ("accG", "f", 3),
    ("wheelSlip", "f", 4), ("wheelLoad", "f", 4), ("wheelsPressure", "f", 4),
    ("wheelAngularSpeed", "f", 4), ("tyreWear", "f", 4), ("tyreDirtyLevel", "f", 4),
    ("tyreCoreTemperature", "f", 4), ("camberRAD", "f", 4), ("suspensionTravel", "f", 4),
    ("drs", "f", 1), ("tc", "f", 1), ("heading", "f", 1), ("pitch", "f", 1), ("roll", "f", 1),
    ("cgHeight", "f", 1), ("carDamage", "f", 5), ("numberOfTyresOut", "i", 1),
    ("pitLimiterOn", "i", 1), ("abs", "f", 1), ("kersCharge", "f", 1), ("kersInput", "f", 1),
    ("autoShifterOn", "i", 1), ("rideHeight", "f", 2), ("turboBoost", "f", 1), ("ballast", "f", 1),
    ("airDensity", "f", 1), ("airTemp", "f", 1), ("roadTemp", "f", 1), ("localAngularVel", "f", 3),
    ("finalFF", "f", 1), ("performanceMeter", "f", 1), ("engineBrake", "i", 1),
    ("ersRecoveryLevel", "i", 1), ("ersPowerLevel", "i", 1), ("ersHeatCharging", "i", 1),
    ("ersIsCharging", "i", 1), ("kersCurrentKJ", "f", 1), ("drsAvailable", "i", 1),
    ("drsEnabled", "i", 1), ("brakeTemp", "f", 4), ("clutch", "f", 1),
    ("tyreTempI", "f", 4), ("tyreTempM", "f", 4), ("tyreTempO", "f", 4), ("isAIControlled", "i", 1),
    ("tyreContactPoint", "f", 12), ("tyreContactNormal", "f", 12), ("tyreContactHeading", "f", 12),
    ("brakeBias", "f", 1), ("localVelocity", "f", 3),
    # extended — per-wheel tyre forces + slips (the system-ID gold). Order matches AC's struct.
    ("P2PActivations", "i", 1), ("P2PStatus", "i", 1), ("currentMaxRpm", "f", 1),
    ("mz", "f", 4), ("fx", "f", 4), ("fy", "f", 4),
    ("slipRatio", "f", 4), ("slipAngle", "f", 4),
]
_PHYS_FMT = "<" + "".join(t * c for _, t, c in _PHYS)
_PHYS_SIZE = struct.calcsize(_PHYS_FMT)


def _unpack(blob, spec, fmt):
    vals = struct.unpack_from(fmt, blob, 0)
    out, i = {}, 0
    for name, _, c in spec:
        out[name] = vals[i] if c == 1 else list(vals[i:i + c])
        i += c
    return out


class SharedMemory:
    """Opens AC's memory-mapped physics page; .physics() returns a fresh dict each call."""
    def __init__(self):
        self.phys = None
        # AC over-allocates the page; request a size we KNOW fits (try big -> small until one attaches).
        for size in (2048, 1600, 1024, _PHYS_SIZE):
            try:
                self.phys = mmap.mmap(-1, size, "Local\\acpmf_physics", access=mmap.ACCESS_READ)
                break
            except Exception:
                self.phys = None

    @property
    def available(self):
        return self.phys is not None

    def physics(self):
        if not self.phys:
            return None
        try:
            self.phys.seek(0)
            blob = self.phys.read(_PHYS_SIZE)
            return _unpack(blob, _PHYS, _PHYS_FMT)
        except Exception:
            return None

    def close(self):
        if self.phys:
            self.phys.close(); self.phys = None


def open_sm():
    return SharedMemory()


if __name__ == "__main__":
    sm = open_sm()
    if not sm.available:
        print("AC shared memory not found — start AC and get on track."); sys.exit(1)
    print("reading acpmf_physics (struct %d bytes). Ctrl+C to stop.\n" % _PHYS_SIZE)
    try:
        while True:
            p = sm.physics()
            if p:
                print("\rspeed %6.1f km/h | rpm %5d | gear %d | gas %.2f brk %.2f steer %+6.1f° | "
                      "AI=%d | slip %s | accG %s" % (
                          p["speedKmh"], p["rpms"], p["gear"] - 1, p["gas"], p["brake"],
                          p["steerAngle"], p["isAIControlled"],
                          ["%.2f" % s for s in p["wheelSlip"]],
                          ["%+.2f" % g for g in p["accG"]]), end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nstopped.")
