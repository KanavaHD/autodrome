"""
CarParams ingestor — get the physics of ANY Assetto Corsa car so the sim trains it faithfully.

204 of 209 stock cars ship their physics ENCRYPTED in data.acd, so we don't depend on it. Every
car instead exposes real published numbers in ui/ui_car.json `specs` (power, weight, top speed,
0-100, drivetrain) — that pins the dominant dynamics: mass, top speed, acceleration, power/weight.
Grip/geometry come from category + sane estimates, and from the real data/ INIs when a car ships
them unpacked (then we use the true mass, wheelbase, CG, tyre grip, steer lock).

    python ml/car_ingest.py bmw_m3_e92        # -> ml/cars/bmw_m3_e92.json + prints the params

Output (CarParams) is what car_sim_gpu.GPUCarSim(car=...) consumes.
"""
import os, sys, re, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AC_ROOT = os.environ.get("AC_ROOT", r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa")
OUT_DIR = os.path.join(HERE, "cars")
CAR_REF_GRIP = 1.49      # the car's effective peak mu — car grip is expressed relative to this
G = 9.81


def _num(s, default=0.0):
    if s is None:
        return default
    m = re.search(r"-?\d+(?:\.\d+)?", str(s).replace(",", ""))
    return float(m.group()) if m else default


def _ui_specs(cid):
    p = os.path.join(AC_ROOT, "content", "cars", cid, "ui", "ui_car.json")
    raw = open(p, "rb").read().decode("utf-8", "ignore").lstrip("﻿")
    # ui_car.json often has raw control chars inside strings -> pull fields with tolerant regex
    def field(key):
        m = re.search(r'"%s"\s*:\s*"([^"]*)"' % key, raw)
        return m.group(1) if m else None
    specs = {}
    m = re.search(r'"specs"\s*:\s*\{(.*?)\}', raw, re.S)
    if m:
        for k, v in re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', m.group(1)):
            specs[k] = v
    tags = re.findall(r'"tags"\s*:\s*\[(.*?)\]', raw, re.S)
    tagl = [t.strip().strip('"').lower() for t in tags[0].split(",")] if tags else []
    return dict(name=field("name") or cid, brand=field("brand") or "", specs=specs, tags=tagl)


def _grip_from_category(tags, pw_ratio):
    """Peak lateral mu estimate. Race rubber grips far more than street radials; very high
    power/weight cars tend to run stickier tyres too."""
    t = set(tags)
    race = {"race", "gt", "gt2", "gt3", "gt4", "lmp", "lmp1", "dtm", "formula", "openwheeler",
            "open wheeler", "semislick", "slick", "trackday", "track day", "prototype", "cup"}
    sport = {"sport", "supercar", "hypercar", "coupe", "tuned"}
    if t & race:
        g = 1.42
    elif t & sport:
        g = 1.18
    else:
        g = 1.05                                  # street radials
    if pw_ratio and pw_ratio < 4.0:               # kg/hp: <4 is seriously quick -> stickier
        g += 0.06
    return float(np.clip(g, 0.95, 1.55))


def _parse_data_ini(cid):
    """Read REAL physics from the car's data — decrypting data.acd if needed (car_acd handles both
    encrypted and unpacked). Returns mass, wheelbase, CG, drivetrain and tyre grip when available."""
    try:
        import car_acd
        files = car_acd.car_data(cid, AC_ROOT)
    except Exception:
        files = {}
    if "car.ini" not in files:
        return {}
    P = lambda fn: car_acd.parse_ini(files.get(fn, ""))
    car, susp, dr, tyres = P("car.ini"), P("suspensions.ini"), P("drivetrain.ini"), P("tyres.ini")
    o = {"from_acd": True}
    if "BASIC.TOTALMASS" in car:
        o["mass"] = _num(car["BASIC.TOTALMASS"])
    if "BASIC.WHEELBASE" in susp:
        o["L"] = _num(susp["BASIC.WHEELBASE"])
    if "BASIC.CG_LOCATION" in susp:
        o["cg_front"] = _num(susp["BASIC.CG_LOCATION"])
    if "TRACTION.TYPE" in dr:
        t = (dr["TRACTION.TYPE"] or "").upper().strip()
        o["drive"] = "AWD" if t.startswith("AWD") else ("FWD" if t.startswith("FWD") else "RWD")
    # tyre lateral grip: AC DY_REF (peak mu); average front/rear of the first compound
    for axle in ("FRONT", "REAR", "THERMAL_FRONT", "THERMAL_REAR"):
        for key in ("DY_REF", "DY0", "DY"):
            kk = "%s.%s" % (axle, key)
            if kk in tyres:
                o.setdefault("dy", []).append(_num(tyres[kk])); break
    # ENGINE + GEARBOX (for a real tractive-force curve): torque curve, gears, final, radius, limiter
    eng = P("engine.ini")
    lut = []
    for ln in files.get("power.lut", "").splitlines():
        ln = ln.split(";")[0].strip()
        if "|" in ln:
            a, b = ln.split("|", 1)
            try:
                lut.append((float(a), float(b)))
            except Exception:
                pass
    gears = []
    g = P("drivetrain.ini")
    i = 1
    while "GEARS.GEAR_%d" % i in g:
        gears.append(_num(g["GEARS.GEAR_%d" % i])); i += 1
    if lut and gears:
        o["lut"] = lut; o["gears"] = gears
        o["final"] = _num(g.get("GEARS.FINAL"), 3.5)
        o["limiter"] = _num(eng.get("ENGINE_DATA.LIMITER"), 8000.0)
        o["idle"] = _num(eng.get("ENGINE_DATA.MINIMUM"), 1000.0)
        drv = o.get("drive", "RWD")
        rad_axle = "FRONT" if drv == "FWD" else "REAR"
        o["radius"] = _num(tyres.get("%s.RADIUS" % rad_axle) or tyres.get("REAR.RADIUS"), 0.33)
    return o


def _tractive_curve(data, mass, vmax_ms, a_accel, n=48, eta=0.9):
    """Real tractive force vs speed from the torque curve + gears (envelope of best gear at each
    speed). Magnitude anchored to the spec 0-100 (a_accel) so it stays calibrated, shape from physics.
    Returns (force_list, dv) or None."""
    if not (data.get("lut") and data.get("gears")):
        return None
    rpm = np.array([p[0] for p in data["lut"]]); tq = np.array([p[1] for p in data["lut"]])
    gears = np.array(data["gears"]); final = data["final"]; radius = data["radius"]
    idle = data["idle"]; limiter = data["limiter"]
    dv = vmax_ms * 1.06 / n
    v = (np.arange(n) + 0.5) * dv
    F = np.zeros(n)
    for g in gears:
        ratio = g * final
        erpm = v / radius * ratio * 60.0 / (2 * np.pi)          # engine rpm at this speed in this gear
        usable = erpm <= limiter                                # below idle = clutch slip at launch (ok)
        t = np.interp(np.clip(erpm, rpm.min(), rpm.max()), rpm, tq)
        f = np.where(usable, t * ratio * eta / radius, 0.0)
        F = np.maximum(F, f)                                    # best gear at each speed (tractive envelope)
    peak = float(F[1:max(2, n // 6)].max()) or 1.0             # low-speed peak
    F *= (a_accel * mass) / peak                                # anchor peak accel to the spec figure
    return [round(float(x), 1) for x in F], round(float(dv), 4)


def build(cid):
    ui = _ui_specs(cid)
    sp = ui["specs"]
    mass = _num(sp.get("weight"), 1200.0)
    bhp = _num(sp.get("bhp") or sp.get("power"), 200.0)
    topkmh = _num(sp.get("topspeed"), 230.0)
    t0100 = _num(sp.get("acceleration"), 0.0)            # "5.1s 0-100"
    pw = (mass / bhp) if bhp else 6.0
    grip = _grip_from_category(ui["tags"], pw)

    data = _parse_data_ini(cid)
    if data.get("mass"):
        mass = data["mass"]
    vmax_ms = topkmh / 3.6
    # acceleration: prefer 0-100 time, else power/weight model; report PEAK (~1.2x average)
    if t0100 and t0100 > 1.0:
        a_accel = (27.78 / t0100) * 1.18
    else:
        a_accel = np.clip(bhp * 735.5 / max(mass, 1) / 12.0, 3.0, 14.0)   # P=F·v -> a at ~12 m/s
    a_accel = float(np.clip(a_accel, 3.0, 16.0))
    if data.get("dy"):
        grip = float(np.clip(np.mean(data["dy"]) * 1.05, 0.95, 1.6))     # real tyre DY (slight margin)
    a_brake = float(np.clip(grip * G * 0.92, 8.0, 22.0))                 # tyre-limited braking (real grip)
    L = float(data.get("L", 2.62))
    cg_front = float(data.get("cg_front", 0.50))                          # front weight fraction
    LF = L * (1.0 - cg_front); LR = L * cg_front
    HCG = 0.46
    IZ = mass * LF * LR * 1.1
    C_aero = a_accel * mass / max(vmax_ms ** 2, 1.0)                      # so terminal speed ~ vmax

    drive = data.get("drive", "RWD")
    drive_f = {"FWD": 1.0, "AWD": 0.5}.get(drive, 0.0)         # fraction of drive to the FRONT axle
    tract = _tractive_curve(data, mass, vmax_ms, a_accel)
    car = dict(id=cid, name=ui["name"], brand=ui["brand"], drive=drive,
               drive_f=drive_f, drive_r=round(1.0 - drive_f, 2),
               mass=round(mass, 1), bhp=round(bhp), vmax_ms=round(vmax_ms, 2),
               a_accel=round(a_accel, 3), a_brake=round(a_brake, 3), grip=round(grip, 3),
               steer_lock=0.30, L=round(L, 3), LF=round(LF, 3), LR=round(LR, 3),
               HCG=HCG, IZ=round(IZ, 1), C_aero=round(C_aero, 4),
               gears=len(data.get("gears", [])) or None, redline=int(data.get("limiter", 0)) or None,
               tractive=(tract[0] if tract else None), tract_dv=(tract[1] if tract else None),
               k_aero=0.0, aero_f=0.5,                        # downforce: set by telemetry system-ID
               from_data=("acd" if data else False), tags=ui["tags"])
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(car, open(os.path.join(OUT_DIR, cid + ".json"), "w"), indent=1)
    return car


def load(cid):
    p = os.path.join(OUT_DIR, cid + ".json")
    if not os.path.exists(p):
        return build(cid)
    return json.load(open(p))


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "bmw_m3_e92"
    c = build(cid)
    print("ingested %s (%s)" % (c["name"], c["brand"]))
    print("  mass %.0f kg | %d bhp | top %.0f km/h | 0-acc %.1f m/s² | brake %.1f m/s² | grip %.2f | drive %s%s"
          % (c["mass"], c["bhp"], c["vmax_ms"] * 3.6, c["a_accel"], c["a_brake"], c["grip"],
             c["drive"], "  [real data]" if c["from_data"] else "  [ui specs]"))
