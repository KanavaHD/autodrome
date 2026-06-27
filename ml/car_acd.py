"""
Assetto Corsa data.acd reader — decrypts a car's packed physics so we read the REAL tyres, mass,
geometry and drivetrain instead of estimating from ui_car.json specs.

Format: entries of [int nameLen][name][int contentLen][content], content = contentLen chars each in
4 bytes (low byte). Cipher: plain[i] = (enc[i] - ord(key[i % len(key)])) & 0xff. The key is 8 dash-
joined numbers derived from the car FOLDER name. Six of the eight parts use the classic AC reductions
(verified); the remaining two (positions 2 and 4) resisted a closed form, so we recover them by
brute-forcing the 256x256 space against decryption validity — robust for every car (30/30 tested).
"""
import os, struct

KEY_ALPHABET = "0123456789-"


def _known6(name):
    """The six key parts with confirmed closed forms (k1,k2,k4,k6,k7,k8)."""
    o = [ord(c) for c in name]; n = len(o)
    p0 = sum(o)
    p1 = 0
    for i in range(0, n - 1, 2):
        p1 = p1 * o[i] - o[i + 1]
    p3 = 0x1683
    for i in range(1, n):
        p3 -= o[i]
    p5 = 0x65
    for i in range(0, n - 2, 2):
        p5 -= o[i]
    p6 = 0xab
    for i in range(0, n - 2, 2):
        if o[i]:
            p6 %= o[i]
    p7 = 0xab
    for i in range(0, n - 1):
        if o[i]:
            p7 = p7 // o[i] + o[i + 1]
    return [p & 0xff for p in (p0, p1, p3, p5, p6, p7)]


# An AC ini is pure printable ASCII (incl. comment punctuation) + whitespace. The EXACT key yields
# 100% of these over the whole text; a near-miss key wraps some bytes out of this range -> we detect
# it over a long enough span. (Scoring a narrow charset fails: near-misses stay 'printable'.)
_INI_OK = set(range(32, 127)) | {9, 10, 13}


_BAD_CAP = 6   # a candidate with more than this many non-ini chars across the samples can't be the key


def _bad_count(samples, kb, kl):
    """Non-ini chars across several files' opening bytes, aborting early once hopeless. The exact
    key gives 0; a 1-digit-wrong key corrupts ~len/period chars in EVERY file, so it loses fast."""
    bad = 0
    for seg in samples:
        for i, e in enumerate(seg):
            if ((e - kb[i % kl]) & 0xff) not in _INI_OK:
                bad += 1
                if bad > _BAD_CAP:
                    return bad
    return bad


def acd_key(name, enc_dict):
    """Derive the full key: 6 parts by formula, P2 & P4 brute-forced against valid INI decryption
    across multiple files (so the exact key beats near-misses). Early-abort makes it fast."""
    p0, p1, p3, p5, p6, p7 = _known6(name)
    vfiles = ("car.ini", "tyres.ini", "engine.ini", "drivetrain.ini", "suspensions.ini",
              "brakes.ini", "aero.ini", "setup.ini")
    samples = [enc_dict[f][:1200] for f in vfiles if f in enc_dict][:6]
    if not samples:
        samples = [next(iter(enc_dict.values()))[:1500]]
    best = None; bb = 1 << 30
    for p2 in range(256):
        for p4 in range(256):
            key = "%d-%d-%d-%d-%d-%d-%d-%d" % (p0, p1, p2, p3, p4, p5, p6, p7)
            kb = [ord(c) for c in key]
            bad = _bad_count(samples, kb, len(kb))
            if bad < bb:
                bb = bad; best = key
                if bad == 0:                   # exact key — clean across every sample file
                    return key
    return best


def _parse_entries(raw):
    pos = 0
    if struct.unpack_from("<i", raw, 0)[0] == -1111:
        pos = 8
    enc = {}
    while pos + 8 <= len(raw):
        nl = struct.unpack_from("<i", raw, pos)[0]; pos += 4
        if nl <= 0 or nl > 256 or pos + nl > len(raw):
            break
        name = raw[pos:pos + nl].decode("ascii", "replace"); pos += nl
        clen = struct.unpack_from("<i", raw, pos)[0]; pos += 4
        if clen < 0 or pos + clen * 4 > len(raw):
            break
        enc[name] = [raw[pos + i * 4] for i in range(clen)]; pos += clen * 4
    return enc


def unpack_acd(acd_path, folder_name):
    raw = open(acd_path, "rb").read()
    enc = _parse_entries(raw)
    if not enc:
        return {}
    key = acd_key(folder_name, enc)
    kb = [ord(c) for c in key]; kl = len(kb)
    out = {}
    for name, e in enc.items():
        out[name] = bytes((e[i] - kb[i % kl]) & 0xff for i in range(len(e))).decode("utf-8", "ignore")
    return out


def car_data(folder_name, ac_root=None):
    """All physics files for a car id: unpacked data/ if present, else decrypt data.acd."""
    ac_root = ac_root or os.environ.get(
        "AC_ROOT", r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa")
    cdir = os.path.join(ac_root, "content", "cars", folder_name)
    udir = os.path.join(cdir, "data")
    if os.path.isfile(os.path.join(udir, "car.ini")):
        out = {}
        for fn in os.listdir(udir):
            try:
                out[fn] = open(os.path.join(udir, fn), "r", encoding="utf-8", errors="ignore").read()
            except Exception:
                pass
        return out
    acd = os.path.join(cdir, "data.acd")
    if os.path.isfile(acd):
        try:
            return unpack_acd(acd, folder_name)
        except Exception:
            return {}
    return {}


def parse_ini(text):
    out, sec = {}, None
    for ln in text.splitlines():
        ln = ln.split(";")[0].strip()
        if ln.startswith("[") and ln.endswith("]"):
            sec = ln[1:-1]
        elif "=" in ln and sec is not None:
            k, v = ln.split("=", 1); out["%s.%s" % (sec, k.strip())] = v.strip()
    return out


if __name__ == "__main__":
    import sys
    cid = sys.argv[1] if len(sys.argv) > 1 else "bmw_m3_e92"
    d = car_data(cid)
    print("decrypted %d files: %s%s" % (len(d), sorted(d)[:10], " ..." if len(d) > 10 else ""))
    for fn in ("car.ini", "tyres.ini", "drivetrain.ini"):
        if fn in d:
            print("\n--- %s ---\n%s" % (fn, "\n".join(d[fn].splitlines()[:5])))
