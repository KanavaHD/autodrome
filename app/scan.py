"""
Scan the local Assetto Corsa install for everything you could train on: every car, and every
track layout, with display names, preview images, and whether an AI line exists (the AI line is
what the track ingestor turns into a centreline -> sim, so 'has_ai' = 'trainable today').

Pure stdlib. Results are cached; call scan_library(force=True) to rescan.
"""
import os, json, functools

AC_ROOT = os.environ.get(
    "AC_ROOT", r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa")
CARS_DIR = os.path.join(AC_ROOT, "content", "cars")
TRACKS_DIR = os.path.join(AC_ROOT, "content", "tracks")


def _read_json_loose(path):
    """AC ui_*.json files are often not strict JSON (raw newlines in strings, trailing commas,
    BOM). Read what we can; fall back to pulling the first "name" with a tolerant scan."""
    try:
        with open(path, "rb") as f:
            raw = f.read().decode("utf-8", "ignore").lstrip("﻿")
        try:
            return json.loads(raw)
        except Exception:
            # tolerant: kill literal newlines inside the file then retry
            try:
                return json.loads(raw.replace("\r", " ").replace("\n", " "))
            except Exception:
                out = {}
                for key in ("name", "brand", "country", "city", "length", "width", "tags"):
                    i = raw.find('"%s"' % key)
                    if i < 0:
                        continue
                    c = raw.find(":", i)
                    q1 = raw.find('"', c + 1)
                    q2 = raw.find('"', q1 + 1)
                    if q1 > 0 and q2 > q1:
                        out[key] = raw[q1 + 1:q2]
                return out
    except Exception:
        return {}


def _first_existing(*paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def scan_cars():
    cars = []
    if not os.path.isdir(CARS_DIR):
        return cars
    for cid in sorted(os.listdir(CARS_DIR)):
        d = os.path.join(CARS_DIR, cid)
        if not os.path.isdir(d):
            continue
        ui = _read_json_loose(os.path.join(d, "ui", "ui_car.json"))
        # a preview lives under the first skin; the badge is the brand roundel
        skins = os.path.join(d, "skins")
        preview = None
        if os.path.isdir(skins):
            for sk in sorted(os.listdir(skins)):
                p = _first_existing(os.path.join(skins, sk, "preview.jpg"),
                                    os.path.join(skins, sk, "preview.png"))
                if p:
                    preview = p
                    break
        badge = _first_existing(os.path.join(d, "ui", "badge.png"), os.path.join(d, "logo.png"))
        cars.append({
            "id": cid,
            "name": (ui.get("name") or cid).strip(),
            "brand": (ui.get("brand") or "").strip(),
            "tags": ui.get("tags") if isinstance(ui.get("tags"), list) else [],
            "power": (ui.get("specs", {}) or {}).get("bhp", "") if isinstance(ui.get("specs"), dict) else "",
            "weight": (ui.get("specs", {}) or {}).get("weight", "") if isinstance(ui.get("specs"), dict) else "",
            "has_preview": bool(preview),
            "has_badge": bool(badge),
            "encrypted": os.path.isfile(os.path.join(d, "data.acd")) and not os.path.isdir(os.path.join(d, "data")),
        })
    return cars


def _track_layouts(tid, d):
    """Return [(layout_id_or_None, ui_dir, ai_path)]. Single-layout tracks keep layout=None."""
    out = []
    root_ai = os.path.join(d, "ai", "fast_lane.ai")
    if os.path.isfile(root_ai):
        out.append((None, os.path.join(d, "ui"), root_ai))
    # layout subfolders: a dir that has its own ai/fast_lane.ai
    for sub in sorted(os.listdir(d)):
        sd = os.path.join(d, sub)
        if not os.path.isdir(sd) or sub in ("ui", "skins", "data", "ai", "texture"):
            continue
        ai = os.path.join(sd, "ai", "fast_lane.ai")
        if os.path.isfile(ai):
            uid = _first_existing(os.path.join(d, "ui", sub, "ui_track.json"))
            out.append((sub, os.path.dirname(uid) if uid else os.path.join(d, "ui", sub), ai))
    return out


def scan_tracks():
    tracks = []
    if not os.path.isdir(TRACKS_DIR):
        return tracks
    for tid in sorted(os.listdir(TRACKS_DIR)):
        d = os.path.join(TRACKS_DIR, tid)
        if not os.path.isdir(d):
            continue
        layouts = _track_layouts(tid, d)
        if not layouts:
            layouts = [(None, os.path.join(d, "ui"), None)]   # show it, mark not-trainable
        for layout, ui_dir, ai in layouts:
            ui = _read_json_loose(os.path.join(ui_dir, "ui_track.json"))
            preview = _first_existing(os.path.join(ui_dir, "preview.png"),
                                      os.path.join(ui_dir, "preview.jpg"))
            outline = _first_existing(os.path.join(ui_dir, "outline.png"),
                                      os.path.join(d, "map.png"))
            key = tid if layout is None else "%s/%s" % (tid, layout)
            nm = (ui.get("name") or tid).strip()
            if layout:
                nm = "%s — %s" % (nm, layout.upper())
            tracks.append({
                "id": key, "base": tid, "layout": layout or "",
                "name": nm,
                "country": (ui.get("country") or "").strip(),
                "length_m": _to_int(ui.get("length")),
                "width_m": _to_int(ui.get("width")),
                "has_ai": bool(ai),
                "has_preview": bool(preview),
                "has_outline": bool(outline),
            })
    return tracks


def _to_int(v):
    try:
        return int(float(str(v).replace(",", "")))
    except Exception:
        return 0


def car_image(cid, kind="preview"):
    d = os.path.join(CARS_DIR, cid)
    if kind == "badge":
        return _first_existing(os.path.join(d, "ui", "badge.png"), os.path.join(d, "logo.png"))
    skins = os.path.join(d, "skins")
    if os.path.isdir(skins):
        for sk in sorted(os.listdir(skins)):
            p = _first_existing(os.path.join(skins, sk, "preview.jpg"),
                                os.path.join(skins, sk, "preview.png"))
            if p:
                return p
    return None


def track_image(key, kind="preview"):
    base, _, layout = key.partition("/")
    d = os.path.join(TRACKS_DIR, base)
    ui_dir = os.path.join(d, "ui", layout) if layout else os.path.join(d, "ui")
    if kind == "outline":
        return _first_existing(os.path.join(ui_dir, "outline.png"), os.path.join(d, "map.png"))
    return _first_existing(os.path.join(ui_dir, "preview.png"), os.path.join(ui_dir, "preview.jpg"),
                           os.path.join(d, "map.png"))


@functools.lru_cache(maxsize=1)
def _cached():
    return {"ac_root": AC_ROOT, "ac_ok": os.path.isdir(CARS_DIR),
            "cars": scan_cars(), "tracks": scan_tracks()}


def scan_library(force=False):
    if force:
        _cached.cache_clear()
    return _cached()


if __name__ == "__main__":
    lib = scan_library(force=True)
    print("AC ok:", lib["ac_ok"], "| root:", lib["ac_root"])
    print("cars:", len(lib["cars"]), "| tracks/layouts:", len(lib["tracks"]),
          "| trainable tracks:", sum(t["has_ai"] for t in lib["tracks"]))
