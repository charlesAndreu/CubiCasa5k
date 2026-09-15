#!/usr/bin/env python3
"""
Focused viewer for the new wall-graph post-process only (post_process_wall.py) --
no legacy Manhattan post-process, no heatmap/entropy/icon panels. One image: the
computed wall skeleton drawn on top of a selectable base layer (the input map, the
room/wall segmentation, or both alpha-blended together), so the skeleton can be
checked directly against exactly what it was scored against. Downloadable as PNG,
plus the raw geometry as JSON.

Reuses VizEngine from viz_web/inference.py (same model loading, LMDB presets, upload
handling) rather than duplicating that infrastructure.

Usage (from repo root):
  conda activate charles-cubicasa
  python viz_skeleton/app.py
  # or: ./viz_skeleton/run.sh

  Open http://127.0.0.1:8060
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request

from flask import Flask, Response, jsonify, request, send_from_directory
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import linemerge, unary_union

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_VIZ_WEB_DIR = os.path.join(_ROOT, "viz_web")
if _VIZ_WEB_DIR not in sys.path:
    sys.path.insert(0, _VIZ_WEB_DIR)
from inference import VizEngine  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB uploads
engine: VizEngine | None = None


def get_engine() -> VizEngine:
    global engine
    if engine is None:
        data_path = os.environ.get("CUBI_DATA_PATH", os.path.join(_ROOT, "data", "cubicasa5k"))
        roots = tuple(
            r.strip()
            for r in os.environ.get(
                "CUBI_RUN_ROOTS", "runs_cubi,runs_cubi_2,runs_cubi_3,runs_cubi_4"
            ).split(",")
            if r.strip()
        )
        engine = VizEngine(data_path=data_path, run_roots=roots)
    return engine


def _parse_source(body=None, args=None):
    """Return (plan_id, upload_id) from JSON body or query args."""
    upload_id = None
    plan_id = None
    if body:
        upload_id = body.get("upload_id") or None
        if body.get("plan_id") is not None:
            plan_id = int(body["plan_id"])
    if args:
        upload_id = upload_id or args.get("upload_id") or None
        if plan_id is None and args.get("plan_id") is not None:
            plan_id = int(args.get("plan_id"))
    if upload_id and plan_id is not None:
        raise ValueError("Specify either plan_id or upload_id, not both")
    if not upload_id and plan_id is None:
        raise ValueError("plan_id or upload_id required")
    return plan_id, upload_id


def _wall_criteria(args):
    """The new post-process's only tunable criteria. method: "score" (confidence/
    length/angle scoring race for each point's arity budget, see select_wall_edges)
    or "evidence" (segmentation directly decides topology, see
    select_wall_edges_by_evidence) -- axis_bias is unused in "evidence" mode."""
    return dict(
        threshold=float(args.get("threshold", 0.25)),
        method=args.get("method", "evidence"),
        axis_bias=float(args.get("axis_bias", 0.35)),
        snap_align=float(args.get("snap_align", 45.0)),
        wall_evidence=float(args.get("wall_evidence", 0.85)),
        min_wall_fraction=float(args.get("min_wall_fraction", 0.8)),
    )


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/edit")
def edit_page():
    """Graph editor: freezes the currently-selected model/plan/upload + criteria
    (passed as query params, same ones index.html's overlayQuery() builds) and lets
    the user manually edit the resulting point/edge graph. See edit.js."""
    return send_from_directory(STATIC_DIR, "edit.html")


@app.route("/geo")
def geo_page():
    """Georeference + export: takes the edited graph (handed off via
    sessionStorage by edit.js's "Continue to export" button -- no server state,
    same as the editor itself), lets the user place it on a real map (geocoded
    search or manual lat/lon) and translate/rotate/stretch it into alignment,
    then exports consolidated wall geometry as GeoJSON. See geo.js."""
    return send_from_directory(STATIC_DIR, "geo.html")


@app.get("/api/plans")
def api_plans():
    return jsonify(get_engine().list_plans())


@app.get("/api/models")
def api_models():
    return jsonify(get_engine().list_models())


@app.post("/api/upload")
def api_upload():
    if "image" not in request.files:
        return jsonify({"error": "Missing form field 'image'"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    data = f.read()
    if not data:
        return jsonify({"error": "Empty file"}), 400
    try:
        entry = get_engine().store_upload(data, filename=f.filename)
        h, w = entry.full_res_shape
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(
        {"ok": True, "upload_id": entry.upload_id, "filename": entry.filename, "height": h, "width": w}
    )


@app.get("/api/input.png")
def api_input_query():
    try:
        plan_id, upload_id = _parse_source(args=request.args)
        png = get_engine().get_input_png(plan_id=plan_id, upload_id=upload_id)
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return Response(png, mimetype="image/png")


@app.get("/api/input/<int:plan_id>.png")
def api_input_preset(plan_id: int):
    try:
        png = get_engine().get_input_png(plan_id=plan_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return Response(png, mimetype="image/png")


@app.post("/api/run")
def api_run():
    body = request.get_json(force=True, silent=True) or {}
    model_id = body.get("model_id")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    try:
        plan_id, upload_id = _parse_source(body=body)
        run = get_engine().run_inference(model_id, plan_id=plan_id, upload_id=upload_id)
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(
        {
            "ok": True,
            "folder": run.folder,
            "height": run.full_res_shape[0],
            "width": run.full_res_shape[1],
            "upload_id": upload_id,
            "plan_id": plan_id,
        }
    )


@app.get("/api/overlay.png")
def api_overlay_png():
    """The computed wall skeleton drawn on top of a selectable base layer.
    base: "map" | "segmentation" | "both" (segmentation alpha-blended over the map,
    alpha controlled by seg_alpha)."""
    model_id = request.args.get("model_id")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    try:
        plan_id, upload_id = _parse_source(args=request.args)
        base = request.args.get("base", "map")
        seg_alpha = float(request.args.get("seg_alpha", 0.5))
        png = get_engine().skeleton_overlay_png(
            model_id, base=base, seg_alpha=seg_alpha,
            plan_id=plan_id, upload_id=upload_id, **_wall_criteria(request.args)
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return Response(png, mimetype="image/png")


@app.get("/api/base.png")
def api_base_png():
    """Same base layer as /api/overlay.png (map | segmentation | both), but with no
    skeleton drawn on top -- the graph editor page draws its own interactive
    point/edge overlay and needs a clean image underneath, at the same native
    resolution/pixel space as /api/skeleton.json."""
    model_id = request.args.get("model_id")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    try:
        plan_id, upload_id = _parse_source(args=request.args)
        base = request.args.get("base", "map")
        seg_alpha = float(request.args.get("seg_alpha", 0.5))
        png = get_engine().base_layer_png(
            model_id, base=base, seg_alpha=seg_alpha, plan_id=plan_id, upload_id=upload_id,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return Response(png, mimetype="image/png")


@app.get("/api/skeleton.json")
def api_skeleton_json():
    """Raw wall-network geometry (points/wall_segments/openings) for download."""
    model_id = request.args.get("model_id")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    try:
        plan_id, upload_id = _parse_source(args=request.args)
        result = get_engine().wall_network_result(
            model_id, plan_id=plan_id, upload_id=upload_id, **_wall_criteria(request.args)
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(result)


_NOMINATIM_USER_AGENT = "cubicasa5k-viz-skeleton/1.0 (internal dev tool)"
_geocode_lock = threading.Lock()
_last_geocode_call = 0.0


@app.get("/api/geocode")
def api_geocode():
    """Proxies Nominatim's free geocoding search (https://nominatim.org) server
    side. Two things a plain browser fetch can't do correctly: Nominatim's usage
    policy (https://operations.osmfoundation.org/policies/nominatim/) requires a
    request identifying the application via User-Agent, which browsers don't let
    a page set on its own requests; and it caps usage at one request/second,
    enforced here with a simple global throttle (single-user local dev tool, so a
    process-wide lock is enough)."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "q required"}), 400

    global _last_geocode_call
    with _geocode_lock:
        wait = 1.0 - (time.time() - _last_geocode_call)
        if wait > 0:
            time.sleep(wait)
        _last_geocode_call = time.time()

    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": q, "format": "jsonv2", "limit": 5}
    )
    req = urllib.request.Request(url, headers={"User-Agent": _NOMINATIM_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return jsonify({"error": f"geocoding request failed: {e}"}), 502

    return jsonify([
        {"display_name": item.get("display_name"), "lat": float(item["lat"]), "lon": float(item["lon"])}
        for item in data
    ])


def _iter_polygons(geom):
    """Every Polygon inside a shapely result, whatever wrapper it came back in
    (Polygon / MultiPolygon / GeometryCollection). A negative buffer in
    particular can return any of the three, or an empty geometry."""
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geom.geoms:
            yield from _iter_polygons(part)


def _polygon_rings(poly):
    """[exterior, *holes] as plain [x, y] lists -- GeoJSON Polygon ring order,
    with GeoJSON's winding too (RFC 7946: exterior counter-clockwise, holes
    clockwise), since these rings are written straight into the export."""
    poly = orient(poly, sign=1.0)
    return [[list(pt) for pt in poly.exterior.coords]] + [
        [list(pt) for pt in hole.coords] for hole in poly.interiors
    ]


def _band(lines, half_width):
    """The wall band for a set of centerlines: each thickened by half_width on
    either side, all of them unioned.

    The linemerge() is not cosmetic. Buffering a bundle of separate two-point
    segments thickens every one of them on its own, so where two walls meet at
    a corner the outer corner square -- half a wall on each side -- belongs to
    neither rectangle, and the band comes out of the corner visibly notched.
    Merged into continuous runs first, that corner is a join rather than two
    flat caps, and join_style=2 (mitre) makes it a proper square 90deg one.
    Flat caps still apply where a wall genuinely ends -- a merged run's own two
    ends -- so a dead-end stub stops at its endpoint instead of overshooting by
    half a wall."""
    if not lines:
        return Polygon()
    return linemerge(unary_union(lines)).buffer(half_width, cap_style=2, join_style=2)


def _seal(band, door_bridge):
    """Morphological closing: seals any opening up to `door_bridge` wide while
    leaving every wall face away from the gap exactly where it was. Mitre
    joins on both halves, so corners stay square instead of being rounded off
    by the erosion."""
    if door_bridge <= 0:
        return band
    r = door_bridge / 2.0
    return band.buffer(r, join_style=2).buffer(-r, join_style=2)


def _filled_outline(band):
    """The building's outer face: the band's exterior rings with their holes
    filled back in, so what's left is the outline around the whole footprint
    and nothing about the rooms inside it. Used only to tell outside walls
    from inside ones -- it is not exported."""
    filled = [Polygon(poly.exterior) for poly in _iter_polygons(band)]
    return unary_union(filled) if filled else band


def _runs_along(line, outline, max_dist, coverage=0.6, samples=16):
    """True when most of `line` runs within max_dist of `outline` -- the test
    for "this wall is on the outside of the building". Sampled along the
    segment rather than measured at its midpoint or its ends: an interior wall
    meeting an exterior one has an endpoint right on the outline, and only
    looking at how much of its LENGTH is out there tells the two apart."""
    hits = sum(
        1
        for i in range(samples + 1)
        if outline.distance(line.interpolate(i / samples, normalized=True)) <= max_dist
    )
    return hits >= coverage * (samples + 1)


@app.post("/api/footprint")
def api_footprint():
    """Turns the wall *centerline* graph into the areal layers the export
    needs: the walls themselves with a real thickness, and the rooms they
    enclose.

    Works in whatever planar unit the caller sends -- geo.js sends plan-meters
    (local pixels times the current m/px scale), so wall_width,
    exterior_width, door_bridge and min_room_area are real-world meters here
    and a buffer is a real wall thickness, independent of the plan's pixel
    resolution or of how the user has stretched the shape on the map.

    Walls: every segment thickened by half its width on each side, then
    unioned, so corners and T-junctions merge into one clean band instead of a
    pile of overlapping rectangles. Flat caps, so a dead-end stub stops at its
    own endpoint instead of overshooting by half a wall; mitre joins, so
    corners stay square rather than rounded off.

    Outside walls are thicker (exterior_width instead of wall_width), and
    which ones those are is read off the skeleton itself: build the band at
    the nominal width, seal its doorways, and any segment that then runs along
    the outer face of that band is an outside wall. Nothing is drawn around
    the building that isn't a wall of the graph -- an outside wall is one of
    the same bands, just wider.

    Rooms: the enclosed voids of the final band -- literally its holes, which
    by construction follow the inner wall faces exactly ("just inside the
    walls") and can never overlap a wall. The band is sealed first (see
    _seal), so an opening up to door_bridge wide -- a doorway, or a gap left
    by an edge deleted in the editor -- can neither cut a room in two nor let
    it bleed into the next room. Voids below min_room_area are dropped as
    slivers.

    Note the sealing also erases any genuine room space narrower than
    door_bridge (a 60cm broom cupboard, say) -- the price of not needing the
    network to be topologically perfect, and the reason door_bridge is a
    request parameter rather than a constant."""
    body = request.get_json(force=True, silent=True) or {}
    segments = body.get("segments") or []
    wall_width = float(body.get("wall_width", 0.2))
    exterior_width = float(body.get("exterior_width", wall_width))
    door_bridge = float(body.get("door_bridge", 0.9))
    min_room_area = float(body.get("min_room_area", 1.0))

    empty = {"walls": [], "exterior_walls": [], "rooms": []}
    lines = [LineString(seg) for seg in segments if len(seg) >= 2]
    if not lines or wall_width <= 0:
        return jsonify(empty)

    try:
        half = wall_width / 2.0
        ext_half = max(exterior_width, wall_width) / 2.0

        # Pass 1, at the nominal width only: what the outer face of the
        # building looks like, which is what says whether a wall is outside.
        outline = _filled_outline(_seal(_band(lines, half), door_bridge))
        tolerance = half + max(wall_width * 0.25, 0.02)
        outside = [_runs_along(ln, outline.boundary, tolerance) for ln in lines]
        exterior = [ln for ln, is_out in zip(lines, outside) if is_out]
        interior = [ln for ln, is_out in zip(lines, outside) if not is_out]

        # Pass 2: the real bands, each at its own width. The interior band
        # gives way to the exterior one where they meet, so the two sets of
        # polygons stay disjoint and each can honestly carry the width it was
        # built with.
        ext_band = _band(exterior, ext_half)
        int_band = _band(interior, half)
        if not ext_band.is_empty and not int_band.is_empty:
            int_band = int_band.difference(ext_band)

        walls = unary_union([ext_band, int_band])
        sealed = _seal(walls, door_bridge)
        rooms = [Polygon(hole) for poly in _iter_polygons(sealed) for hole in poly.interiors]
        rooms = [r for r in rooms if r.is_valid and r.area >= min_room_area]

        return jsonify(
            {
                "walls": [_polygon_rings(p) for p in _iter_polygons(int_band)],
                "exterior_walls": [_polygon_rings(p) for p in _iter_polygons(ext_band)],
                "rooms": [_polygon_rings(p) for p in rooms],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="Wall-graph post-process viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
