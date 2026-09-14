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
from shapely.geometry import LineString
from shapely.ops import unary_union

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


@app.post("/api/outer_boundary")
def api_outer_boundary():
    """Finds the building's single outer envelope from a set of wall segments
    (local pixel coordinates), for the georeferencing step's "outside" export --
    one closed ring around the whole footprint, distinct from (and drawn bolder
    than) the individual interior wall runs.

    Deliberately NOT shapely.ops.polygonize: that only works on an *exactly*
    closed line network, so any small gap or dangling stub -- routine in a
    hand-edited graph -- makes the whole computation come back with nothing at
    all for that section, which looks exactly like "missing segments" in the
    export. Instead: thicken every wall line into a band (like giving it real
    wall thickness, `gap_tolerance_px` wide) and union the bands together.
    This bridges any gap up to about 2x the tolerance, and a stray unconnected
    stub just shows up as a small bump instead of breaking anything -- works on
    any set of segments, not just a topologically perfect closed loop. The
    tradeoff is a small, constant outward offset (by gap_tolerance_px) versus
    the true wall centerline, which is negligible next to hand-placing the
    plan on a map. Returns {"boundary": null} only when there's truly nothing
    to work with (no segments at all)."""
    body = request.get_json(force=True, silent=True) or {}
    segments = body.get("segments") or []
    gap_tolerance_px = float(body.get("gap_tolerance_px", 6.0))
    lines = [LineString(seg) for seg in segments if len(seg) >= 2]
    if not lines:
        return jsonify({"boundary": None})
    try:
        merged_lines = unary_union(lines)
        dilated = merged_lines.buffer(gap_tolerance_px, join_style=2)
        if dilated.is_empty:
            return jsonify({"boundary": None})
        exterior = (
            dilated.exterior
            if dilated.geom_type == "Polygon"
            else max(dilated.geoms, key=lambda g: g.area).exterior
        )
        return jsonify({"boundary": [list(pt) for pt in exterior.coords]})
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
