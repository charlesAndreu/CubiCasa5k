#!/usr/bin/env python3
"""
Interactive viewer for CubiCasa5k full-model predictions.

Usage (from repo root):
  conda activate charles-cubicasa
  python viz_web/app.py
  # or: ./viz_web/run.sh

  Open http://127.0.0.1:5050
"""

from __future__ import annotations

import argparse
import os
import sys

from flask import Flask, Response, jsonify, request, send_from_directory

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
if _VIZ_DIR not in sys.path:
    sys.path.insert(0, _VIZ_DIR)
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


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


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
        {
            "ok": True,
            "upload_id": entry.upload_id,
            "filename": entry.filename,
            "height": h,
            "width": w,
        }
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
        run = get_engine().run_inference(
            model_id, plan_id=plan_id, upload_id=upload_id
        )
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


@app.get("/api/artifact/<name>.png")
def api_artifact(name: str):
    model_id = request.args.get("model_id")
    if not model_id:
        return jsonify({"error": "model_id query param required"}), 400
    try:
        plan_id, upload_id = _parse_source(args=request.args)
        png = get_engine().artifact_png(
            model_id, name, plan_id=plan_id, upload_id=upload_id
        )
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return Response(png, mimetype="image/png")


@app.get("/api/postproc.png")
def api_postproc():
    model_id = request.args.get("model_id")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    try:
        threshold = float(request.args.get("threshold", 0.25))
        plan_id, upload_id = _parse_source(args=request.args)
        png = get_engine().postproc_png(
            model_id,
            threshold,
            plan_id=plan_id,
            upload_id=upload_id,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return Response(png, mimetype="image/png")


def main():
    parser = argparse.ArgumentParser(description="CubiCasa5k prediction viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
