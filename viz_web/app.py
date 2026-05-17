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
engine: VizEngine | None = None


def get_engine() -> VizEngine:
    global engine
    if engine is None:
        data_path = os.environ.get("CUBI_DATA_PATH", os.path.join(_ROOT, "data", "cubicasa5k"))
        roots = tuple(
            r.strip()
            for r in os.environ.get("CUBI_RUN_ROOTS", "runs_cubi,runs_cubi_2,runs_cubi_3").split(",")
            if r.strip()
        )
        engine = VizEngine(data_path=data_path, run_roots=roots)
    return engine


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/plans")
def api_plans():
    return jsonify(get_engine().list_plans())


@app.get("/api/models")
def api_models():
    return jsonify(get_engine().list_models())


@app.get("/api/input/<int:plan_id>.png")
def api_input(plan_id: int):
    try:
        png = get_engine().get_input_png(plan_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return Response(png, mimetype="image/png")


@app.post("/api/run")
def api_run():
    body = request.get_json(force=True, silent=True) or {}
    plan_id = int(body.get("plan_id", 0))
    model_id = body.get("model_id")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    try:
        run = get_engine().run_inference(plan_id, model_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(
        {
            "ok": True,
            "folder": run.folder,
            "height": run.full_res_shape[0],
            "width": run.full_res_shape[1],
        }
    )


@app.get("/api/artifact/<name>.png")
def api_artifact(name: str):
    plan_id = int(request.args.get("plan_id", 0))
    model_id = request.args.get("model_id")
    if not model_id:
        return jsonify({"error": "model_id query param required"}), 400
    try:
        png = get_engine().artifact_png(plan_id, model_id, name)
    except KeyError:
        return jsonify({"error": f"unknown artifact: {name}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return Response(png, mimetype="image/png")


@app.get("/api/postproc.png")
def api_postproc():
    plan_id = int(request.args.get("plan_id", 0))
    model_id = request.args.get("model_id")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    try:
        threshold = float(request.args.get("threshold", 0.25))
    except ValueError:
        return jsonify({"error": "invalid threshold"}), 400
    try:
        png = get_engine().postproc_png(plan_id, model_id, threshold)
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
