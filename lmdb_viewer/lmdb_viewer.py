import io
import pickle
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, send_file

LMDB_PATH = "data/cubicasa5k/cubi_lmdb"
MAX_KEYS = 2000

app = Flask(__name__)
env = lmdb.open(
    LMDB_PATH,
    readonly=True,
    lock=False,
    readahead=True,
    meminit=False,
    max_readers=32,
)

ROOM_CLASSES = [
    "Background",
    "Outdoor",
    "Wall",
    "Kitchen",
    "Living Room",
    "Bed Room",
    "Bath",
    "Entry",
    "Railing",
    "Storage",
    "Garage",
    "Undefined",
]

ICON_CLASSES = [
    "No Icon",
    "Window",
    "Door",
    "Closet",
    "Electrical Appliance",
    "Toilet",
    "Sink",
    "Sauna Bench",
    "Fire Place",
    "Bathtub",
    "Chimney",
]

# Colors from floortrans/plotting.py discrete_cmap().
ROOM_COLORS = [
    "#DCDCDC",
    "#b3de69",
    "#000000",
    "#8dd3c7",
    "#fdb462",
    "#fccde5",
    "#80b1d3",
    "#808080",
    "#fb8072",
    "#696969",
    "#577a4d",
    "#ffffb3",
]

ICON_COLORS = [
    "#DCDCDC",
    "#8dd3c7",
    "#b15928",
    "#fdb462",
    "#ffff99",
    "#fccde5",
    "#80b1d3",
    "#808080",
    "#fb8072",
    "#696969",
    "#577a4d",
]

POI_TYPES = {
    0: "I-up",
    1: "I-left",
    2: "I-down",
    3: "I-right",
    4: "L-up-left",
    5: "L-up-right",
    6: "L-down-right",
    7: "L-down-left",
    8: "T-up",
    9: "T-right",
    10: "T-down",
    11: "T-left",
    12: "X-cross",
    13: "Opening-left",
    14: "Opening-right",
    15: "Opening-up",
    16: "Opening-down",
    17: "Icon-upper-left",
    18: "Icon-upper-right",
    19: "Icon-lower-left",
    20: "Icon-lower-right",
}

POI_GROUPS = {
    "Junctions": list(range(0, 13)),
    "Openings": list(range(13, 17)),
    "Icon corners": list(range(17, 21)),
}


def _png_from_chw_uint8(chw):
    chw = np.asarray(chw)
    if chw.ndim != 3:
        raise ValueError(f"Expected CHW, got {chw.shape}")
    # label is single channel
    if chw.shape[0] == 1:
        img = chw[0]
        mode = "L"
    # image is 3 channels
    elif chw.shape[0] == 3:
        img = np.moveaxis(chw, 0, -1)
        mode = "RGB"
    else:
        raise ValueError(f"Unsupported channel count {chw.shape[0]}")

    if img.dtype != np.uint8:
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)

    pil = Image.fromarray(img, mode=mode)
    buff = io.BytesIO()
    pil.save(buff, format="PNG")
    buff.seek(0)
    return buff


def _hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]


def _indexed_to_color_png(index_map, colors_hex):
    index_map = np.asarray(index_map)
    if index_map.ndim != 2:
        raise ValueError(f"Expected HW map, got {index_map.shape}")
    palette = np.array([_hex_to_rgb(c) for c in colors_hex], dtype=np.uint8)
    idx = np.clip(np.rint(index_map), 0, len(colors_hex) - 1).astype(np.int64)
    rgb = palette[idx]
    pil = Image.fromarray(rgb, mode="RGB")
    buff = io.BytesIO()
    pil.save(buff, format="PNG")
    buff.seek(0)
    return buff


@app.get("/api/keys")
def keys():
    out = []
    with env.begin(write=False) as txn:
        cur = txn.cursor()
        for i, (k, _) in enumerate(cur):
            out.append(k.decode("utf-8"))
            if i + 1 >= MAX_KEYS:
                break
    return jsonify(out)


@app.get("/api/sample")
def sample():
    key = request.args.get("key", "")
    if not key:
        return jsonify({"error": "missing key"}), 400

    with env.begin(write=False) as txn:
        raw = txn.get(key.encode("utf-8"))
    if raw is None:
        return jsonify({"error": f"key not found: {key}"}), 404

    s = pickle.loads(raw)
    image = np.asarray(s["image"])
    label = np.asarray(s["label"])
    heatmaps = s.get("heatmaps", {})
    scale = s.get("scale", None)

    meta = {
        "folder": s.get("folder", key),
        "image_shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "label_shape": list(label.shape),
        "label_dtype": str(label.dtype),
        "heatmap_channels": len(heatmaps) if isinstance(heatmaps, dict) else None,
        "scale": scale,
    }
    return jsonify(meta)


@app.get("/api/config")
def config():
    return jsonify(
        {
            "room_classes": ROOM_CLASSES,
            "icon_classes": ICON_CLASSES,
            "room_colors": ROOM_COLORS,
            "icon_colors": ICON_COLORS,
            "poi_types": {str(k): v for k, v in POI_TYPES.items()},
            "poi_groups": {
                k: [str(v) for v in values] for k, values in POI_GROUPS.items()
            },
        }
    )


@app.get("/api/image")
def image():
    key = request.args.get("key", "")
    with env.begin(write=False) as txn:
        raw = txn.get(key.encode("utf-8"))
    if raw is None:
        return "missing", 404
    s = pickle.loads(raw)
    image = np.asarray(s["image"])
    return send_file(_png_from_chw_uint8(image), mimetype="image/png")


@app.get("/api/label_channel")
def label_channel():
    key = request.args.get("key", "")
    c = int(request.args.get("c", "0"))

    with env.begin(write=False) as txn:
        raw = txn.get(key.encode("utf-8"))
    if raw is None:
        return "missing", 404
    s = pickle.loads(raw)
    label = np.asarray(s["label"])
    if c < 0 or c >= label.shape[0]:
        return "bad channel", 400

    ch = label[c]
    colors_hex = ROOM_COLORS if c == 0 else ICON_COLORS
    return send_file(_indexed_to_color_png(ch, colors_hex), mimetype="image/png")


@app.get("/api/points")
def points():
    key = request.args.get("key", "")
    with env.begin(write=False) as txn:
        raw = txn.get(key.encode("utf-8"))
    if raw is None:
        return jsonify({"error": "missing"}), 404
    s = pickle.loads(raw)
    heatmaps = s.get("heatmaps", {})
    pts = {str(k): v for k, v in heatmaps.items()}
    return jsonify(pts)


@app.get("/")
def index():
    return Path("lmdb_viewer/lmdb_viewer.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    app.run(debug=True, port=8080)
