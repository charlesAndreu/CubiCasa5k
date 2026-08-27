"""Inference and image rendering for the floor-plan visualization web UI."""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field

import cv2
import lmdb
import matplotlib.colors as mcolors
import numpy as np
import torch
from PIL import Image

# Project root on sys.path when launched as `python viz_web/app.py`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eval_full import (  # noqa: E402
    COMBINED_CLASSES,
    INPUT_SLICE,
    N_HEATMAPS,
    N_ICON_CLASSES,
    N_ROOM_CLASSES,
    WALL_CLASS,
    _entropy_from_probs,
    build_combined_map,
    combined_map_colors,
    load_eval_args,
    predict_at_resolution,
    run_postproc_mini,
)
from eval_simple import (  # noqa: E402
    _tab20_segmentation_colors,
    _tensor_to_bgr_uint8,
    entropy_hw_to_inferno_rgb,
)
from floortrans import post_prosessing  # noqa: E402
from floortrans.loaders.augmentations import DictToTensor  # noqa: E402
from floortrans.loaders.room_icon_loaders import FullLoader  # noqa: E402
from model import cubi_casa5k_full_model  # noqa: E402
from post_process_wall import (  # noqa: E402
    build_wall_network,
    remap_prediction_to_wall_heatmaps,
    render_wall_network_bgr,
)

CHANNEL_GROUPS = {
    "wall": (0, 13),
    "opening": (13, 17),
    "icon": (17, 21),
}

PRESETS_PATH = os.path.join(os.path.dirname(__file__), "presets.json")
DEFAULT_DATA_PATH = os.path.join(_ROOT, "data", "cubicasa5k")
DEFAULT_RUN_ROOTS = ("runs_cubi", "runs_cubi_2", "runs_cubi_3")
# Longest side of images sent to the browser (native plans are often 600–2000 px).
DISPLAY_MAX_SIDE = int(os.environ.get("CUBI_VIZ_MAX_SIDE", "560"))
UPLOAD_MAX_SIDE = int(os.environ.get("CUBI_VIZ_UPLOAD_MAX_SIDE", "2048"))


def _image_bytes_to_model_tensor(data: bytes) -> torch.Tensor:
    """Decode image file bytes -> (3, H, W) float in [-1, 1] (same as FullLoader)."""
    pil = Image.open(io.BytesIO(data))
    pil = pil.convert("RGB")
    arr = np.array(pil, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected an RGB image")
    h, w = arr.shape[:2]
    if max(h, w) > UPLOAD_MAX_SIDE:
        scale = UPLOAD_MAX_SIDE / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(arr.transpose(2, 0, 1)).float()
    return 2.0 * (t / 255.0) - 1.0


def _resize_rgb(rgb: np.ndarray, max_side: int = DISPLAY_MAX_SIDE) -> np.ndarray:
    h, w = rgb.shape[:2]
    if max(h, w) <= max_side:
        return rgb
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _png_bytes(rgb: np.ndarray, max_side: int = DISPLAY_MAX_SIDE) -> bytes:
    """RGB uint8 (H, W, 3) -> PNG bytes (downscaled for the web UI)."""
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb = _resize_rgb(rgb, max_side)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _entropy_rgb_from_probs(probs_chw: np.ndarray, n_classes: int) -> np.ndarray:
    """Softmax probs (C, H, W) -> inferno RGB via eval_full + eval_simple helpers."""
    entropy_hw = _entropy_from_probs(probs_chw, n_classes).numpy()
    return entropy_hw_to_inferno_rgb(entropy_hw)


def _heatmap_max_rgb(planes: np.ndarray) -> np.ndarray:
    """Max over channel dim, colormap inferno -> RGB uint8."""
    hm = np.max(planes, axis=0)
    hm = np.clip(hm, 0.0, 1.0)
    u8 = (hm * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _seg_rgb(seg_hw: np.ndarray, n_classes: int, extra_colors=None) -> np.ndarray:
    colors = _tab20_segmentation_colors(n_classes)
    if extra_colors is not None:
        colors = extra_colors
    h, w = seg_hw.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(n_classes):
        mask = seg_hw == c
        if not np.any(mask):
            continue
        rgba = colors[c]
        out[mask] = (np.array(rgba[:3]) * 255.0).astype(np.uint8)
    return out


def _combined_rgb(combined_hw: np.ndarray) -> np.ndarray:
    room_colors = _tab20_segmentation_colors(N_ROOM_CLASSES)
    cmap = mcolors.ListedColormap(combined_map_colors(room_colors))
    combined_hw = np.clip(combined_hw, 0, COMBINED_CLASSES - 1)
    rgba = cmap(combined_hw / max(COMBINED_CLASSES - 1, 1))
    return (rgba[..., :3] * 255.0).astype(np.uint8)


@dataclass
class UploadedImage:
    upload_id: str
    filename: str
    image_chw: torch.Tensor  # (3, H, W) on CPU

    @property
    def full_res_shape(self) -> tuple[int, int]:
        return int(self.image_chw.shape[1]), int(self.image_chw.shape[2])


@dataclass
class CachedRun:
    folder: str
    heatmaps: np.ndarray
    rooms: np.ndarray
    icons: np.ndarray
    rooms_seg: np.ndarray
    icons_seg: np.ndarray
    full_res_shape: tuple[int, int]
    input_bgr: np.ndarray
    postproc: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    wall_postproc: dict[tuple[float, float, float, float], dict] = field(default_factory=dict)


class VizEngine:
    def __init__(
        self, data_path: str | None = None, run_roots: tuple[str, ...] | None = None
    ):
        self.data_path = (data_path or DEFAULT_DATA_PATH).rstrip(os.sep) + os.sep
        self.run_roots = run_roots or DEFAULT_RUN_ROOTS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._presets = self._load_presets()
        self._lmdb_env = None
        self._loader: FullLoader | None = None
        self._models: dict[str, torch.nn.Module] = {}
        self._cache: dict[tuple, CachedRun] = {}
        self._uploads: dict[str, UploadedImage] = {}
        self._log = logging.getLogger("viz_web")

    @staticmethod
    def _load_presets():
        with open(PRESETS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_plans(self):
        return self._presets

    def list_models(self):
        models = []
        for root_name in self.run_roots:
            root = os.path.join(_ROOT, root_name)
            if not os.path.isdir(root):
                continue
            for name in sorted(os.listdir(root), reverse=True):
                run_dir = os.path.join(root, name)
                ckpt = os.path.join(run_dir, "model_best_val_loss.pkl")
                if os.path.isfile(ckpt):
                    models.append(
                        {
                            "id": f"{root_name}/{name}",
                            "label": f"{root_name} / {name}",
                            "run_dir": run_dir,
                        }
                    )
        return models

    def _ensure_loader(self):
        if self._loader is not None:
            return
        lmdb_path = os.path.join(self.data_path.rstrip(os.sep), "cubi_lmdb")
        self._lmdb_env = lmdb.open(
            lmdb_path,
            readonly=True,
            max_readers=8,
            lock=False,
            readahead=True,
            meminit=False,
        )
        self._loader = FullLoader(
            self.data_path,
            "test.txt",
            self._lmdb_env,
            augmentations=DictToTensor(),
        )

    def _get_model(self, model_id: str) -> torch.nn.Module:
        if model_id in self._models:
            return self._models[model_id]
        run_dir = os.path.join(_ROOT, model_id)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Unknown model: {model_id}")
        args = load_eval_args(run_dir)
        args.data_path = self.data_path
        model = cubi_casa5k_full_model(args, self._log)
        model.eval()
        self._models[model_id] = model
        return model

    def _load_sample(self, dataset_index: int):
        self._ensure_loader()
        sample = self._loader[dataset_index]
        image = sample["image"].unsqueeze(0)
        label = sample["label"]
        folder = sample["folder"]
        return image, label, folder

    def _cache_key(self, model_id: str, plan_id: int | None, upload_id: str | None):
        if upload_id:
            return ("upload", upload_id, model_id)
        if plan_id is None:
            raise ValueError("plan_id or upload_id required")
        return ("preset", plan_id, model_id)

    def store_upload(self, data: bytes, filename: str = "upload") -> UploadedImage:
        image_chw = _image_bytes_to_model_tensor(data)
        upload_id = uuid.uuid4().hex
        entry = UploadedImage(
            upload_id=upload_id,
            filename=filename or "upload",
            image_chw=image_chw,
        )
        self._uploads[upload_id] = entry
        # Drop cached inference for this upload id if re-uploaded under same id (new id each time).
        drop = [k for k in self._cache if k[0] == "upload" and k[1] == upload_id]
        for k in drop:
            del self._cache[k]
        return entry

    def get_upload(self, upload_id: str) -> UploadedImage:
        if upload_id not in self._uploads:
            raise KeyError(f"Unknown upload_id: {upload_id}")
        return self._uploads[upload_id]

    def get_input_png(
        self, plan_id: int | None = None, upload_id: str | None = None
    ) -> bytes:
        if upload_id:
            bgr = _tensor_to_bgr_uint8(self.get_upload(upload_id).image_chw)
        else:
            preset = self._presets[plan_id]
            image, _, _ = self._load_sample(preset["dataset_index"])
            bgr = _tensor_to_bgr_uint8(image[0])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return _png_bytes(rgb)

    def _forward_image(self, image_chw: torch.Tensor, model_id: str) -> CachedRun:
        full_res_shape = (int(image_chw.shape[1]), int(image_chw.shape[2]))
        images = image_chw.unsqueeze(0).to(self.device)
        model = self._get_model(model_id)
        with torch.no_grad():
            outputs = predict_at_resolution(model, images, full_res_shape)
        outputs_cpu = outputs.detach().cpu().float()
        heatmaps, rooms, icons = post_prosessing.split_prediction(
            outputs_cpu, full_res_shape, INPUT_SLICE
        )
        rooms_seg = np.argmax(rooms, axis=0).astype(np.int64)
        icons_seg = np.argmax(icons, axis=0).astype(np.int64)
        input_bgr = _tensor_to_bgr_uint8(image_chw)
        return CachedRun(
            folder="(upload)",
            heatmaps=heatmaps,
            rooms=rooms,
            icons=icons,
            rooms_seg=rooms_seg,
            icons_seg=icons_seg,
            full_res_shape=full_res_shape,
            input_bgr=input_bgr,
        )

    def run_inference(
        self,
        model_id: str,
        plan_id: int | None = None,
        upload_id: str | None = None,
    ) -> CachedRun:
        key = self._cache_key(model_id, plan_id, upload_id)
        if key in self._cache:
            return self._cache[key]

        if upload_id:
            image_chw = self.get_upload(upload_id).image_chw
            folder = f"upload:{self._uploads[upload_id].filename}"
        else:
            preset = self._presets[plan_id]
            images, label, folder = self._load_sample(preset["dataset_index"])
            image_chw = images[0].cpu()
            folder = folder

        cached = self._forward_image(image_chw, model_id)
        cached.folder = folder
        self._cache[key] = cached
        return cached

    def clear_cache(
        self,
        plan_id: int | None = None,
        upload_id: str | None = None,
        model_id: str | None = None,
    ):
        if plan_id is None and upload_id is None and model_id is None:
            self._cache.clear()
            return
        drop = []
        for k in self._cache:
            kind, src_id, m = k
            if model_id is not None and m != model_id:
                continue
            if upload_id is not None and (kind != "upload" or src_id != upload_id):
                continue
            if plan_id is not None and (kind != "preset" or src_id != plan_id):
                continue
            drop.append(k)
        for k in drop:
            del self._cache[k]

    def artifact_png(
        self,
        model_id: str,
        name: str,
        plan_id: int | None = None,
        upload_id: str | None = None,
    ) -> bytes:
        run = self.run_inference(model_id, plan_id=plan_id, upload_id=upload_id)
        if name == "input":
            rgb = cv2.cvtColor(run.input_bgr, cv2.COLOR_BGR2RGB)
            return _png_bytes(rgb)
        if name == "wall_hm":
            lo, hi = CHANNEL_GROUPS["wall"]
            return _png_bytes(_heatmap_max_rgb(run.heatmaps[lo:hi]))
        if name == "opening_hm":
            lo, hi = CHANNEL_GROUPS["opening"]
            return _png_bytes(_heatmap_max_rgb(run.heatmaps[lo:hi]))
        if name == "icon_hm":
            lo, hi = CHANNEL_GROUPS["icon"]
            return _png_bytes(_heatmap_max_rgb(run.heatmaps[lo:hi]))
        if name == "room_seg":
            return _png_bytes(_seg_rgb(run.rooms_seg, N_ROOM_CLASSES))
        if name == "icon_seg":
            return _png_bytes(_seg_rgb(run.icons_seg, N_ICON_CLASSES))
        if name == "room_entropy":
            return _png_bytes(_entropy_rgb_from_probs(run.rooms, N_ROOM_CLASSES))
        if name == "icon_entropy":
            return _png_bytes(_entropy_rgb_from_probs(run.icons, N_ICON_CLASSES))
        raise KeyError(name)

    def postproc_png(
        self,
        model_id: str,
        threshold: float,
        plan_id: int | None = None,
        upload_id: str | None = None,
    ) -> bytes:
        run = self.run_inference(model_id, plan_id=plan_id, upload_id=upload_id)
        thr = round(float(threshold), 3)
        if thr not in run.postproc:
            pol_rooms, pol_icons = run_postproc_mini(
                run.heatmaps,
                run.rooms,
                run.icons,
                run.full_res_shape,
                thr,
            )
            combined = build_combined_map(pol_rooms, pol_icons)
            run.postproc[thr] = (pol_rooms, pol_icons, combined)
        _, _, combined = run.postproc[thr]
        return _png_bytes(_combined_rgb(combined))

    def wall_network_result(
        self,
        model_id: str,
        threshold: float,
        axis_bias: float = 0.35,
        snap_align: float = 0.0,
        wall_evidence: float = 0.9,
        min_wall_fraction: float = 0.5,
        plan_id: int | None = None,
        upload_id: str | None = None,
    ) -> dict:
        """New non-Manhattan wall-graph post-process (post_process_wall.py), applied
        directly to the existing train_full model's 21-channel heatmap prediction --
        no separately trained train_wall.py checkpoint required. Returns the raw
        result dict (points/opening_points/wall_segments/openings); see
        postproc_wall_png for the rendered-PNG version and render_wall_network_bgr.
        axis_bias: strength of the horizontal/vertical preference (angle_bonus_weight).
        snap_align: 0 disables; otherwise forces points connected by an accepted
        edge onto a shared exact x/y, gated by an absolute pixel deviation (not a
        fixed angle -- see snap_axis_aligned_points for why) of up to this many px.
        wall_evidence: strength of real pixel evidence from this same model's own
        room/wall segmentation (run.rooms_seg) -- candidates under min_wall_fraction
        are hard-rejected outright; this controls a softer preference for
        more-covered candidates on top of that.
        min_wall_fraction: the hard-reject threshold for wall_evidence itself."""
        run = self.run_inference(model_id, plan_id=plan_id, upload_id=upload_id)
        key = (
            round(float(threshold), 3), round(float(axis_bias), 3),
            round(float(snap_align), 2), round(float(wall_evidence), 3),
            round(float(min_wall_fraction), 3),
        )
        if key not in run.wall_postproc:
            wall_heatmaps = remap_prediction_to_wall_heatmaps(run.heatmaps)
            run.wall_postproc[key] = build_wall_network(
                wall_heatmaps, point_threshold=key[0], opening_threshold=key[0],
                angle_bonus_weight=key[1], snap_axis_tolerance_px=key[2],
                room_seg=run.rooms_seg, wall_class_id=WALL_CLASS, wall_evidence_weight=key[3],
                min_wall_fraction=key[4],
            )
        return run.wall_postproc[key]

    def postproc_wall_png(
        self,
        model_id: str,
        threshold: float,
        axis_bias: float = 0.35,
        snap_align: float = 0.0,
        wall_evidence: float = 0.9,
        min_wall_fraction: float = 0.5,
        plan_id: int | None = None,
        upload_id: str | None = None,
    ) -> bytes:
        run = self.run_inference(model_id, plan_id=plan_id, upload_id=upload_id)
        result = self.wall_network_result(
            model_id, threshold, axis_bias=axis_bias, snap_align=snap_align,
            wall_evidence=wall_evidence, min_wall_fraction=min_wall_fraction,
            plan_id=plan_id, upload_id=upload_id,
        )
        overlay_bgr = render_wall_network_bgr(run.input_bgr, result)
        return _png_bytes(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))

    def skeleton_overlay_png(
        self,
        model_id: str,
        threshold: float,
        base: str = "map",
        seg_alpha: float = 0.5,
        axis_bias: float = 0.35,
        snap_align: float = 0.0,
        wall_evidence: float = 0.9,
        min_wall_fraction: float = 0.5,
        plan_id: int | None = None,
        upload_id: str | None = None,
    ) -> bytes:
        """One image: the computed wall skeleton drawn on top of `base`, so the
        skeleton can be checked directly against the exact segmentation it was
        scored against, not just the raw plan. base: "map" (input plan only),
        "segmentation" (room_seg only), or "both" (segmentation alpha-blended over
        the map at seg_alpha, then skeleton drawn on top of that)."""
        run = self.run_inference(model_id, plan_id=plan_id, upload_id=upload_id)
        result = self.wall_network_result(
            model_id, threshold, axis_bias=axis_bias, snap_align=snap_align,
            wall_evidence=wall_evidence, min_wall_fraction=min_wall_fraction,
            plan_id=plan_id, upload_id=upload_id,
        )
        if base == "segmentation":
            base_bgr = cv2.cvtColor(_seg_rgb(run.rooms_seg, N_ROOM_CLASSES), cv2.COLOR_RGB2BGR)
        elif base == "both":
            seg_bgr = cv2.cvtColor(_seg_rgb(run.rooms_seg, N_ROOM_CLASSES), cv2.COLOR_RGB2BGR)
            a = max(0.0, min(1.0, float(seg_alpha)))
            base_bgr = cv2.addWeighted(seg_bgr, a, run.input_bgr, 1.0 - a, 0.0)
        else:
            base_bgr = run.input_bgr
        overlay_bgr = render_wall_network_bgr(base_bgr, result)
        return _png_bytes(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
