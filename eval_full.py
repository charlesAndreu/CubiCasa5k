"""
Test-set evaluation for the full Cubicasa5k model
(21 heatmap channels + 3 room logits + 4 icon logits = 28 output channels).

Mini room layout (3 classes): 0 = outside, 1 = walls, 2 = inside
Mini icon layout (4 classes): 0 = empty,   1 = window, 2 = door, 3 = others

Outputs per run (saved under <run_dir>/):
  * eval.csv — 8 rows: raw no_rotation / rotation, then post-processing @ 0.30/0.35/0.40
  * eval.json — same metrics in JSON form
  * confusion_{room|icon}_{raw|postproc}.png / _raw.json (no-TTA raw + TTA @ 0.35)
  * eval_samples_seed42/ — 3 random samples (TTA, postproc threshold 0.35)

Test images are processed at native (LMDB) resolution. The fully-convolutional
Furukawa model accepts arbitrary sizes; predictions are bilinearly resampled
back to the ground-truth resolution via post_prosessing.split_prediction
before metrics and post-processing run.
"""

import csv
import json
import logging
import math
import os
import sys
from types import SimpleNamespace

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataloader import build_cubicasa5k_full_eval_dataloaders_native_res
from eval_simple import (
    _save_combined_segmentation_map_png,
    _save_entropy_heatmap_png,
    _save_segmentation_map_png,
    _tab20_segmentation_colors,
    _tensor_to_bgr_uint8,
    _to_jsonable,
    save_confusion_matrix_artifacts,
)
from floortrans import post_prosessing
from floortrans.loaders.augmentations import RotateNTurns
from floortrans.metrics import polygons_to_tensor, runningScore
from floortrans.post_prosessing import (
    get_icon_polygon,
    get_junction_points,
    get_opening_polygon,
    get_polygon_class,
    get_rectangle_polygons,
    get_wall_polygon,
    merge_rectangles,
    remove_overlapping_openings,
)
from model import cubi_casa5k_full_model
from train_full import TRAIN_FULL_CONFIG_DEFAULTS


N_HEATMAPS = 21
N_ROOM_CLASSES = 3  # 0=outside, 1=walls, 2=inside
N_ICON_CLASSES = 4  # 0=empty, 1=window, 2=door, 3=others
INPUT_SLICE = [N_HEATMAPS, N_ROOM_CLASSES, N_ICON_CLASSES]

WALL_CLASS = 1
WINDOW_CLASS = 1
DOOR_CLASS = 2
COMBINED_CLASSES = 5  # 0=outside, 1=walls, 2=inside, 3=windows, 4=doors

# Post-processing peak thresholds (see diagnose_heatmaps.py).
EVAL_THRESHOLDS = [0.30, 0.35, 0.40]
VIS_POSTPROC_THRESHOLD = 0.35
ROTATIONS = [(0, 0), (1, -1), (2, 2), (-1, 1)]

def _eval_row_specs():
    """8 evaluation rows: 2 raw + 3 thresholds × 2 (no_rotation / rotation + post_processing)."""
    specs = [
        ("no_rotation", None, False, False),
        ("rotation", None, False, True),
    ]
    for thr in EVAL_THRESHOLDS:
        specs.append((f"{thr:.2f} no_rotation + post_processing", thr, True, False))
        specs.append((f"{thr:.2f} rotation + post_processing", thr, True, True))
    return specs


def get_polygons_mini(predictions, threshold, all_opening_types):
    """
    Adapted from floortrans.post_prosessing.get_polygons for the mini scheme:
      * room walls live in a single class (WALL_CLASS = 1)
      * icon scheme: empty=0, window=1, door=2, others=3
    Mirrors the original control flow, only the wall_layers list changes.
    """
    heatmaps, room_seg, icon_seg = predictions
    height = icon_seg.shape[1]
    width = icon_seg.shape[2]

    point_orientations = [
        [(2,), (3,), (0,), (1,)],
        [(0, 3), (0, 1), (1, 2), (2, 3)],
        [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)],
        [(0, 1, 2, 3)],
    ]
    orientation_ranges = [
        [width, 0, 0, 0],
        [width, height, width, 0],
        [width, height, 0, height],
        [0, height, 0, 0],
    ]

    wall_heatmaps = heatmaps[:13]
    wall_layers = [WALL_CLASS]
    (
        walls,
        wall_types,
        wall_points,
        wall_lines,
        wall_point_orientation_lines_map,
    ) = get_wall_polygon(
        wall_heatmaps,
        room_seg,
        threshold,
        wall_layers,
        point_orientations,
        orientation_ranges,
    )

    icons, icon_types = get_icon_polygon(
        heatmaps, icon_seg, threshold, point_orientations, orientation_ranges
    )

    openings, opening_types = get_opening_polygon(
        heatmaps,
        walls,
        icon_seg,
        wall_points,
        wall_lines,
        wall_point_orientation_lines_map,
        threshold,
        point_orientations,
        orientation_ranges,
        all_opening_types,
    )

    junction_points = get_junction_points(wall_points, wall_lines)
    grid_polygons = get_rectangle_polygons(junction_points, (height, width))

    # zero out wall channel(s) before argmax so room-class assignment ignores walls
    c, h, w = room_seg.shape
    for i in range(c):
        if i in wall_layers:
            room_seg[i] = np.zeros((h, w))

    room_seg_2D = np.argmax(room_seg, axis=0)
    room_types = []
    grid_polygons_new = []
    for pol in grid_polygons:
        room_class = get_polygon_class(pol, room_seg_2D)
        if room_class is not None:
            grid_polygons_new.append(pol)
            room_types.append({"type": "room", "class": room_class})

    room_polygons, room_types = merge_rectangles(grid_polygons_new, room_types)

    polygons = np.concatenate([walls, icons, openings])
    types = wall_types + icon_types + opening_types

    classes = {"door": [DOOR_CLASS], "window": [WINDOW_CLASS]}
    if len(polygons) > 0:
        polygons, types = remove_overlapping_openings(polygons, types, classes)

    return polygons, types, room_polygons, room_types


def build_combined_map(pol_rooms, pol_icons):
    """5-class map: rooms (0/1/2) overlaid with windows (→3) and doors (→4)."""
    combined = pol_rooms.astype(np.int64).copy()
    combined[pol_icons == WINDOW_CLASS] = 3
    combined[pol_icons == DOOR_CLASS] = 4
    return combined


def _resize_pred(pred, target_hw):
    if pred.shape[-2:] == target_hw:
        return pred
    return F.interpolate(
        pred, size=target_hw, mode="bilinear", align_corners=True
    )


def predict_no_rotation(model, images, target_hw):
    """Single forward; returns (1, C, H, W) on device."""
    pred = model(images)
    return _resize_pred(pred, target_hw)


def predict_with_rotation_tta(model, images, target_hw, n_channels, device):
    """4-rotation TTA mean, matching floortrans.metrics.get_evaluation_tensors."""
    rot = RotateNTurns()
    h, w = target_hw
    acc = torch.zeros(len(ROTATIONS), n_channels, h, w, device=device)
    for i, (forward, back) in enumerate(ROTATIONS):
        rot_img = rot(images, "tensor", forward)
        pred = model(rot_img)
        pred = rot(pred, "tensor", back)
        pred = rot(pred, "points", back)
        pred = _resize_pred(pred, (h, w))
        acc[i] = pred[0]
    return acc.mean(dim=0, keepdim=True)


def run_postproc_mini(heatmaps, rooms, icons, full_res_shape, threshold):
    """Polygon post-processing; returns (pol_rooms, pol_icons) or (None, None)."""
    polygons, types, room_polygons, room_types = get_polygons_mini(
        (heatmaps, rooms.copy(), icons),
        threshold=threshold,
        all_opening_types=[WINDOW_CLASS, DOOR_CLASS],
    )
    predicted_classes = polygons_to_tensor(
        polygons,
        types,
        room_polygons,
        room_types,
        full_res_shape,
        split=[N_ROOM_CLASSES, N_ICON_CLASSES],
    )
    pol_rooms = np.argmax(predicted_classes[:N_ROOM_CLASSES], axis=0).astype(
        np.int64
    )
    pol_icons = np.argmax(predicted_classes[N_ROOM_CLASSES:], axis=0).astype(
        np.int64
    )
    return pol_rooms, pol_icons


def _head_metrics_dict(score, class_metrics, prefix, n_classes):
    """Aggregate + per-class IoU and accuracy for one segmentation head."""
    out = {
        f"{prefix}_overall_acc": score["Overall Acc"],
        f"{prefix}_mean_acc": score["Mean Acc"],
        f"{prefix}_freqw_acc": score["FreqW Acc"],
        f"{prefix}_mean_iou": score["Mean IoU"],
    }
    for c in range(n_classes):
        out[f"{prefix}_iou_{c}"] = class_metrics["Class IoU"][str(c)]
        out[f"{prefix}_acc_{c}"] = class_metrics["Class Acc"][str(c)]
    return out


def build_eval_rows(
    running_room_raw_no_tta,
    running_icon_raw_no_tta,
    running_room_raw_tta,
    running_icon_raw_tta,
    running_room_pp_no_tta,
    running_icon_pp_no_tta,
    running_room_pp_tta,
    running_icon_pp_tta,
    post_failed_no_tta,
    post_failed_tta,
    mean_room_entropy_no_tta,
    mean_room_entropy_tta,
    mean_icon_entropy_no_tta,
    mean_icon_entropy_tta,
):
    rows = []
    for mode_label, thr, use_pp, use_tta in _eval_row_specs():
        row = {"mode": mode_label}
        if thr is not None:
            row["postproc_threshold"] = thr

        if use_pp:
            if use_tta:
                rr, ir = running_room_pp_tta[thr], running_icon_pp_tta[thr]
                row["post_failed"] = post_failed_tta[thr]
            else:
                rr, ir = running_room_pp_no_tta[thr], running_icon_pp_no_tta[thr]
                row["post_failed"] = post_failed_no_tta[thr]
            row["mean_room_entropy"] = None
            row["mean_icon_entropy"] = None
        else:
            if use_tta:
                rr, ir = running_room_raw_tta, running_icon_raw_tta
                row["mean_room_entropy"] = mean_room_entropy_tta
                row["mean_icon_entropy"] = mean_icon_entropy_tta
            else:
                rr, ir = running_room_raw_no_tta, running_icon_raw_no_tta
                row["mean_room_entropy"] = mean_room_entropy_no_tta
                row["mean_icon_entropy"] = mean_icon_entropy_no_tta
            row["post_failed"] = None

        room_score, room_cm = rr.get_scores()
        icon_score, icon_cm = ir.get_scores()
        row.update(_head_metrics_dict(room_score, room_cm, "room", N_ROOM_CLASSES))
        row.update(_head_metrics_dict(icon_score, icon_cm, "icon", N_ICON_CLASSES))
        rows.append(row)
    return rows


def _eval_csv_columns():
    col_order = ["mode", "postproc_threshold"]
    for prefix, n in (("room", N_ROOM_CLASSES), ("icon", N_ICON_CLASSES)):
        col_order.extend(
            [
                f"{prefix}_overall_acc",
                f"{prefix}_mean_acc",
                f"{prefix}_freqw_acc",
                f"{prefix}_mean_iou",
            ]
        )
        col_order.extend([f"{prefix}_iou_{c}" for c in range(n)])
        col_order.extend([f"{prefix}_acc_{c}" for c in range(n)])
    col_order.extend(["mean_room_entropy", "mean_icon_entropy", "post_failed"])
    return col_order


def write_eval_csv(path, rows):
    columns = _eval_csv_columns()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in columns}
            for k, v in out.items():
                if v is None:
                    out[k] = ""
            writer.writerow(out)


def _entropy_from_probs(probs_chw, n_classes):
    """Shannon entropy normalized to [0, 1] from already-softmaxed probs (C, H, W)."""
    p = torch.as_tensor(probs_chw, dtype=torch.float32).clamp(min=1e-12)
    entropy = -(p * p.log()).sum(dim=0)
    return entropy / math.log(n_classes)


def save_sample_pngs(
    out_dir,
    idx,
    image_chw,
    rooms_seg,
    icons_seg,
    room_entropy_hw,
    icon_entropy_hw,
    pol_rooms,
    pol_icons,
    combined,
):
    stem = f"sample_{idx:05d}"
    cv2.imwrite(
        os.path.join(out_dir, f"{stem}_input.png"), _tensor_to_bgr_uint8(image_chw)
    )

    _save_segmentation_map_png(
        os.path.join(out_dir, f"{stem}_room_segmentation.png"),
        rooms_seg,
        N_ROOM_CLASSES,
    )
    _save_segmentation_map_png(
        os.path.join(out_dir, f"{stem}_icon_segmentation.png"),
        icons_seg,
        N_ICON_CLASSES,
    )

    wall_mask = (rooms_seg == WALL_CLASS).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(out_dir, f"{stem}_wall.png"), wall_mask)

    _save_entropy_heatmap_png(
        os.path.join(out_dir, f"{stem}_room_entropy_all_classes.png"),
        room_entropy_hw,
        "Room entropy (all classes)",
    )
    _save_entropy_heatmap_png(
        os.path.join(out_dir, f"{stem}_icon_entropy_all_classes.png"),
        icon_entropy_hw,
        "Icon entropy (all classes)",
    )

    if pol_rooms is not None:
        _save_segmentation_map_png(
            os.path.join(out_dir, f"{stem}_room_segmentation_postproc.png"),
            pol_rooms,
            N_ROOM_CLASSES,
        )
    if pol_icons is not None:
        _save_segmentation_map_png(
            os.path.join(out_dir, f"{stem}_icon_segmentation_postproc.png"),
            pol_icons,
            N_ICON_CLASSES,
        )
    if combined is not None:
        room_colors = _tab20_segmentation_colors(N_ROOM_CLASSES)
        icon_colors = _tab20_segmentation_colors(N_ICON_CLASSES)
        _save_combined_segmentation_map_png(
            os.path.join(out_dir, f"{stem}_combined_postproc.png"),
            combined,
            COMBINED_CLASSES,
            room_colors,
            icon_colors,
            WINDOW_CLASS,
            DOOR_CLASS,
        )


class FullSegEvaluator:

    def __init__(self, args):
        self.args = args
        self.input_slice = INPUT_SLICE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger("eval")

    def dataloader_setup(self):
        return build_cubicasa5k_full_eval_dataloaders_native_res(self.args, self.device)

    def model_setup(self):
        # cubi_casa5k_full_model uses args.resume_from when set to load checkpoint
        return cubi_casa5k_full_model(self.args, self.logger)

    def evaluate(self, results_dir):
        testloader = self.dataloader_setup()
        self.model = self.model_setup()
        self.model.eval()
        n_channels = sum(self.input_slice)

        running_room_raw_no_tta = runningScore(N_ROOM_CLASSES)
        running_icon_raw_no_tta = runningScore(N_ICON_CLASSES)
        running_room_raw_tta = runningScore(N_ROOM_CLASSES)
        running_icon_raw_tta = runningScore(N_ICON_CLASSES)

        running_room_pp_no_tta = {t: runningScore(N_ROOM_CLASSES) for t in EVAL_THRESHOLDS}
        running_icon_pp_no_tta = {t: runningScore(N_ICON_CLASSES) for t in EVAL_THRESHOLDS}
        running_room_pp_tta = {t: runningScore(N_ROOM_CLASSES) for t in EVAL_THRESHOLDS}
        running_icon_pp_tta = {t: runningScore(N_ICON_CLASSES) for t in EVAL_THRESHOLDS}

        post_failed_no_tta = {t: 0 for t in EVAL_THRESHOLDS}
        post_failed_tta = {t: 0 for t in EVAL_THRESHOLDS}

        n_samples = len(testloader.dataset)
        rng = np.random.default_rng(42)
        k_vis = min(3, n_samples)
        vis_indices = set(rng.choice(n_samples, size=k_vis, replace=False).tolist())
        vis_dir = os.path.join(results_dir, "eval_samples_seed42")
        os.makedirs(vis_dir, exist_ok=True)

        entropy_room_no_tta = entropy_icon_no_tta = 0.0
        entropy_room_tta = entropy_icon_tta = 0.0
        entropy_pixel_count = 0

        with torch.no_grad():
            global_idx = 0
            for samples in tqdm(
                testloader,
                total=len(testloader),
                ncols=80,
                leave=False,
                desc="Eval",
            ):
                images = samples["image"].to(
                    self.device, non_blocking=(self.device.type == "cuda")
                )
                labels = samples["label"]  # CPU, (1, 23, H_full, W_full)
                full_h, full_w = labels.shape[2], labels.shape[3]
                full_res_shape = (full_h, full_w)

                outputs_no_tta = predict_no_rotation(
                    self.model, images, full_res_shape
                )
                outputs_tta = predict_with_rotation_tta(
                    self.model,
                    images,
                    full_res_shape,
                    n_channels,
                    self.device,
                )

                rooms_gt = labels[0, N_HEATMAPS].long().numpy()
                icons_gt = labels[0, N_HEATMAPS + 1].long().numpy()

                preds = (
                    ("no_tta", outputs_no_tta),
                    ("tta", outputs_tta),
                )
                for tag, outputs in preds:
                    outputs_cpu = outputs.detach().cpu().float()
                    heatmaps, rooms, icons = post_prosessing.split_prediction(
                        outputs_cpu, full_res_shape, self.input_slice
                    )
                    rooms_seg = np.argmax(rooms, axis=0)
                    icons_seg = np.argmax(icons, axis=0)

                    if tag == "no_tta":
                        running_room_raw_no_tta.update([rooms_gt], [rooms_seg])
                        running_icon_raw_no_tta.update([icons_gt], [icons_seg])
                        room_ent = _entropy_from_probs(rooms, N_ROOM_CLASSES)
                        icon_ent = _entropy_from_probs(icons, N_ICON_CLASSES)
                        entropy_room_no_tta += float(room_ent.sum().item())
                        entropy_icon_no_tta += float(icon_ent.sum().item())
                    else:
                        running_room_raw_tta.update([rooms_gt], [rooms_seg])
                        running_icon_raw_tta.update([icons_gt], [icons_seg])
                        room_ent = _entropy_from_probs(rooms, N_ROOM_CLASSES)
                        icon_ent = _entropy_from_probs(icons, N_ICON_CLASSES)
                        entropy_room_tta += float(room_ent.sum().item())
                        entropy_icon_tta += float(icon_ent.sum().item())

                    pp_room = running_room_pp_no_tta if tag == "no_tta" else running_room_pp_tta
                    pp_icon = running_icon_pp_no_tta if tag == "no_tta" else running_icon_pp_tta
                    pf = post_failed_no_tta if tag == "no_tta" else post_failed_tta

                    for thr in EVAL_THRESHOLDS:
                        try:
                            pol_rooms, pol_icons = run_postproc_mini(
                                heatmaps, rooms, icons, full_res_shape, thr
                            )
                            pp_room[thr].update([rooms_gt], [pol_rooms])
                            pp_icon[thr].update([icons_gt], [pol_icons])
                        except Exception as e:
                            pf[thr] += 1
                            self.logger.warning(
                                "Post-processing failed sample %d (%s thr=%.2f): %s",
                                global_idx,
                                tag,
                                thr,
                                e,
                            )
                            pol_rooms = pol_icons = None

                        if (
                            global_idx in vis_indices
                            and tag == "tta"
                            and thr == VIS_POSTPROC_THRESHOLD
                        ):
                            combined = (
                                build_combined_map(pol_rooms, pol_icons)
                                if pol_rooms is not None
                                else None
                            )
                            save_sample_pngs(
                                vis_dir,
                                global_idx,
                                images[0],
                                rooms_seg,
                                icons_seg,
                                room_ent.cpu().numpy(),
                                icon_ent.cpu().numpy(),
                                pol_rooms,
                                pol_icons,
                                combined,
                            )

                entropy_pixel_count += room_ent.numel()
                global_idx += 1

        denom = max(1, entropy_pixel_count)
        eval_rows = build_eval_rows(
            running_room_raw_no_tta,
            running_icon_raw_no_tta,
            running_room_raw_tta,
            running_icon_raw_tta,
            running_room_pp_no_tta,
            running_icon_pp_no_tta,
            running_room_pp_tta,
            running_icon_pp_tta,
            post_failed_no_tta,
            post_failed_tta,
            entropy_room_no_tta / denom,
            entropy_room_tta / denom,
            entropy_icon_no_tta / denom,
            entropy_icon_tta / denom,
        )

        vis_thr = VIS_POSTPROC_THRESHOLD
        return {
            "eval_rows": eval_rows,
            "n_samples": n_samples,
            "vis_dir": vis_dir,
            "confusion_room_raw": running_room_raw_no_tta.confusion_matrix.copy(),
            "confusion_icon_raw": running_icon_raw_no_tta.confusion_matrix.copy(),
            "confusion_room_raw_tta": running_room_raw_tta.confusion_matrix.copy(),
            "confusion_icon_raw_tta": running_icon_raw_tta.confusion_matrix.copy(),
            "confusion_room_postproc": running_room_pp_tta[vis_thr].confusion_matrix.copy(),
            "confusion_icon_postproc": running_icon_pp_tta[vis_thr].confusion_matrix.copy(),
            "running_room_raw_no_tta": running_room_raw_no_tta,
            "running_icon_raw_no_tta": running_icon_raw_no_tta,
            "running_room_raw_tta": running_room_raw_tta,
            "running_icon_raw_tta": running_icon_raw_tta,
            "running_room_pp_no_tta": running_room_pp_no_tta,
            "running_icon_pp_no_tta": running_icon_pp_no_tta,
            "running_room_pp_tta": running_room_pp_tta,
            "running_icon_pp_tta": running_icon_pp_tta,
            "post_failed_no_tta": post_failed_no_tta,
            "post_failed_tta": post_failed_tta,
            "mean_room_entropy_no_tta": entropy_room_no_tta / denom,
            "mean_icon_entropy_no_tta": entropy_icon_no_tta / denom,
            "mean_room_entropy_tta": entropy_room_tta / denom,
            "mean_icon_entropy_tta": entropy_icon_tta / denom,
        }


def load_eval_args(run_dir):
    """Build args namespace from <run_dir>/args.json + defaults; point resume_from
    to the trained checkpoint so cubi_casa5k_full_model loads its weights."""
    with open(os.path.join(run_dir, "args.json"), "r") as f:
        run_args = json.load(f)
    merged = {**TRAIN_FULL_CONFIG_DEFAULTS, **run_args}
    weights_path = os.path.join(run_dir, "model_best_val_loss.pkl")
    merged["weights"] = weights_path
    merged["resume_from"] = weights_path  # makes CubiCasa5KFurukawa load the checkpoint
    merged["furukawa_weights"] = None
    merged["log_path"] = run_dir + "/eval.log"
    merged["num_workers"] = 4
    return SimpleNamespace(**merged)


def _json_block(runner):
    score, class_metrics = runner.get_scores()
    return {
        "score": _to_jsonable(score),
        "class_iou": _to_jsonable(class_metrics),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python eval_full.py <run_dir>")
    run_dir = sys.argv[1]
    args = load_eval_args(run_dir)

    evaluator = FullSegEvaluator(args)
    results = evaluator.evaluate(results_dir=run_dir)

    csv_path = os.path.join(run_dir, "eval.csv")
    write_eval_csv(csv_path, results["eval_rows"])

    cm_paths = {}
    for stem, cm in [
        ("confusion_room_raw", results["confusion_room_raw"]),
        ("confusion_icon_raw", results["confusion_icon_raw"]),
        ("confusion_room_raw_tta", results["confusion_room_raw_tta"]),
        ("confusion_icon_raw_tta", results["confusion_icon_raw_tta"]),
        ("confusion_room_postproc", results["confusion_room_postproc"]),
        ("confusion_icon_postproc", results["confusion_icon_postproc"]),
    ]:
        img_path, raw_path = save_confusion_matrix_artifacts(run_dir, cm, stem=stem)
        cm_paths[stem] = (img_path, raw_path)

    json_rows = []
    for mode_label, thr, use_pp, use_tta in _eval_row_specs():
        entry = {"mode": mode_label}
        if use_pp:
            entry["postproc_threshold"] = thr
            if use_tta:
                entry["room"] = _json_block(results["running_room_pp_tta"][thr])
                entry["icon"] = _json_block(results["running_icon_pp_tta"][thr])
                entry["post_failed"] = results["post_failed_tta"][thr]
            else:
                entry["room"] = _json_block(results["running_room_pp_no_tta"][thr])
                entry["icon"] = _json_block(results["running_icon_pp_no_tta"][thr])
                entry["post_failed"] = results["post_failed_no_tta"][thr]
        else:
            if use_tta:
                entry["room"] = _json_block(results["running_room_raw_tta"])
                entry["icon"] = _json_block(results["running_icon_raw_tta"])
                entry["mean_room_entropy_all_classes"] = results[
                    "mean_room_entropy_tta"
                ]
                entry["mean_icon_entropy_all_classes"] = results[
                    "mean_icon_entropy_tta"
                ]
            else:
                entry["room"] = _json_block(results["running_room_raw_no_tta"])
                entry["icon"] = _json_block(results["running_icon_raw_no_tta"])
                entry["mean_room_entropy_all_classes"] = results[
                    "mean_room_entropy_no_tta"
                ]
                entry["mean_icon_entropy_all_classes"] = results[
                    "mean_icon_entropy_no_tta"
                ]
        json_rows.append(entry)

    json_payload = {
        "thresholds": EVAL_THRESHOLDS,
        "rows": json_rows,
        "n_samples": results["n_samples"],
        "vis_postproc_threshold": VIS_POSTPROC_THRESHOLD,
    }

    output_path = os.path.join(run_dir, "eval.json")
    with open(output_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    print(f"Saved evaluation CSV to {csv_path} ({len(results['eval_rows'])} rows)")
    print(f"Saved evaluation results to {output_path}")
    for stem, (img_path, raw_path) in cm_paths.items():
        print(f"  {stem}: {img_path}, {raw_path}")
    print(f"Saved eval samples under {results['vis_dir']}")
