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
from floortrans.loaders.room_icon_loaders import ROOM_MINI_WALL_LAYERS
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
N_ROOM_CLASSES = 4  # 0=background, 1=outside, 2=walls, 3=room
N_ICON_CLASSES = 4  # 0=empty, 1=window, 2=door, 3=others
INPUT_SLICE = [N_HEATMAPS, N_ROOM_CLASSES, N_ICON_CLASSES]

WALL_CLASS = ROOM_MINI_WALL_LAYERS[0]  # 2
WINDOW_CLASS = 1
DOOR_CLASS = 2
COMBINED_CLASSES = 6  # room 0–3 + windows (4) + doors (5)
WINDOW_COMBINED_CLASS = 4
DOOR_COMBINED_CLASS = 5

# Dedicated overlay colors for combined maps (not reused from icon-seg tab20).
COMBINED_WINDOW_RGBA = (0.20, 0.75, 0.25, 1.0)  # green
COMBINED_DOOR_RGBA = (0.95, 0.35, 0.10, 1.0)  # orange


def combined_map_colors(room_class_colors, n_combined_classes=COMBINED_CLASSES):
    """Listed colormap entries: room-mini classes + window + door."""
    n_room = n_combined_classes - 2
    return list(room_class_colors[:n_room]) + [
        COMBINED_WINDOW_RGBA,
        COMBINED_DOOR_RGBA,
    ]


# Post-processing peak thresholds.
EVAL_THRESHOLDS = [0.20, 0.25, 0.30]


def select_best_postproc_threshold(room_runners, icon_runners, thresholds):
    """Pick threshold maximizing mean of room and icon Mean IoU."""
    best_thr = thresholds[0]
    best_score = -1.0
    for thr in thresholds:
        room_score, _ = room_runners[thr].get_scores()
        icon_score, _ = icon_runners[thr].get_scores()
        score = (room_score["Mean IoU"] + icon_score["Mean IoU"]) / 2.0
        if score > best_score:
            best_score = score
            best_thr = thr
    return best_thr


def _eval_row_specs(best_thr):
    """Raw row, then one post-processing row per swept threshold."""
    specs = [("raw", None, False)]
    for thr in EVAL_THRESHOLDS:
        specs.append((f"postproc @ {thr:.2f}", thr, True))
    return specs


def get_polygons_mini(predictions, threshold, all_opening_types):
    """
    Adapted from floortrans.post_prosessing.get_polygons for the mini scheme:
      * room walls live in a single class (WALL_CLASS = 2)
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
    wall_layers = list(ROOM_MINI_WALL_LAYERS)
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
    """6-class map: room mini (0–3) overlaid with windows (4) and doors (5)."""
    combined = pol_rooms.astype(np.int64).copy()
    combined[pol_icons == WINDOW_CLASS] = WINDOW_COMBINED_CLASS
    combined[pol_icons == DOOR_CLASS] = DOOR_COMBINED_CLASS
    return combined


def _resize_pred(pred, target_hw):
    if pred.shape[-2:] == target_hw:
        return pred
    # align_corners=False matches floortrans.post_prosessing.split_prediction
    return F.interpolate(pred, size=target_hw, mode="bilinear", align_corners=False)


def predict_at_resolution(model, images, target_hw):
    """Single forward pass; returns (1, C, H, W) on device."""
    pred = model(images)
    return _resize_pred(pred, target_hw)


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
    pol_rooms = np.argmax(predicted_classes[:N_ROOM_CLASSES], axis=0).astype(np.int64)
    pol_icons = np.argmax(predicted_classes[N_ROOM_CLASSES:], axis=0).astype(np.int64)
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
    best_thr,
    running_room_raw,
    running_icon_raw,
    running_room_pp,
    running_icon_pp,
    mean_room_entropy,
    mean_icon_entropy,
):
    rows = []
    for mode_label, thr, use_pp in _eval_row_specs(best_thr):
        row = {"mode": mode_label}
        if thr is not None:
            row["postproc_threshold"] = thr

        if use_pp:
            rr, ir = running_room_pp[thr], running_icon_pp[thr]
            row["mean_room_entropy"] = None
            row["mean_icon_entropy"] = None
        else:
            rr, ir = running_room_raw, running_icon_raw
            row["mean_room_entropy"] = mean_room_entropy
            row["mean_icon_entropy"] = mean_icon_entropy

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
    col_order.extend(["mean_room_entropy", "mean_icon_entropy"])
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


def save_sample_raw_pngs(
    out_dir, idx, image_chw, rooms_seg, icons_seg, rooms_probs, icons_probs
):
    """Input, raw argmax segmentations, combined map, and entropy heatmaps."""
    stem = f"sample_{idx:05d}"
    cv2.imwrite(
        os.path.join(out_dir, f"{stem}_input.png"),
        _tensor_to_bgr_uint8(image_chw),
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
    room_colors = _tab20_segmentation_colors(N_ROOM_CLASSES)
    _save_combined_segmentation_map_png(
        os.path.join(out_dir, f"{stem}_combined.png"),
        build_combined_map(rooms_seg, icons_seg),
        COMBINED_CLASSES,
        room_colors,
        combined_map_colors(room_colors),
    )
    room_ent = _entropy_from_probs(rooms_probs, N_ROOM_CLASSES).numpy()
    icon_ent = _entropy_from_probs(icons_probs, N_ICON_CLASSES).numpy()
    _save_entropy_heatmap_png(
        os.path.join(out_dir, f"{stem}_room_entropy.png"),
        room_ent,
        "Room entropy (normalized)",
    )
    _save_entropy_heatmap_png(
        os.path.join(out_dir, f"{stem}_icon_entropy.png"),
        icon_ent,
        "Icon entropy (normalized)",
    )


def save_sample_postproc_pngs(out_dir, idx, pol_rooms, pol_icons):
    """Post-processed segmentation PNGs for a visualized sample."""
    if pol_rooms is None and pol_icons is None:
        return
    stem = f"sample_{idx:05d}"
    tag = "_postproc"
    if pol_rooms is not None:
        _save_segmentation_map_png(
            os.path.join(out_dir, f"{stem}_room_segmentation{tag}.png"),
            pol_rooms,
            N_ROOM_CLASSES,
        )
    if pol_icons is not None:
        _save_segmentation_map_png(
            os.path.join(out_dir, f"{stem}_icon_segmentation{tag}.png"),
            pol_icons,
            N_ICON_CLASSES,
        )
    if pol_rooms is not None and pol_icons is not None:
        combined = build_combined_map(pol_rooms, pol_icons)
        room_colors = _tab20_segmentation_colors(N_ROOM_CLASSES)
        _save_combined_segmentation_map_png(
            os.path.join(out_dir, f"{stem}_combined{tag}.png"),
            combined,
            COMBINED_CLASSES,
            room_colors,
            combined_map_colors(room_colors),
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

        running_room_raw = runningScore(N_ROOM_CLASSES)
        running_icon_raw = runningScore(N_ICON_CLASSES)
        running_room_pp = {t: runningScore(N_ROOM_CLASSES) for t in EVAL_THRESHOLDS}
        running_icon_pp = {t: runningScore(N_ICON_CLASSES) for t in EVAL_THRESHOLDS}
        n_samples = len(testloader.dataset)
        rng = np.random.default_rng(42)
        k_vis = min(3, n_samples)
        vis_indices = set(rng.choice(n_samples, size=k_vis, replace=False).tolist())
        vis_dir = os.path.join(results_dir, "eval_samples_seed42")
        os.makedirs(vis_dir, exist_ok=True)

        entropy_room = entropy_icon = 0.0
        entropy_pixel_count = 0
        vis_cache = {}

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

                outputs = predict_at_resolution(self.model, images, full_res_shape)
                rooms_gt = labels[0, N_HEATMAPS].long().numpy()
                icons_gt = labels[0, N_HEATMAPS + 1].long().numpy()

                outputs_cpu = outputs.detach().cpu().float()
                heatmaps, rooms, icons = post_prosessing.split_prediction(
                    outputs_cpu, full_res_shape, self.input_slice
                )
                rooms_seg = np.argmax(rooms, axis=0)
                icons_seg = np.argmax(icons, axis=0)

                running_room_raw.update([rooms_gt], [rooms_seg])
                running_icon_raw.update([icons_gt], [icons_seg])
                room_ent = _entropy_from_probs(rooms, N_ROOM_CLASSES)
                icon_ent = _entropy_from_probs(icons, N_ICON_CLASSES)
                entropy_room += float(room_ent.sum().item())
                entropy_icon += float(icon_ent.sum().item())

                for thr in EVAL_THRESHOLDS:
                    try:
                        pol_rooms, pol_icons = run_postproc_mini(
                            heatmaps, rooms, icons, full_res_shape, thr
                        )
                        running_room_pp[thr].update([rooms_gt], [pol_rooms])
                        running_icon_pp[thr].update([icons_gt], [pol_icons])
                    except Exception as e:
                        self.logger.warning(
                            "Post-processing failed sample %d (thr=%.2f): %s",
                            global_idx,
                            thr,
                            e,
                        )

                if global_idx in vis_indices:
                    vis_cache[global_idx] = {
                        "image": images[0].detach().cpu(),
                        "heatmaps": heatmaps,
                        "rooms": rooms.copy(),
                        "icons": icons,
                        "rooms_seg": rooms_seg,
                        "icons_seg": icons_seg,
                        "full_res_shape": full_res_shape,
                    }

                entropy_pixel_count += room_ent.numel()
                global_idx += 1

        best_thr = select_best_postproc_threshold(
            running_room_pp, running_icon_pp, EVAL_THRESHOLDS
        )
        self.logger.info("Best post-processing threshold: %.2f", best_thr)

        for idx, cached in vis_cache.items():
            save_sample_raw_pngs(
                vis_dir,
                idx,
                cached["image"],
                cached["rooms_seg"],
                cached["icons_seg"],
                cached["rooms"],
                cached["icons"],
            )
            try:
                pol_rooms, pol_icons = run_postproc_mini(
                    cached["heatmaps"],
                    cached["rooms"],
                    cached["icons"],
                    cached["full_res_shape"],
                    best_thr,
                )
                save_sample_postproc_pngs(vis_dir, idx, pol_rooms, pol_icons)
            except Exception as e:
                self.logger.warning(
                    "Vis post-processing failed sample %d (thr=%.2f): %s",
                    idx,
                    best_thr,
                    e,
                )

        denom = max(1, entropy_pixel_count)
        eval_rows = build_eval_rows(
            best_thr,
            running_room_raw,
            running_icon_raw,
            running_room_pp,
            running_icon_pp,
            entropy_room / denom,
            entropy_icon / denom,
        )

        return {
            "eval_rows": eval_rows,
            "best_postproc_threshold": best_thr,
            "n_samples": n_samples,
            "vis_dir": vis_dir,
            "confusion_room_raw": running_room_raw.confusion_matrix.copy(),
            "confusion_icon_raw": running_icon_raw.confusion_matrix.copy(),
            "confusion_room_postproc": running_room_pp[
                best_thr
            ].confusion_matrix.copy(),
            "confusion_icon_postproc": running_icon_pp[
                best_thr
            ].confusion_matrix.copy(),
            "running_room_raw": running_room_raw,
            "running_icon_raw": running_icon_raw,
            "running_room_pp": running_room_pp,
            "running_icon_pp": running_icon_pp,
            "mean_room_entropy": entropy_room / denom,
            "mean_icon_entropy": entropy_icon / denom,
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
        ("confusion_room_postproc", results["confusion_room_postproc"]),
        ("confusion_icon_postproc", results["confusion_icon_postproc"]),
    ]:
        img_path, raw_path = save_confusion_matrix_artifacts(run_dir, cm, stem=stem)
        cm_paths[stem] = (img_path, raw_path)

    best_thr = results["best_postproc_threshold"]
    json_rows = []
    for mode_label, thr, use_pp in _eval_row_specs(best_thr):
        entry = {"mode": mode_label}
        if use_pp:
            entry["postproc_threshold"] = thr
            entry["room"] = _json_block(results["running_room_pp"][thr])
            entry["icon"] = _json_block(results["running_icon_pp"][thr])
        else:
            entry["room"] = _json_block(results["running_room_raw"])
            entry["icon"] = _json_block(results["running_icon_raw"])
            entry["mean_room_entropy_all_classes"] = results["mean_room_entropy"]
            entry["mean_icon_entropy_all_classes"] = results["mean_icon_entropy"]
        json_rows.append(entry)

    json_payload = {
        "thresholds_swept": EVAL_THRESHOLDS,
        "best_postproc_threshold": best_thr,
        "rows": json_rows,
        "n_samples": results["n_samples"],
    }

    output_path = os.path.join(run_dir, "eval.json")
    with open(output_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    print(f"Saved evaluation CSV to {csv_path} ({len(results['eval_rows'])} rows)")
    print(f"Saved evaluation results to {output_path}")
    for stem, (img_path, raw_path) in cm_paths.items():
        print(f"  {stem}: {img_path}, {raw_path}")
    print(f"Saved eval samples under {results['vis_dir']}")
    print(f"Best post-processing threshold: {best_thr:.2f}")
