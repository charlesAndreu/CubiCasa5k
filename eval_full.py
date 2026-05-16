"""
Test-set evaluation for the full Cubicasa5k model
(21 heatmap channels + 3 room logits + 4 icon logits = 28 output channels).

Mini room layout (3 classes): 0 = outside, 1 = walls, 2 = inside
Mini icon layout (4 classes): 0 = empty,   1 = window, 2 = door, 3 = others

Outputs per run (saved under <run_dir>/):
  * eval.json — raw and post-processed metrics for both heads + mean entropies
  * confusion_{room|icon}_{raw|postproc}.png / _raw.json
  * eval_samples_seed42/sample_XXXXX_*.png — 3 random test samples:
      input, room/icon argmax, wall mask, room/icon entropy heatmaps,
      post-processed room/icon segmentation, 5-class combined map
      (outside / walls / inside / windows / doors).

Test images are processed at native (LMDB) resolution. The fully-convolutional
Furukawa model accepts arbitrary sizes; predictions are bilinearly resampled
back to the ground-truth resolution via post_prosessing.split_prediction
before metrics and post-processing run.
"""

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
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataloader import build_cubicasa5k_full_eval_dataloaders_native_res
from eval_simple import (
    _save_entropy_heatmap_png,
    _save_segmentation_map_png,
    _tensor_to_bgr_uint8,
    _to_jsonable,
    save_confusion_matrix_artifacts,
)
from floortrans import post_prosessing
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

POSTPROC_THRESHOLD = 0.4


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
        _save_segmentation_map_png(
            os.path.join(out_dir, f"{stem}_combined_postproc.png"),
            combined,
            COMBINED_CLASSES,
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
        running_room_pp = runningScore(N_ROOM_CLASSES)
        running_icon_pp = runningScore(N_ICON_CLASSES)

        n_samples = len(testloader.dataset)
        rng = np.random.default_rng(42)
        k_vis = min(3, n_samples)
        vis_indices = set(rng.choice(n_samples, size=k_vis, replace=False).tolist())
        vis_dir = os.path.join(results_dir, "eval_samples_seed42")
        os.makedirs(vis_dir, exist_ok=True)

        entropy_room_sum = 0.0
        entropy_icon_sum = 0.0
        entropy_pixel_count = 0
        post_failed = 0

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

                outputs = self.model(images)  # (1, 28, h_out, w_out)

                # CubiCasa5KFurukawa (full) is now built with n_heatmap_channels=21,
                # so the forward pass already applies sigmoid to the first 21
                # channels. Pass outputs straight through to split_prediction.
                #
                # WARNING: older checkpoints trained when n_heatmap_channels was 0
                # were optimized as raw MSE without sigmoid; running them through
                # this model class will sigmoid-squash already-small raw outputs
                # (baseline raw 0 → 0.5 ≥ POSTPROC_THRESHOLD), swamping
                # post-processing. For those checkpoints, retrain (preferred) or
                # temporarily revert n_heatmap_channels to 0 in model.py.
                outputs_cpu = outputs.detach().cpu().float()

                # bilinear-upsample to full resolution + softmax over rooms & icons
                heatmaps, rooms, icons = post_prosessing.split_prediction(
                    outputs_cpu, full_res_shape, self.input_slice
                )

                rooms_seg = np.argmax(rooms, axis=0)
                icons_seg = np.argmax(icons, axis=0)

                rooms_gt = labels[0, N_HEATMAPS].long().numpy()
                icons_gt = labels[0, N_HEATMAPS + 1].long().numpy()

                running_room_raw.update([rooms_gt], [rooms_seg])
                running_icon_raw.update([icons_gt], [icons_seg])

                room_entropy = _entropy_from_probs(rooms, N_ROOM_CLASSES)
                icon_entropy = _entropy_from_probs(icons, N_ICON_CLASSES)
                entropy_room_sum += float(room_entropy.sum().item())
                entropy_icon_sum += float(icon_entropy.sum().item())
                entropy_pixel_count += room_entropy.numel()

                pol_rooms = pol_icons = combined = None
                try:
                    polygons, types, room_polygons, room_types = get_polygons_mini(
                        (heatmaps, rooms.copy(), icons),
                        threshold=POSTPROC_THRESHOLD,
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
                    pol_rooms = np.argmax(
                        predicted_classes[:N_ROOM_CLASSES], axis=0
                    ).astype(np.int64)
                    pol_icons = np.argmax(
                        predicted_classes[N_ROOM_CLASSES:], axis=0
                    ).astype(np.int64)
                    running_room_pp.update([rooms_gt], [pol_rooms])
                    running_icon_pp.update([icons_gt], [pol_icons])
                    combined = build_combined_map(pol_rooms, pol_icons)
                except Exception as e:
                    post_failed += 1
                    self.logger.warning(
                        "Post-processing failed for sample %d: %s", global_idx, e
                    )

                if global_idx in vis_indices:
                    save_sample_pngs(
                        vis_dir,
                        global_idx,
                        images[0],
                        rooms_seg,
                        icons_seg,
                        room_entropy.cpu().numpy(),
                        icon_entropy.cpu().numpy(),
                        pol_rooms,
                        pol_icons,
                        combined,
                    )

                global_idx += 1

        room_raw_score, room_raw_iou = running_room_raw.get_scores()
        icon_raw_score, icon_raw_iou = running_icon_raw.get_scores()
        room_pp_score, room_pp_iou = running_room_pp.get_scores()
        icon_pp_score, icon_pp_iou = running_icon_pp.get_scores()

        denom = max(1, entropy_pixel_count)
        return {
            "room_raw_score": room_raw_score,
            "room_raw_class_iou": room_raw_iou,
            "icon_raw_score": icon_raw_score,
            "icon_raw_class_iou": icon_raw_iou,
            "room_postproc_score": room_pp_score,
            "room_postproc_class_iou": room_pp_iou,
            "icon_postproc_score": icon_pp_score,
            "icon_postproc_class_iou": icon_pp_iou,
            "mean_room_entropy_all_classes": entropy_room_sum / denom,
            "mean_icon_entropy_all_classes": entropy_icon_sum / denom,
            "confusion_room_raw": running_room_raw.confusion_matrix.copy(),
            "confusion_icon_raw": running_icon_raw.confusion_matrix.copy(),
            "confusion_room_postproc": running_room_pp.confusion_matrix.copy(),
            "confusion_icon_postproc": running_icon_pp.confusion_matrix.copy(),
            "n_samples": n_samples,
            "post_failed": post_failed,
            "vis_dir": vis_dir,
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python eval_full.py <run_dir>")
    run_dir = sys.argv[1]
    args = load_eval_args(run_dir)

    evaluator = FullSegEvaluator(args)
    results = evaluator.evaluate(results_dir=run_dir)

    cm_paths = {}
    for stem, cm in [
        ("confusion_room_raw", results["confusion_room_raw"]),
        ("confusion_icon_raw", results["confusion_icon_raw"]),
        ("confusion_room_postproc", results["confusion_room_postproc"]),
        ("confusion_icon_postproc", results["confusion_icon_postproc"]),
    ]:
        img_path, raw_path = save_confusion_matrix_artifacts(run_dir, cm, stem=stem)
        cm_paths[stem] = (img_path, raw_path)

    json_payload = {
        "room_raw": {
            "score": _to_jsonable(results["room_raw_score"]),
            "class_iou": _to_jsonable(results["room_raw_class_iou"]),
            "mean_entropy_all_classes": results["mean_room_entropy_all_classes"],
        },
        "icon_raw": {
            "score": _to_jsonable(results["icon_raw_score"]),
            "class_iou": _to_jsonable(results["icon_raw_class_iou"]),
            "mean_entropy_all_classes": results["mean_icon_entropy_all_classes"],
        },
        "room_postproc": {
            "score": _to_jsonable(results["room_postproc_score"]),
            "class_iou": _to_jsonable(results["room_postproc_class_iou"]),
        },
        "icon_postproc": {
            "score": _to_jsonable(results["icon_postproc_score"]),
            "class_iou": _to_jsonable(results["icon_postproc_class_iou"]),
        },
        "n_samples": results["n_samples"],
        "post_failed": results["post_failed"],
    }

    output_path = os.path.join(run_dir, "eval.json")
    with open(output_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    print(f"Saved evaluation results to {output_path}")
    for stem, (img_path, raw_path) in cm_paths.items():
        print(f"  {stem}: {img_path}, {raw_path}")
    print(
        f"Mean room entropy (3 classes): "
        f"{results['mean_room_entropy_all_classes']:.6f}"
    )
    print(
        f"Mean icon entropy (4 classes): "
        f"{results['mean_icon_entropy_all_classes']:.6f}"
    )
    print(
        f"Post-processing failures: "
        f"{results['post_failed']}/{results['n_samples']}"
    )
    print(f"Saved eval samples under {results['vis_dir']}")
