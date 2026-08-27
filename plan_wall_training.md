# Plan: `train_wall.py` + `post_process_wall.py` — points-only wall training & non-Manhattan post-processing

## Context

The repo already trains a 21-channel junction heatmap (`House.get_heatmap_dict`, `floortrans/loaders/house.py:635-699`) as part of the full multi-task model (`train_full.py`), then reconstructs wall polygons with `floortrans/post_prosessing.py`. That post-processor only ever connects two points if their encoded "orientation" codes are exact opposite cardinal directions (`calc_point_info`, `post_prosessing.py:1100-1202`), and it forcibly snaps connected groups onto shared x/y values (`points_to_manhantan`, `post_prosessing.py:624-635`). That's the root cause of every wall coming out at 0/90/180/270°.

Goal: a much narrower pipeline whose only job is to detect wall-junction points very precisely and connect them into a realistic (not forced-rectilinear) wall network, including door/window gaps. Two new scripts: `train_wall.py` (training) and `post_process_wall.py` (graph reconstruction from predicted heatmaps).

**Decisions already made:**
- Keep point "arity" (how many wall segments meet at a point: 1/2/3/4) as separate heatmap channels, instead of the current 13 direction-quantized junction-type channels. This gives post-processing a strong, direction-free structural prior.
- Post-processing scores candidate lines using pure graph/geometric rules (confidence, arity match, length, angle consistency, no crossing) — no image-gradient sampling, no auxiliary trained mask.
- Also include door/window opening points, so wall segments are correctly broken at openings (added mid-discussion).

## Key findings that shape the design

1. **The "13px" gaussian isn't the live one.** `get_gaussian2D(13)` (`house.py:761`, `augmentations.py:223`) is dead code — never called by the actual training path. The live kernel is `get_gaussian2D(int(30 * scale))` in `DictToTensor.cubi` (`floortrans/loaders/augmentations.py:202`), where `scale` is a per-sample constant fixed at LMDB-build time (`coef_width` in `svg_loader.py:96-98`) and is **never adjusted** when a sample is later cropped/resized to `image_size` for training (`augmentations.py:500,750` just pass `scale` through unchanged). So today's effective kernel footprint in final training pixels is inconsistent from sample to sample and not simply "13px". `train_wall.py` will sidestep this: kernel size becomes a plain config value in final (post-resize) pixels, applied the same way for every sample.

2. **Arity is already implicit in the existing LMDB — no data rebuild needed for wall points.** `House.get_number` (`house.py:1207-1208`) encodes `channel = (g-1)*4 + t`, where `g` (1-4) is exactly the junction arity (I=1, L=2, T=3, X=4) and `t` (1-4) is the now-unwanted rotation/orientation code. So channels 0-11 map to arity `g = channel//4 + 1`, and channel 12 is arity 4. **We can collapse the existing 13 channels into 4 arity channels by remapping, with no LMDB regeneration.** Same story for openings: channels 13-16 (`left/right/up/down` opening corners, `house.py:649-673`) can be merged into one generic "opening endpoint" channel — also no rebuild needed. Door vs. window type is *not* separable from the current pickled dict (both share the same `opening_corners` buckets, `house.py:404,453-454,469-470,496-497,512-513`) — separating them later is possible but requires touching `House` and rebuilding the LMDB; out of scope for v1 (noted below).

3. **Known limitation carried over from the dataset, not introduced by us:** `House.lines_to_points` explicitly skips any wall pair that isn't axis-aligned (`house.py:786-789`, `get_lineDim` returns -1 for diagonal → `continue`). This means truly diagonal-wall corners are **absent from ground truth today**, in the existing LMDB. Our new arity channels inherit this gap (garbage in, garbage out) since we're reusing the existing pickled points rather than rebuilding from raw wall segments (`self.new_walls`/`representation["walls"]`, which *are* angle-agnostic). Practical effect: v1 will still be excellent at ordinary corners with much sharper/precise localization, and post-processing will *allow* non-cardinal connections (no forced Manhattan-snap) whenever prediction noise or an actually-angled pair of points supports it — but the model won't proactively learn to detect genuinely diagonal corners it was never shown. If that turns out to matter in practice, a phase-2 fix is: extend `House` with a new angle-agnostic junction extractor built from `self.new_walls` (clustering endpoints via `calc_distance`, `svg_utils.py:265`) and regenerate the LMDB. Flagging this now; not building it in this pass.

4. `extract_local_max` / `maximum_suppression` (`post_prosessing.py:1058-1097`) are generic peak-picking/NMS with zero coupling to the 13-channel scheme — fully reusable for the new heatmaps as-is.

5. No existing segment-crossing test exists anywhere in the repo (only unbounded line-line intersection and bbox-overlap helpers) — `post_process_wall.py` needs a small new bounded segment-intersection check (`shapely.geometry.LineString.crosses`, already a repo dependency via `post_prosessing.py:9-10`).

## Data / target design (channels)

5 output channels, each a Gaussian heatmap (sharp, config-controlled kernel, default ~7px at training resolution — configurable):
- **0-3: wall-point arity** — arity 1 (dead-end), 2 (corner), 3 (T-junction), 4 (X-junction), remapped from existing channels 0-12 as described above.
- **4: opening endpoint** — merged from existing channels 13-16 (door+window corners together, type-agnostic for v1).

Icon-corner channels (17-20 in the old scheme) are dropped entirely — out of scope ("just walls + openings").

## `train_wall.py`

**New lightweight loader** (new module, e.g. `floortrans/loaders/wall_loader.py`, modeled directly on `FullLoader`, `floortrans/loaders/room_icon_loaders.py:224-282`):
- Unpickle LMDB sample, drop room/icon label entirely, remap the 21-channel point dict → 5-channel dict as above, **before** running geometric augmentations.
- Reuse unchanged: `RandomCropToSizeTorch(data_format="dict")`, `ResizePaddedTorch(data_format="dict")` (`augmentations.py`) — both are channel-count-agnostic for the point-dict representation.
- `RandomRotations.cubi`'s channel-remap table (`hmapp_convert_map`, `augmentations.py:59-81`) encoded 90°-rotation semantics for direction-coded channels. Arity is rotation-invariant, so for the new 5-channel scheme this table collapses to the identity (channel 4 "opening endpoint" is also direction-agnostic) — write a small `hmapp_convert_map_wall` (identity map) rather than reusing the old table.
- `DictToTensor`: reuse `.cubi` structure but parameterize `n_channels=5` and a fixed `kernel_px` from config instead of the current `21` / `int(30*scale)` (`augmentations.py:192,202`).

**Model** (`model.py`): add a `cubi_casa5k_wall_model(args, logger)` analogous to `cubi_casa5k_full_model`/`cubi_casa5k_simple_model` (`model.py:152-163`):
- Furukawa: reuse the exact `n_heatmap_channels=0` pattern from the `train_simple` branch (`model.py:47-55`) so the model returns **raw logits** uniformly, rather than Furukawa's special internal-sigmoid-on-first-K-channels behavior (`hg_furukawa_original.py:213-216`) — avoids a dual-convention headache across architectures.
- U-Net / SegFormer: same channel-count parameterization already in place (`classes=`/`num_labels=`), just pass 5 instead of `n_segmentation_classes(...)`.
- Trainer applies `torch.sigmoid` uniformly after `model(images)`, consistently for all 3 architectures.

**Loss**: plain per-channel MSE between `sigmoid(logits)` and the target heatmap (resized to output resolution the same way `train_simple.py:62-72` does for its segmentation target). Also expose a `focal-heatmap` option (CenterNet/CornerNet-style penalty-reduced pixelwise focal loss) as an alternate `criterion` config value — this directly helps with a *thinner* Gaussian, since plain MSE on a much smaller positive region gets a weak/imbalanced gradient signal, and this is the standard fix in the keypoint-heatmap literature. Both are small (~20-40 line) additions modeled on `criterion.py`'s existing `homosced_heatmap_mse_loss` (`criterion.py:244-267`) pattern; no need for the full uncertainty-weighting machinery.

**Trainer skeleton**: copy `Cubicasa5kFullTrainer`'s structure (`train_full.py:83-382`) minus everything room/icon-specific (drop `runningScore`, the seg-argmax val block, `_seg_argmax_at_label_size`). Add one meaningful new **validation metric**: per-epoch, run `extract_local_max` (`post_prosessing.py:1058-1079`, confirmed reusable as-is) on predicted heatmaps and compute point-level precision/recall/F1 against ground-truth points at a small pixel-distance tolerance — a far more meaningful signal than raw MSE for this task.

**TensorBoard**: new `WallTrainingTensorBoard(SimpleTrainingTensorBoard)` reusing the already-generic `_figure_heatmap_sum`/`_pred_heatmap_sum_display` helpers (`training_tensorboard.py:19-33`), logging the 5 channels + the new point-F1 metric.

**Config**: new `train_wall_config.yaml`, same conventions as `train_simple_config.yaml`/`train_full_config.yaml` (`data_path`, `n_epoch`, `batch_size`, `image_size`, `l_rate`, `optimizer`, `patience`, `log_path`, `model`, `segformer_model_name`, `num_workers`, `prefetch_factor`, `debug`, `plot_samples`), plus new keys: `kernel_px` (gaussian size, default small e.g. 7), `criterion` (`mse` default / `focal-heatmap`).

## `post_process_wall.py`

Input: the 5-channel sigmoid heatmap prediction (+ image size). Output: a wall centerline graph (segments) with opening gaps marked — **not** full wall-thickness polygons (that's an orthogonal problem; noted as an easy phase-2 add once centerlines are validated, generalizing `extract_wall_polygon`'s perpendicular-width-walk, `post_prosessing.py:861-984`, from axis-locked to arbitrary angle).

1. **Point extraction**: sum channels 0-3 into one "corner confidence" map, run `extract_local_max` once on the sum for location, then read each channel's raw value at that (x,y) and take the arity = argmax across the 4 channels there. (This avoids the same physical corner producing 2-4 near-duplicate detections from separate per-channel peak-picking.) Separately run `extract_local_max` on channel 4 for opening endpoints.

2. **Candidate generation ("not so much")**: for each wall point, only consider its K nearest neighbor points (K≈6-8, plus anything within a generous max-wall-length radius) as candidate partners — bounded to O(N·K), not full O(N²).

3. **Scoring + selection — simple rules, applied as one greedy pass over candidates sorted by score (highest first):**
   - Score = point-confidence product × length-plausibility (reject below a min-wall-length; soft-penalize very long unsupported spans) × a small bonus for angles close to the floorplan's own *observed* dominant angles (found by a quick histogram/mode over all high-score candidate angles for *this* sample — not a hardcoded global 0/90/180/270 — so genuinely angled walls aren't penalized just for not being that sample's dominant angle).
   - Accept a candidate edge only if: (a) neither endpoint has already used up its predicted arity budget, and (b) it doesn't cross any already-accepted edge except at a shared endpoint (new small `shapely.geometry.LineString.crosses` check, `post_prosessing.py` has no such utility today).
   - This is a standard greedy degree-constrained planar-graph construction — simple, deterministic, and directly matches "try candidates, keep only the most relevant with simple rules."

4. **Openings**: for each opening-endpoint point, find its nearest accepted wall edge (perpendicular distance below a small tolerance), pair up opening endpoints two-at-a-time along that edge (by projection order), and split the wall edge at the gap.

5. **Output**: a plain list of wall segments (endpoint pairs) + a list of opening gaps (edge id + gap span), plus a debug visualization reusing `floortrans/plotting.py`'s polygon/segment drawing utilities (generalized off `polygons_to_image`, `plotting.py:654-672`, which is already angle-agnostic).

## Explicitly out of scope for this pass (call out, don't build)

- Rebuilding the LMDB to fix the diagonal-wall ground-truth gap (finding #3 above).
- Distinguishing door vs. window at the point level (would need `House`/LMDB changes).
- Full wall-thickness polygon extraction at arbitrary angles (centerlines only for v1).

## Verification

- Sanity-check the new loader on a handful of LMDB samples: plot the 5 remapped channels next to the original 21-channel ones to confirm arity/opening remapping is correct before any training run.
- Train `train_wall.py` for a short smoke run (few epochs, small `n_epoch`) on the existing `data/cubicasa5k/cubi_lmdb`, confirm loss decreases and TensorBoard point-F1 metric moves.
- Run `post_process_wall.py` on a val-set prediction, visually compare against the ground-truth SVG wall layout, and against the old `floortrans/post_prosessing.py` output on the same sample to confirm angled/T/X cases are now handled sensibly and openings produce gaps.
