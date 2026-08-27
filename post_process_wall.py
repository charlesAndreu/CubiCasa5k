"""
Build a realistic (non-Manhattan-forced) wall graph from predicted wall-point heatmaps.

Input: a (5, H, W) heatmap array (channels 0-3 = junction arity 1/2/3/4, channel 4 =
opening endpoint — see floortrans/loaders/wall_loader.py and plan_wall_training.md).
Output: wall centerline segments + opening gaps, built with graph/geometric rules --
plus, optionally, real pixel evidence from a room/wall segmentation when one is
available (see generate_candidates' room_seg param) -- unlike floortrans/post_prosessing.py,
which only connects points whose encoded orientation codes are exact 0/90/180/270 opposites
(calc_point_info) and then force-snaps everything onto a Manhattan grid (points_to_manhantan).
"""

import argparse
import logging
import math

import numpy as np
import torch
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from floortrans.post_prosessing import extract_local_max, bresenham_line
from floortrans.loaders.wall_loader import (
    ARITY_CHANNEL_BY_JUNCTION_CHANNEL,
    OPENING_JUNCTION_CHANNELS,
    N_WALL_CHANNELS,
)

N_ARITY_CHANNELS = 4  # channels 0-3: arity 1,2,3,4
OPENING_CHANNEL = 4


# ------------------------------------------------------------------
# Adapting a train_full-style 21-channel prediction (no train_wall.py
# checkpoint needed -- reuses the *same* junction-type semantics, just
# grouped by arity instead of kept as separate orientation channels)
# ------------------------------------------------------------------

def remap_prediction_to_wall_heatmaps(heatmaps_21chw):
    """21-channel train_full heatmap prediction (already sigmoid-activated by the
    Furukawa model, see model.py) -> 5-channel wall heatmaps (arity + opening), by
    taking the max across each junction-type/orientation group instead of picking one
    specific orientation channel -- this is the inference-time analogue of
    wall_loader.remap_heatmap_dict_to_wall_channels, which does the same grouping on
    ground-truth point *lists* at training time; here we're grouping continuous
    activations, so max (not sum) keeps the result naturally in the same [0,1] range."""
    h, w = heatmaps_21chw.shape[1:]
    wall_heatmaps = np.zeros((N_WALL_CHANNELS, h, w), dtype=heatmaps_21chw.dtype)
    for channel, arity_channel in ARITY_CHANNEL_BY_JUNCTION_CHANNEL.items():
        wall_heatmaps[arity_channel] = np.maximum(
            wall_heatmaps[arity_channel], heatmaps_21chw[channel]
        )
    for channel in OPENING_JUNCTION_CHANNELS:
        wall_heatmaps[OPENING_CHANNEL] = np.maximum(
            wall_heatmaps[OPENING_CHANNEL], heatmaps_21chw[channel]
        )
    return wall_heatmaps


# ------------------------------------------------------------------
# Point extraction
# ------------------------------------------------------------------

def extract_wall_points(heatmaps, threshold=0.5, max_points=300, min_separation_px=15):
    """Sum the 4 arity channels for peak location (avoids the same physical corner
    firing separate near-duplicate peaks on adjacent arity channels), then read the
    per-channel value at each peak to assign arity = argmax across the 4 channels.

    close_point_suppression=True additionally zeroes a min_separation_px box around
    each accepted peak, on top of extract_local_max's own flood-fill NMS -- the
    flood-fill alone only kills a *contiguous, strictly-decreasing* blob, so two close
    but separately-shaped bumps for the same physical corner (very possible when
    combining 4 independently-predicted channels, or on a heatmap trained with a wide
    kernel) survive as distinct points without this."""
    arity_maps = heatmaps[:N_ARITY_CHANNELS]
    corner_sum = arity_maps.sum(axis=0)
    raw_points = extract_local_max(
        corner_sum, max_points, info=[], heatmap_value_threshold=threshold,
        close_point_suppression=True, gap=int(min_separation_px),
    )

    points = []
    for x, y, conf in raw_points:
        arity = int(np.argmax(arity_maps[:, y, x])) + 1  # 1..4
        points.append({"x": float(x), "y": float(y), "conf": float(conf), "arity": arity})
    return points


def extract_opening_points(heatmaps, threshold=0.5, max_points=300, min_separation_px=15):
    raw_points = extract_local_max(
        heatmaps[OPENING_CHANNEL], max_points, info=[], heatmap_value_threshold=threshold,
        close_point_suppression=True, gap=int(min_separation_px),
    )
    return [{"x": float(x), "y": float(y), "conf": float(conf)} for x, y, conf in raw_points]


# ------------------------------------------------------------------
# Candidate generation + geometric scoring
# ------------------------------------------------------------------

def _project_onto_segment(p, seg_start, seg_end):
    """Return (t, perp_dist): t is the [0,1] param along the segment, perp_dist the
    perpendicular distance from p to the infinite line through the segment."""
    sx, sy = seg_start
    ex, ey = seg_end
    dx, dy = ex - sx, ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return 0.0, math.hypot(p[0] - sx, p[1] - sy)
    t = ((p[0] - sx) * dx + (p[1] - sy) * dy) / length_sq
    proj_x, proj_y = sx + t * dx, sy + t * dy
    perp_dist = math.hypot(p[0] - proj_x, p[1] - proj_y)
    return t, perp_dist


def _dominant_angles(
    candidates, n_bins=180, min_separation_deg=15.0, top_k_extra=1,
    cardinal_angles=(0.0, 90.0), extra_min_weight_fraction=0.5, extra_min_count=4,
):
    """Horizontal/vertical always get the angle bonus baseline -- real floorplans are
    overwhelmingly axis-aligned, so cardinal directions shouldn't have to win a
    popularity contest against whatever this sample's candidates happen to contain.
    On top of that fixed baseline, still look for up to top_k_extra genuinely
    well-supported *other* directions (a real angled wall, e.g. a bay window) -- this
    is what keeps the design non-Manhattan-forced rather than reverting to the old
    post-process's hardcoded 0/90/180/270-only behavior: a non-cardinal angle can
    still earn the bonus, it just has to actually earn it.

    extra_min_count matters more than it looks: with a sparse candidate set, a single
    coincidental off-axis candidate can trivially tie extra_min_weight_fraction's
    *relative* bar against another single coincidental candidate (nothing else is
    competing, so whatever is largest passes "at least half of the max" by
    definition) and get promoted -- sometimes even outscoring a real, slightly
    jittered cardinal edge, stealing its point's arity budget and dropping a real
    wall entirely. Requiring an *absolute* minimum number of distinct supporting
    candidates (not just a fraction of an otherwise-small max) is what actually
    filters that out; a genuine repeated architectural angle has several edges
    reinforcing it, a coincidence usually has one or two."""
    dominant = list(cardinal_angles)
    if not candidates:
        return dominant

    angles = np.array([c["angle_deg"] % 180.0 for c in candidates])
    weights = np.array([c["base_score"] for c in candidates])
    hist, edges = np.histogram(angles, bins=n_bins, range=(0.0, 180.0), weights=weights)
    counts, _ = np.histogram(angles, bins=n_bins, range=(0.0, 180.0))
    max_weight = float(hist.max()) if hist.size else 0.0

    def support_count(angle):
        total = 0
        for bi in range(n_bins):
            bin_angle = (edges[bi] + edges[bi + 1]) / 2.0
            d = min(abs(bin_angle - angle), 180.0 - abs(bin_angle - angle))
            if d <= min_separation_deg:
                total += int(counts[bi])
        return total

    hist = hist.copy()
    for cardinal in cardinal_angles:
        for bi in range(n_bins):
            bin_angle = (edges[bi] + edges[bi + 1]) / 2.0
            d = min(abs(bin_angle - cardinal), 180.0 - abs(bin_angle - cardinal))
            if d <= min_separation_deg:
                hist[bi] = 0

    for _ in range(top_k_extra):
        b = int(np.argmax(hist))
        if hist[b] <= 0 or hist[b] < extra_min_weight_fraction * max_weight:
            break
        angle = (edges[b] + edges[b + 1]) / 2.0
        if support_count(angle) < extra_min_count:
            hist[b] = 0
            continue
        dominant.append(angle)
        for bi in range(n_bins):
            bin_angle = (edges[bi] + edges[bi + 1]) / 2.0
            d = min(abs(bin_angle - angle), 180.0 - abs(bin_angle - angle))
            if d <= min_separation_deg:
                hist[bi] = 0
    return dominant


def _angle_bonus(angle_deg, dominant_angles, tolerance_deg=6.0, bonus=0.35):
    if not dominant_angles:
        return 0.0
    a = angle_deg % 180.0
    best = min(min(abs(a - d), 180.0 - abs(a - d)) for d in dominant_angles)
    if best >= tolerance_deg:
        return 0.0
    return bonus * (1.0 - best / tolerance_deg)


def _passes_through_other_point(points, i, j, exclude, tolerance_px):
    """True if some point other than i/j sits almost exactly on segment (i,j) --
    a real wall shouldn't "jump over" an intermediate junction that itself has
    unused arity budget; the two sub-segments should be used instead."""
    seg_start = (points[i]["x"], points[i]["y"])
    seg_end = (points[j]["x"], points[j]["y"])
    for m, p in enumerate(points):
        if m in exclude:
            continue
        t, perp = _project_onto_segment((p["x"], p["y"]), seg_start, seg_end)
        if 0.02 < t < 0.98 and perp <= tolerance_px:
            return True
    return False


def _wall_pixel_fraction(room_seg, wall_class_id, x1, y1, x2, y2):
    """Fraction of the candidate line's rasterized pixels that the model's own room
    segmentation actually calls wall -- real pixel evidence, not geometry. A genuine
    wall's line should be wall-covered nearly end to end; a spurious diagonal cutting
    across a room's interior mostly crosses non-wall pixels instead. Reuses the same
    bresenham_line the legacy post_prosessing.get_wall_lines uses for this exact
    purpose (post_prosessing.py:243-246), which returns (row, col) tuples."""
    line_pixels = bresenham_line(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
    if not line_pixels:
        return 1.0
    h, w = room_seg.shape
    on_wall = 0
    total = 0
    for row, col in line_pixels:
        if 0 <= row < h and 0 <= col < w:
            total += 1
            if room_seg[row, col] == wall_class_id:
                on_wall += 1
    return on_wall / total if total else 1.0


def generate_candidates(
    points, k_neighbors=12, min_length_px=6.0, max_length_px=None, pass_through_tolerance_px=4.0,
    room_seg=None, wall_class_id=2, wall_evidence_weight=0.9, min_wall_fraction=0.5,
):
    """Bounded candidate set: each point only pairs with its k nearest neighbors
    (plus symmetric pairs), not the full N^2 pairing."""
    n = len(points)
    if n < 2:
        return []
    coords = np.array([[p["x"], p["y"]] for p in points])
    tree = cKDTree(coords)
    k = min(k_neighbors + 1, n)  # +1 because a point is its own nearest neighbor
    _, neighbor_idx = tree.query(coords, k=k)

    seen = set()
    candidates = []
    for i in range(n):
        for j in neighbor_idx[i]:
            j = int(j)
            if j == i:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)

            dx = points[j]["x"] - points[i]["x"]
            dy = points[j]["y"] - points[i]["y"]
            length = math.hypot(dx, dy)
            if length < min_length_px:
                continue
            if max_length_px is not None and length > max_length_px:
                continue
            if _passes_through_other_point(points, i, j, {i, j}, pass_through_tolerance_px):
                continue

            angle_deg = math.degrees(math.atan2(dy, dx))
            conf_score = points[i]["conf"] * points[j]["conf"]
            # length plausibility: only a soft rolloff for very short (noisy) spans.
            # No penalty at all beyond that -- a genuinely long wall in its own
            # direction is not "less plausible" than a short one just for being long;
            # the actual bypass-ambiguity case (a short edge vs. a longer collinear
            # superset of it) is handled structurally by _passes_through_other_point
            # above, not by a blanket length preference that would otherwise also
            # penalize long walls with no such ambiguity at all.
            noise_band = 3 * min_length_px
            length_score = length / noise_band if length < noise_band else 1.0

            # Real pixel evidence, when available: heavily favor a candidate whose
            # line actually sits on wall-classified pixels over one that doesn't,
            # instead of relying purely on point-graph geometry to guess that. Below
            # min_wall_fraction this is a hard reject, not just a discount -- a pure
            # scaling factor can't actually veto a candidate, since a heavily
            # discounted score can still "win" a slot by default when nothing else
            # is available to fill it; less than half a line's own length being
            # wall-classified is strong enough evidence that no real wall is there to
            # justify never even offering it as a candidate.
            wall_evidence_score = 1.0
            if room_seg is not None:
                fraction = _wall_pixel_fraction(
                    room_seg, wall_class_id,
                    points[i]["x"], points[i]["y"], points[j]["x"], points[j]["y"],
                )
                if fraction < min_wall_fraction:
                    continue
                wall_evidence_score = (1.0 - wall_evidence_weight) + wall_evidence_weight * fraction

            candidates.append({
                "i": i, "j": j,
                "length": length,
                "angle_deg": angle_deg,
                "base_score": conf_score * length_score * wall_evidence_score,
            })
    return candidates


def select_wall_edges(
    points, candidates, angle_bonus_weight=0.35, angle_tolerance_deg=6.0,
    arity_grace=0, max_hard_degree=4, grace_relative_threshold=0.85,
):
    """Greedy degree-constrained planar-graph construction: accept candidates in
    descending score order, subject to (a) each point's predicted arity budget and
    (b) no edge crossing an already-accepted edge except at a shared endpoint.

    Two segments sharing only an endpoint (the common case: two walls meeting at a
    corner) intersect on their boundary, not their interior, so shapely's `.crosses()`
    correctly returns False for them and True only for a genuine interior crossing --
    no separate shared-endpoint special-casing needed.

    arity_grace: OFF (0) by default -- testing showed this is a genuine trade-off, not
    a clean win. The predicted arity can be off by one (a true 3-way T-junction
    predicted as arity-2), and capping hard at the raw prediction then permanently
    drops that real wall. But scoring a "should this extra edge be rescued" decision
    by relative strength alone is fragile in both directions: loose enough to rescue
    a genuine miss, and it also lets a coincidentally similar-scoring spurious diagonal
    through elsewhere; strict enough to reject that noise, and it can also reject the
    genuine miss it was meant to rescue (its natural length/angle are often weaker than
    the point's other edges simply because of geometry, not because it's wrong). Set
    arity_grace=1 deliberately if you want to trade a few possible extra spurious edges
    for a chance at recovering under-predicted junctions -- it is not a strict
    improvement, so verify the result rather than assuming it will only ever help. Up
    to arity_grace extra edges (bounded by the physically-sane max_hard_degree=4) are
    allowed past a point's normal budget -- but ONLY if the candidate's score is at
    least grace_relative_threshold times the weakest edge *already accepted* at that
    point. A genuine missed wall is normally just as strong as its siblings at the
    same junction (similar confidence/length/angle); a spurious diagonal chord that
    merely happens to have a leftover slot to fall into is not, and this gate is what
    tells them apart -- an unconditional extra slot would accept both.
    """
    dominant_angles = _dominant_angles(candidates)
    for c in candidates:
        c["score"] = c["base_score"] * (1.0 + _angle_bonus(
            c["angle_deg"], dominant_angles, angle_tolerance_deg, angle_bonus_weight
        ))
    # Length only breaks a near-tie (e.g. two candidates at the same angle, one a
    # short real edge and one a coincidentally-parallel longer one) -- it must NOT
    # shift the ranking of two candidates that already differ meaningfully on
    # confidence/angle, or a genuinely long wall in its own direction gets penalized
    # for being long with nothing to actually disambiguate against. length_tiebreak_eps
    # is deliberately tiny relative to a real angle-bonus/confidence gap.
    length_tiebreak_eps = 0.001
    for c in candidates:
        c["score"] -= length_tiebreak_eps * c["length"]

    candidates_sorted = sorted(candidates, key=lambda c: c["score"], reverse=True)
    remaining_arity = [p["arity"] for p in points]
    grace_remaining = [max(0, min(arity_grace, max_hard_degree - p["arity"])) for p in points]
    accepted_scores_at = [[] for _ in points]
    accepted = []
    accepted_segments = []

    for c in candidates_sorted:
        i, j = c["i"], c["j"]
        over_i = remaining_arity[i] <= 0
        over_j = remaining_arity[j] <= 0
        if over_i and (grace_remaining[i] <= 0 or (
            accepted_scores_at[i] and c["score"] < grace_relative_threshold * min(accepted_scores_at[i])
        )):
            continue
        if over_j and (grace_remaining[j] <= 0 or (
            accepted_scores_at[j] and c["score"] < grace_relative_threshold * min(accepted_scores_at[j])
        )):
            continue

        seg = ((points[i]["x"], points[i]["y"]), (points[j]["x"], points[j]["y"]))
        if any(LineString(seg).crosses(LineString(other)) for other in accepted_segments):
            continue

        if over_i:
            grace_remaining[i] -= 1
        else:
            remaining_arity[i] -= 1
        if over_j:
            grace_remaining[j] -= 1
        else:
            remaining_arity[j] -= 1
        accepted_scores_at[i].append(c["score"])
        accepted_scores_at[j].append(c["score"])
        accepted.append(c)
        accepted_segments.append(seg)

    return accepted, accepted_segments


# ------------------------------------------------------------------
# Openings: split wall edges at door/window gaps
# ------------------------------------------------------------------

def snap_axis_aligned_points(points, accepted_edges, snap_tolerance_px):
    """Force-align points connected by an already-accepted near-horizontal (resp.
    near-vertical) edge onto a shared exact y (resp. x) -- their average -- turning
    "nearly straight" into "exactly straight". Opt-in (snap_tolerance_px=0/falsy
    disables it): this deliberately only touches edges *already* close to cardinal,
    so it cleans up noisy point predictions into crisp walls without ever forcing a
    genuinely angled wall onto the grid -- same clustering idea as the legacy
    points_to_manhantan (post_prosessing.py), but opt-in rather than always-on, and
    only ever applied to edges the graph itself already decided were axis-aligned.

    The gate is an absolute *pixel* deviation from perfectly cardinal
    (length * sin(angle-from-cardinal)), not a fixed angle: the same few degrees of
    deviation implies a tiny offset on a short wall but a large one on a long wall,
    so a fixed-angle gate would either be too eager to snap a long, genuinely tilted
    wall, or too strict to clean up an obviously-jittery short one. A fixed pixel
    budget scales correctly with segment length instead.

    Returns a new list of point dicts; does not mutate the input."""
    if not snap_tolerance_px:
        return points

    n = len(points)
    parent_h = list(range(n))
    parent_v = list(range(n))

    def find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(parent, a, b):
        ra, rb = find(parent, a), find(parent, b)
        if ra != rb:
            parent[ra] = rb

    for c in accepted_edges:
        angle = c["angle_deg"] % 180.0
        dist_to_0 = min(angle, 180.0 - angle)
        dist_to_90 = abs(angle - 90.0)
        if dist_to_0 <= dist_to_90:
            if c["length"] * math.sin(math.radians(dist_to_0)) <= snap_tolerance_px:
                union(parent_h, c["i"], c["j"])
        else:
            if c["length"] * math.sin(math.radians(dist_to_90)) <= snap_tolerance_px:
                union(parent_v, c["i"], c["j"])

    snapped = [dict(p) for p in points]

    def snap_clusters(parent, coord_key):
        groups = {}
        for idx in range(n):
            groups.setdefault(find(parent, idx), []).append(idx)
        for members in groups.values():
            if len(members) < 2:
                continue
            avg = sum(points[m][coord_key] for m in members) / len(members)
            for m in members:
                snapped[m][coord_key] = avg

    snap_clusters(parent_h, "y")  # same-horizontal-cluster points share one y
    snap_clusters(parent_v, "x")  # same-vertical-cluster points share one x
    return snapped


def attach_openings(wall_segments, opening_points, perp_tolerance_px=12.0):
    """For each accepted wall edge, find opening endpoints that lie close to it, pair
    them two at a time along the edge (by projection order), and record the gap span.
    Returns a list of {"edge_index": i, "gap": (p_start, p_end)}."""
    openings = []
    for edge_idx, (seg_start, seg_end) in enumerate(wall_segments):
        matches = []
        for op in opening_points:
            p = (op["x"], op["y"])
            t, perp = _project_onto_segment(p, seg_start, seg_end)
            if -0.05 <= t <= 1.05 and perp <= perp_tolerance_px:
                matches.append((t, p))
        matches.sort(key=lambda tp: tp[0])
        for k in range(0, len(matches) - 1, 2):
            openings.append({
                "edge_index": edge_idx,
                "gap": (matches[k][1], matches[k + 1][1]),
            })
    return openings


# ------------------------------------------------------------------
# End-to-end pipeline
# ------------------------------------------------------------------

def build_wall_network(
    heatmaps,
    point_threshold=0.5,
    opening_threshold=0.5,
    k_neighbors=8,
    min_length_px=6.0,
    max_length_px=None,
    perp_tolerance_px=12.0,
    min_separation_px=15,
    arity_grace=0,
    angle_bonus_weight=0.35,
    snap_axis_tolerance_px=0,
    room_seg=None,
    wall_class_id=2,
    wall_evidence_weight=0.9,
    min_wall_fraction=0.5,
):
    """heatmaps: (5, H, W) numpy array of sigmoid-activated model predictions.
    angle_bonus_weight: strength of the horizontal/vertical preference (0 = no axis
    preference at all, higher = walls need stronger non-axis evidence to compete with
    an axis-aligned alternative for the same point's arity budget).
    snap_axis_tolerance_px: 0 disables (default). Otherwise, force-align points
    connected by an accepted edge whose implied cardinal deviation is within this many
    pixels onto a shared y/x -- see snap_axis_aligned_points.
    room_seg: optional (H, W) room/wall segmentation array (e.g. from the same
    train_full model's own room head -- viz_web already has this for free). When
    given, candidates are scored by real pixel evidence (what fraction of the
    candidate line actually falls on wall_class_id) instead of purely geometric
    heuristics -- see generate_candidates. None (default) skips this entirely, which
    is required for a points-only train_wall.py model that has no room segmentation
    at all."""
    points = extract_wall_points(heatmaps, threshold=point_threshold, min_separation_px=min_separation_px)
    opening_points = extract_opening_points(heatmaps, threshold=opening_threshold, min_separation_px=min_separation_px)

    candidates = generate_candidates(
        points, k_neighbors=k_neighbors, min_length_px=min_length_px, max_length_px=max_length_px,
        room_seg=room_seg, wall_class_id=wall_class_id, wall_evidence_weight=wall_evidence_weight,
        min_wall_fraction=min_wall_fraction,
    )
    accepted_edges, wall_segments = select_wall_edges(
        points, candidates, arity_grace=arity_grace, angle_bonus_weight=angle_bonus_weight,
    )

    if snap_axis_tolerance_px:
        points = snap_axis_aligned_points(points, accepted_edges, snap_axis_tolerance_px)
        wall_segments = [
            ((points[c["i"]]["x"], points[c["i"]]["y"]), (points[c["j"]]["x"], points[c["j"]]["y"]))
            for c in accepted_edges
        ]

    openings = attach_openings(wall_segments, opening_points, perp_tolerance_px=perp_tolerance_px)

    return {
        "points": points,
        "opening_points": opening_points,
        "wall_segments": wall_segments,
        "openings": openings,
    }


def render_wall_network_bgr(base_bgr, result):
    """Draw points + wall segments + opening gaps directly onto a BGR image with cv2
    (no matplotlib) -- for callers like viz_web that render a PNG per HTTP request and
    want the same lightweight drawing style already used for its other artifacts."""
    import cv2

    img = base_bgr.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    for (x1, y1), (x2, y2) in result["wall_segments"]:
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.line(img, p1, p2, (255, 255, 255), 4, cv2.LINE_AA)  # white halo for contrast
        cv2.line(img, p1, p2, (255, 0, 255), 2, cv2.LINE_AA)  # BGR magenta -- flashy, unlikely to clash with plan ink

    # Openings sit *in* a wall, they don't cut it -- wall_segments is never split (see
    # attach_openings), and the drawing must not visually contradict that either: no
    # thick overlay painted along the wall itself. Mark each opening endpoint with a
    # short tick crossing the wall instead, so the magenta line stays visibly
    # continuous underneath.
    tick_half_len = 8
    for op in result["openings"]:
        (sx1, sy1), (sx2, sy2) = result["wall_segments"][op["edge_index"]]
        wdx, wdy = sx2 - sx1, sy2 - sy1
        wlen = math.hypot(wdx, wdy) or 1.0
        perp_x, perp_y = -wdy / wlen, wdx / wlen
        for gx, gy in op["gap"]:
            p1 = (int(round(gx - perp_x * tick_half_len)), int(round(gy - perp_y * tick_half_len)))
            p2 = (int(round(gx + perp_x * tick_half_len)), int(round(gy + perp_y * tick_half_len)))
            cv2.line(img, p1, p2, (0, 140, 255), 2, cv2.LINE_AA)  # BGR orange

    for p in result["points"]:
        center = (int(round(p["x"])), int(round(p["y"])))
        cv2.circle(img, center, 4, (255, 130, 0), -1, cv2.LINE_AA)  # BGR blue
        cv2.putText(
            img, str(p["arity"]), (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 130, 0), 1, cv2.LINE_AA,
        )
    for op in result["opening_points"]:
        center = (int(round(op["x"])), int(round(op["y"])))
        cv2.circle(img, center, 3, (0, 140, 255), -1, cv2.LINE_AA)

    return img


def visualize(result, height, width, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")

    for seg in result["wall_segments"]:
        (x1, y1), (x2, y2) = seg
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2)

    for op in result["openings"]:
        (x1, y1), (x2, y2) = op["gap"]
        ax.plot([x1, x2], [y1, y2], color="white", linewidth=4, zorder=3)
        ax.plot([x1, x2], [y1, y2], color="tab:orange", linewidth=2, linestyle="--", zorder=4)

    for p in result["points"]:
        ax.scatter([p["x"]], [p["y"]], c="tab:blue", s=20, zorder=5)
        ax.annotate(str(p["arity"]), (p["x"], p["y"]), fontsize=7, color="tab:blue")

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# CLI: run a trained train_wall.py checkpoint on one LMDB sample
# ------------------------------------------------------------------

def _load_wall_model(checkpoint_path, model_args, device, logger):
    from types import SimpleNamespace
    from model import cubi_casa5k_wall_model

    args = SimpleNamespace(resume_from=None, furukawa_weights=None, **model_args)
    model = cubi_casa5k_wall_model(args, logger)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model


def main():
    import lmdb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="train_wall.py .pkl checkpoint")
    parser.add_argument("--data_path", default="data/cubicasa5k/")
    parser.add_argument("--folder", required=True, help="LMDB key from val.txt, e.g. 'high_quality_architectural/2'")
    parser.add_argument("--model", default="segformer")
    parser.add_argument("--segformer_model_name", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--n_wall_channels", type=int, default=5)
    parser.add_argument("--point_threshold", type=float, default=0.5)
    parser.add_argument("--opening_threshold", type=float, default=0.5)
    parser.add_argument("--k_neighbors", type=int, default=8)
    parser.add_argument("--out", default="wall_network.png")
    args = parser.parse_args()

    from floortrans.loaders.wall_loader import build_wall_val_augmentations, WallLoader

    logger = logging.getLogger("post_process_wall")
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_wall_model(
        args.checkpoint,
        dict(model=args.model, segformer_model_name=args.segformer_model_name, n_wall_channels=args.n_wall_channels),
        device,
        logger,
    )

    lmdb_env = lmdb.open(args.data_path.rstrip("/") + "/cubi_lmdb", readonly=True, lock=False)
    val_args = argparse.Namespace(image_size=256, kernel_px=7)
    loader = WallLoader(args.data_path, "val.txt", lmdb_env, augmentations=build_wall_val_augmentations(val_args))
    idx = list(loader.folders).index(args.folder)
    sample = loader[idx]

    with torch.no_grad():
        image = sample["image"].unsqueeze(0).to(device)
        outputs = model(image)
        heatmaps = torch.sigmoid(outputs)[0].cpu().numpy()

    result = build_wall_network(
        heatmaps,
        point_threshold=args.point_threshold,
        opening_threshold=args.opening_threshold,
        k_neighbors=args.k_neighbors,
    )
    logger.info(
        "%d points, %d wall segments, %d openings",
        len(result["points"]), len(result["wall_segments"]), len(result["openings"]),
    )
    visualize(result, heatmaps.shape[1], heatmaps.shape[2], args.out)
    logger.info("Saved visualization to %s", args.out)


if __name__ == "__main__":
    main()
