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
    candidates, n_bins=180, min_separation_deg=15.0, top_k_extra=2,
    cardinal_angles=(0.0, 90.0), extra_min_weight_fraction=0.5, extra_min_count=3,
):
    """Horizontal/vertical always get the angle bonus baseline -- real floorplans are
    overwhelmingly axis-aligned, so cardinal directions shouldn't have to win a
    popularity contest against whatever this sample's candidates happen to contain.
    On top of that fixed baseline, still look for up to top_k_extra genuinely
    well-supported *other* directions (a real angled wall with several points on it,
    e.g. a 15deg wall, or a whole non-cardinal room -- top_k_extra=2 so a diagonal
    wall AND its perpendicular partner can both be found independently if both are
    well-evidenced, without hardcoding that they must come as a pair) -- this is
    what keeps the design non-Manhattan-forced rather than reverting to the old
    post-process's hardcoded 0/90/180/270-only behavior: a non-cardinal angle can
    still earn the bonus, it just has to actually earn it. The returned list is also
    reused directly by snap_axis_aligned_points, so a well-evidenced non-cardinal
    wall gets cleanly straightened too, not just preferred during selection.

    extra_min_count matters more than it looks: with a sparse candidate set, a single
    coincidental off-axis candidate can trivially tie extra_min_weight_fraction's
    *relative* bar against another single coincidental candidate (nothing else is
    competing, so whatever is largest passes "at least half of the max" by
    definition) and get promoted -- sometimes even outscoring a real, slightly
    jittered cardinal edge, stealing its point's arity budget and dropping a real
    wall entirely. Requiring an *absolute* minimum number of distinct supporting
    candidates (not just a fraction of an otherwise-small max) is what actually
    filters that out; a genuine repeated architectural angle has several edges
    reinforcing it (3, all-pairs among just 3 points on the same line), a
    coincidence usually has one or two."""
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
    segmentation calls wall -- real pixel evidence, used both as a soft scoring
    signal (generate_candidates' wall_evidence_weight) and as a hard accept/reject
    threshold (min_wall_fraction, both generate_candidates and
    select_wall_edges_by_evidence). Reuses the same bresenham_line the legacy
    post_prosessing.get_wall_lines uses for this exact purpose
    (post_prosessing.py:243-246), (row, col) tuples."""
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
    points, min_length_px=6.0, max_length_px=None, pass_through_tolerance_px=4.0,
    room_seg=None, wall_class_id=2, wall_evidence_weight=0.9, min_wall_fraction=0.5,
):
    """Every point pair is considered -- at real floorplan point counts (tens,
    rarely more than ~100) this is trivial (O(n^2)), and a k-nearest-neighbor
    prefilter was actively wrong, not just a performance shortcut: it can
    permanently exclude a real wall just for being geometrically "far" (e.g.
    spanning a wide room), whenever enough *other*, unrelated points happen to be
    closer in raw distance -- no matter how large k is set, there's always a room
    shape where the correct partner isn't in the top-k. The real filtering happens
    later and doesn't need a distance prefilter to work: _passes_through_other_point
    rejects bypasses, wall-pixel evidence rejects unsupported lines, and arity
    budget + no-crossing during selection reject the rest."""
    n = len(points)
    if n < 2:
        return []

    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
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

            # Real pixel evidence, when available: reject a candidate whose line
            # mostly doesn't sit on wall-classified pixels -- a soft scaling factor
            # alone can't act as a real veto (a heavily discounted score can still
            # "win" a slot by default when nothing better is available to fill it),
            # and a spurious diagonal cutting across open room space should never
            # win regardless of what else is or isn't competing for that slot.
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
    Returns (accepted, accepted_segments, dominant_angles) -- dominant_angles is the
    same list _angle_bonus scored against (0, 90, plus any well-evidenced extra
    angle from _dominant_angles), returned so snap_axis_aligned_points can clean up
    against exactly the same reference set scoring used, rather than a separate,
    narrower hardcoded one.

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
    remaining_arity = [p["arity"] for p in points]
    grace_remaining = [max(0, min(arity_grace, max_hard_degree - p["arity"])) for p in points]
    accepted_scores_at = [[] for _ in points]
    accepted = []
    accepted_segments = []

    def try_accept(c, score):
        i, j = c["i"], c["j"]
        over_i = remaining_arity[i] <= 0
        over_j = remaining_arity[j] <= 0
        if over_i and (grace_remaining[i] <= 0 or (
            accepted_scores_at[i] and score < grace_relative_threshold * min(accepted_scores_at[i])
        )):
            return False
        if over_j and (grace_remaining[j] <= 0 or (
            accepted_scores_at[j] and score < grace_relative_threshold * min(accepted_scores_at[j])
        )):
            return False

        seg = ((points[i]["x"], points[i]["y"]), (points[j]["x"], points[j]["y"]))
        if any(LineString(seg).crosses(LineString(other)) for other in accepted_segments):
            return False

        if over_i:
            grace_remaining[i] -= 1
        else:
            remaining_arity[i] -= 1
        if over_j:
            grace_remaining[j] -= 1
        else:
            remaining_arity[j] -= 1
        accepted_scores_at[i].append(score)
        accepted_scores_at[j].append(score)
        accepted.append(c)
        accepted_segments.append(seg)
        return True

    dominant_angles = _dominant_angles(candidates)
    for c in candidates:
        c["score"] = c["base_score"] * (1.0 + _angle_bonus(
            c["angle_deg"], dominant_angles, angle_tolerance_deg, angle_bonus_weight
        ))

    # Break exact score ties by preferring the shorter candidate -- but only as a
    # tie-break, never folded into `score` itself: at real floorplan scale (walls
    # spanning hundreds of px) an *additive* per-pixel length penalty stops being a
    # negligible epsilon and starts materially outscoring genuine long walls, which
    # directly contradicts generate_candidates' own length_score design (long spans
    # aren't penalized for being long). Sorting on a (score, length) tuple keeps the
    # tie-break meaningless whenever scores actually differ, at any scale.
    candidates_sorted = sorted(candidates, key=lambda c: (-c["score"], c["length"]))
    for c in candidates_sorted:
        try_accept(c, c["score"])

    return accepted, accepted_segments, dominant_angles


# ------------------------------------------------------------------
# Alternate selection principle: segmentation decides topology directly,
# instead of being one signal in a confidence/length/angle scoring race.
# ------------------------------------------------------------------

def select_wall_edges_by_evidence(
    points, room_seg, wall_class_id=2, min_wall_fraction=0.5,
    min_length_px=6.0, max_length_px=None, pass_through_tolerance_px=4.0,
):
    """Alternate to generate_candidates + select_wall_edges: an edge is a *candidate*
    whenever the model's own segmentation directly supports it (wall-pixel fraction
    along the line >= min_wall_fraction) -- no point-confidence product, no
    length/angle scoring. Segmentation is treated as ground truth for whether a line
    could be a wall at all, not a secondary veto layered onto a scoring race.

    Each point's predicted arity is still enforced as a hard cap on its degree,
    though: segmentation alone can't tell a genuine junction from a point that
    merely sits near wall pixels in several directions (thick walls, a nearby
    perpendicular corridor, segmentation noise), so an uncapped point can end up
    with more accepted edges than it physically has arms for. When a point has more
    evidence-valid candidates than its arity allows, only the top `arity` by
    wall_fraction are kept -- still an evidence-only tie-break (the strongest
    pixel support wins), not a reintroduction of the confidence/length/angle race
    select_wall_edges uses.

    This is still a real difference from that scoring race, though: the race can
    drop a genuine connection because an unrelated, higher-scoring pair *elsewhere
    in the graph* claimed a shared point's budget first (e.g. a long, low-confidence
    corridor edge losing to a short high-confidence one at the same point). Here,
    only candidates *at the same point* compete for that point's budget, and only
    on wall_fraction -- there's no global confidence/length/angle race to lose to.

    Still rejects a candidate that bypasses an intermediate point sitting on it
    (_passes_through_other_point) -- that's basic geometry (the "real" wall is the
    two sub-segments, not a redundant superset), not part of the scoring machinery
    this replaces; without it, a straight run of points would each also connect to
    every other point further down the same line, since the wall pixels support
    those bypasses too.

    No crossing check: that's the remaining tradeoff versus the scoring approach --
    this can still accept a pair of edges that cross in their interior, if both
    independently have wall evidence and arity room. That's part of the actual
    comparison this alternate is for.

    Requires room_seg: there's no other signal here to accept an edge on."""
    if room_seg is None:
        raise ValueError(
            "select_wall_edges_by_evidence requires room_seg -- "
            "evidence-first selection has no other signal to accept an edge on"
        )

    n = len(points)
    valid = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[j]["x"] - points[i]["x"]
            dy = points[j]["y"] - points[i]["y"]
            length = math.hypot(dx, dy)
            if length < min_length_px:
                continue
            if max_length_px is not None and length > max_length_px:
                continue
            if _passes_through_other_point(points, i, j, {i, j}, pass_through_tolerance_px):
                continue
            fraction = _wall_pixel_fraction(
                room_seg, wall_class_id, points[i]["x"], points[i]["y"], points[j]["x"], points[j]["y"],
            )
            if fraction < min_wall_fraction:
                continue
            angle_deg = math.degrees(math.atan2(dy, dx))
            valid.append({
                "i": i, "j": j, "length": length, "angle_deg": angle_deg,
                "wall_fraction": fraction,
                "base_score": fraction,  # _dominant_angles' vote weight -- no other score exists here
            })

    remaining_arity = [p["arity"] for p in points]
    accepted = []
    accepted_segments = []
    for c in sorted(valid, key=lambda c: c["wall_fraction"], reverse=True):
        i, j = c["i"], c["j"]
        if remaining_arity[i] <= 0 or remaining_arity[j] <= 0:
            continue
        remaining_arity[i] -= 1
        remaining_arity[j] -= 1
        accepted.append(c)
        accepted_segments.append(((points[i]["x"], points[i]["y"]), (points[j]["x"], points[j]["y"])))

    dominant_angles = _dominant_angles(accepted)
    return accepted, accepted_segments, dominant_angles


# ------------------------------------------------------------------
# Openings: split wall edges at door/window gaps
# ------------------------------------------------------------------

def snap_axis_aligned_points(points, accepted_edges, snap_tolerance_px, dominant_angles=(0.0, 90.0)):
    """Force-align points connected by an already-accepted edge close to one of
    dominant_angles onto a shared coordinate perpendicular to that direction --
    turning "nearly straight" into "exactly straight". Opt-in (snap_tolerance_px=0/
    falsy disables it): this deliberately only touches edges *already* close to one
    of these directions, so it cleans up noisy point predictions without ever
    forcing a genuinely angled wall onto the grid -- same clustering idea as the
    legacy points_to_manhantan (post_prosessing.py), but opt-in rather than
    always-on, and only ever applied to edges the graph itself already decided fit.

    dominant_angles: defaults to plain (0, 90) for standalone use, but is meant to
    be the *same* list select_wall_edges/_dominant_angles produced for scoring --
    when a non-cardinal direction is well-evidenced enough to earn the selection
    bonus, it should also get snapped cleanly, not just tolerated as noisy. Each
    listed angle defines its own family (points connected by an edge near that
    angle share the coordinate perpendicular to it, via a rotated-frame
    projection); 0 and 90 as explicit separate entries is what reduces this
    exactly to the plain horizontal/vertical formula for those two.

    The gate is an absolute *pixel* deviation from the reference direction
    (length * sin(angle-from-reference)), not a fixed angle: the same few degrees of
    deviation implies a tiny offset on a short wall but a large one on a long wall,
    so a fixed-angle gate would either be too eager to snap a long, genuinely tilted
    wall, or too strict to clean up an obviously-jittery short one. A fixed pixel
    budget scales correctly with segment length instead.

    Returns a new list of point dicts; does not mutate the input."""
    if not snap_tolerance_px:
        return points

    n = len(points)
    refs = [a % 180.0 for a in dominant_angles]
    parents = {ref: list(range(n)) for ref in refs}

    def find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(parent, a, b):
        ra, rb = find(parent, a), find(parent, b)
        if ra != rb:
            parent[ra] = rb

    def closest_ref(angle):
        angle = angle % 180.0
        best_ref, best_dist = refs[0], 180.0
        for ref in refs:
            d = min(abs(angle - ref), 180.0 - abs(angle - ref))
            if d < best_dist:
                best_dist, best_ref = d, ref
        return best_ref, best_dist

    for c in accepted_edges:
        ref, dist = closest_ref(c["angle_deg"])
        if c["length"] * math.sin(math.radians(dist)) <= snap_tolerance_px:
            union(parents[ref], c["i"], c["j"])

    snapped = [dict(p) for p in points]

    # Reads/writes `snapped` (not the original `points`): two of these families can
    # share a coordinate axis (e.g. 0 and 90 are both expressed in plain x/y), so if
    # this read from pristine `points` each time, whichever family ran second would
    # silently overwrite the first family's fix to that shared axis. Reading the
    # already-partially-snapped state instead makes all passes accumulate correctly:
    # each only ever moves points along ITS OWN perpendicular direction, leaving
    # whatever an earlier pass already set along that same axis alone.
    for ref in refs:
        parent = parents[ref]
        groups = {}
        for idx in range(n):
            groups.setdefault(find(parent, idx), []).append(idx)
        phi = math.radians(ref)
        u = (math.cos(phi), math.sin(phi))  # parallel unit vector
        v = (-math.sin(phi), math.cos(phi))  # perpendicular unit vector
        for members in groups.values():
            if len(members) < 2:
                continue
            perp_avg = sum(snapped[m]["x"] * v[0] + snapped[m]["y"] * v[1] for m in members) / len(members)
            for m in members:
                par = snapped[m]["x"] * u[0] + snapped[m]["y"] * u[1]
                snapped[m]["x"] = par * u[0] + perp_avg * v[0]
                snapped[m]["y"] = par * u[1] + perp_avg * v[1]

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
    angle_bonus_weight: strength of the alignment preference during edge SELECTION
    (0 = no preference at all, higher = a non-aligned wall needs stronger other
    evidence to compete with an aligned alternative for the same point's arity
    budget) -- against the image's global 0/90 axes plus any well-evidenced
    non-cardinal direction found per-plan (see _dominant_angles).
    snap_axis_tolerance_px: 0 disables (default). Otherwise, force-align points
    connected by an accepted edge whose implied deviation from the *same* detected
    angle set used for selection is within this many pixels onto a shared
    coordinate -- see snap_axis_aligned_points. Distinct from angle_bonus_weight:
    that affects which edges get selected (topology), this only cleans up the
    coordinates of edges already selected (geometry) -- e.g. you can want strong
    selection bias with no coordinate snapping, or vice versa.
    room_seg: optional (H, W) room/wall segmentation array (e.g. from the same
    train_full model's own room head -- viz_web already has this for free). When
    given, a candidate with less than min_wall_fraction of its line on wall_class_id
    pixels is hard-rejected outright (a spurious diagonal across open room space
    should never win purely for lack of a better competitor); wall_evidence_weight
    controls a softer preference for more-covered candidates on top of that -- see
    generate_candidates. None (default) skips this entirely, which is required for a
    points-only train_wall.py model that has no room segmentation at all."""
    points = extract_wall_points(heatmaps, threshold=point_threshold, min_separation_px=min_separation_px)
    opening_points = extract_opening_points(heatmaps, threshold=opening_threshold, min_separation_px=min_separation_px)

    candidates = generate_candidates(
        points, min_length_px=min_length_px, max_length_px=max_length_px,
        room_seg=room_seg, wall_class_id=wall_class_id, wall_evidence_weight=wall_evidence_weight,
        min_wall_fraction=min_wall_fraction,
    )
    accepted_edges, wall_segments, dominant_angles = select_wall_edges(
        points, candidates, arity_grace=arity_grace, angle_bonus_weight=angle_bonus_weight,
    )

    if snap_axis_tolerance_px:
        points = snap_axis_aligned_points(points, accepted_edges, snap_axis_tolerance_px, dominant_angles=dominant_angles)
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


def build_wall_network_evidence(
    heatmaps,
    room_seg,
    point_threshold=0.5,
    opening_threshold=0.5,
    min_length_px=6.0,
    max_length_px=None,
    perp_tolerance_px=12.0,
    min_separation_px=15,
    pass_through_tolerance_px=4.0,
    snap_axis_tolerance_px=0,
    wall_class_id=2,
    min_wall_fraction=0.5,
):
    """Same stages, input, and output shape as build_wall_network (point extraction,
    edge selection, optional axis-snap cleanup, opening attachment), but topology
    comes from select_wall_edges_by_evidence instead of generate_candidates +
    select_wall_edges -- see that function's docstring for the principle. Each
    point's predicted arity is still a hard cap there, same as build_wall_network,
    just with no angle_bonus_weight and no arity_grace: ties among a point's
    evidence-valid candidates break on wall_fraction alone, and there's no soft
    over-budget rescue. room_seg is required (unlike build_wall_network, where it's
    optional extra evidence on top of scoring)."""
    points = extract_wall_points(heatmaps, threshold=point_threshold, min_separation_px=min_separation_px)
    opening_points = extract_opening_points(heatmaps, threshold=opening_threshold, min_separation_px=min_separation_px)

    accepted_edges, wall_segments, dominant_angles = select_wall_edges_by_evidence(
        points, room_seg, wall_class_id=wall_class_id, min_wall_fraction=min_wall_fraction,
        min_length_px=min_length_px, max_length_px=max_length_px,
        pass_through_tolerance_px=pass_through_tolerance_px,
    )

    if snap_axis_tolerance_px:
        points = snap_axis_aligned_points(points, accepted_edges, snap_axis_tolerance_px, dominant_angles=dominant_angles)
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
    )
    logger.info(
        "%d points, %d wall segments, %d openings",
        len(result["points"]), len(result["wall_segments"]), len(result["openings"]),
    )
    visualize(result, heatmaps.shape[1], heatmaps.shape[2], args.out)
    logger.info("Saved visualization to %s", args.out)


if __name__ == "__main__":
    main()
