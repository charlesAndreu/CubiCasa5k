"""
Regression + behavior tests for post_process_wall.py.

Run with: python test_post_process_wall.py [-v]

No pytest dependency (not installed in this env) -- plain unittest, runnable directly.
"""

import math
import unittest

import cv2
import numpy as np

from floortrans.loaders.svg_utils import get_gaussian2D
from post_process_wall import (
    build_wall_network,
    build_wall_network_evidence,
    generate_candidates,
    select_wall_edges_by_evidence,
    _dominant_angles,
    extract_wall_points,
)

KERNEL = get_gaussian2D(7)


def _heatmaps_from_points(arity_points, opening_points=(), size=220):
    """arity_points: list of (x, y, arity in 1..4). opening_points: list of (x, y)."""
    h = w = size
    heatmaps = np.zeros((5, h, w), dtype=np.float64)
    for x, y, arity in arity_points:
        heatmaps[arity - 1, int(round(y)), int(round(x))] = 1.0
    for x, y in opening_points:
        heatmaps[4, int(round(y)), int(round(x))] = 1.0
    for c in range(5):
        heatmaps[c] = cv2.filter2D(heatmaps[c], -1, KERNEL)
    return heatmaps


def _segment_angles_deg(wall_segments):
    angles = []
    for (x1, y1), (x2, y2) in wall_segments:
        angles.append(math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0)
    return angles


def _room_seg_from_walls(segments, size, wall_class_id=2, background_id=3, thickness=5):
    """Paint a synthetic room/wall segmentation: background everywhere except a
    `thickness`-px band along each of `segments` -- real pixel evidence for
    select_wall_edges_by_evidence to check candidates against, independent of
    whatever points/heatmaps a test also builds."""
    room_seg = np.full((size, size), background_id, dtype=np.int32)
    for (x1, y1), (x2, y2) in segments:
        cv2.line(
            room_seg, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
            wall_class_id, thickness,
        )
    return room_seg


def _tilted_chain(start, tilt_deg, n_segments, segment_len=40.0, jitter=0.3):
    """n_segments+1 points along a line at tilt_deg from horizontal, small alternating
    jitter so points aren't perfectly machine-precise collinear (more realistic)."""
    a = math.radians(tilt_deg)
    pts = []
    for k in range(n_segments + 1):
        d = segment_len * k
        x = start[0] + d * math.cos(a) + (jitter if k % 2 == 0 else -jitter)
        y = start[1] + d * math.sin(a) - (jitter if k % 2 == 0 else -jitter)
        pts.append((x, y))
    return pts


class TestBaselineRegression(unittest.TestCase):
    """Core scenarios that must never break, regardless of unrelated changes."""

    def test_rectangle_with_T_junction_and_opening(self):
        pts = [(20, 20, 2), (100, 20, 2), (100, 100, 2), (20, 100, 2), (60, 20, 3), (60, 70, 1)]
        heatmaps = _heatmaps_from_points(pts, opening_points=[(20, 40), (20, 55)], size=128)
        result = build_wall_network(heatmaps, point_threshold=0.3, opening_threshold=0.3, min_length_px=3)
        self.assertEqual(len(result["wall_segments"]), 6)
        self.assertEqual(len(result["openings"]), 1)

    def test_jittered_rectangle_selected_and_cleanly_snapped(self):
        pts = [(20, 20, 2), (100, 22, 2), (99, 100, 2), (21, 99, 2)]
        heatmaps = _heatmaps_from_points(pts, size=128)
        result = build_wall_network(heatmaps, point_threshold=0.3, min_length_px=3)
        self.assertEqual(len(result["wall_segments"]), 4)

        snapped = build_wall_network(heatmaps, point_threshold=0.3, min_length_px=3, snap_axis_tolerance_px=3)
        xs = sorted({round(p, 3) for s in snapped["wall_segments"] for p in (s[0][0], s[1][0])})
        ys = sorted({round(p, 3) for s in snapped["wall_segments"] for p in (s[0][1], s[1][1])})
        self.assertEqual(len(xs), 2)
        self.assertEqual(len(ys), 2)

    def test_phantom_diagonal_rejected_by_wall_evidence(self):
        pts = [(20, 20, 2), (100, 20, 3), (180, 20, 2), (130, 90, 2)]
        heatmaps = _heatmaps_from_points(pts, size=200)
        room_seg = np.full((200, 200), 3, dtype=np.int64)
        room_seg[18:23, :] = 2  # the only real wall band
        result = build_wall_network(
            heatmaps, point_threshold=0.3, min_length_px=3, room_seg=room_seg, wall_class_id=2
        )
        self.assertEqual(len(result["wall_segments"]), 2)


class TestTiltedEdgeKeptOrNot(unittest.TestCase):
    """The user-requested matrix: a near-rectangular shape where one edge is tilted
    5/10/15/30 degrees off cardinal, checked against varying evidence (how many
    points sit on that edge) and parameters (angle_tolerance_deg, snap tolerance).
    Documents the actual, nuanced behavior rather than a single blanket rule:

    - Selection (topology) always keeps the tilted edge, regardless of tilt or
      evidence -- it's never dropped from the graph just for being non-cardinal.
    - Without snapping, geometry is never altered -- the true tilt is always
      preserved as-detected (snap is opt-in).
    - With snapping on, the outcome depends on evidence:
      * A single tilted edge (no supporting chain) is NOT recognized as its own
        "dominant" direction, so it competes only against the 0/90 baseline. A
        small tilt (e.g. 5 deg) implies a small pixel offset even at moderate
        length, so a generous snap tolerance pulls it to cardinal -- correctly
        treating a small tilt as most likely prediction noise, not an intentional
        angle, when nothing else corroborates it.
      * A larger tilt (15/30 deg) with only one edge and no corroborating chain is
        simply left alone by snap once its implied pixel deviation exceeds the
        tolerance -- neither forced to cardinal nor artificially straightened.
      * The same tilt WITH a corroborating chain of points (>=3 supporting
        candidate pairs) is recognized as a well-evidenced direction and gets
        cleanly snapped to ITS OWN angle, not forced toward 0/90.
    """

    def test_selection_always_keeps_the_tilted_edge(self):
        for tilt in (5, 10, 15, 30):
            for n_segments in (1, 3):
                with self.subTest(tilt=tilt, n_segments=n_segments):
                    pts = _tilted_chain((30, 30), tilt, n_segments)
                    heatmaps = _heatmaps_from_points([(x, y, 2) for x, y in pts])
                    result = build_wall_network(heatmaps, point_threshold=0.3, min_length_px=3)
                    self.assertEqual(len(result["wall_segments"]), n_segments)

    def test_without_snap_geometry_is_never_altered(self):
        for tilt in (5, 10, 15, 30):
            with self.subTest(tilt=tilt):
                pts = _tilted_chain((30, 30), tilt, n_segments=1, jitter=0.0)
                heatmaps = _heatmaps_from_points([(x, y, 2) for x, y in pts])
                result = build_wall_network(heatmaps, point_threshold=0.3, min_length_px=3)
                angle = _segment_angles_deg(result["wall_segments"])[0]
                self.assertAlmostEqual(angle, tilt, delta=1.0)

    def test_single_edge_small_tilt_gets_pulled_to_cardinal_by_generous_snap(self):
        # 5 degrees over a 40px edge implies ~3.5px offset -- well within a 15px budget.
        pts = _tilted_chain((30, 30), tilt_deg=5, n_segments=1, jitter=0.0)
        heatmaps = _heatmaps_from_points([(x, y, 2) for x, y in pts])
        result = build_wall_network(heatmaps, point_threshold=0.3, min_length_px=3, snap_axis_tolerance_px=15)
        angle = _segment_angles_deg(result["wall_segments"])[0]
        self.assertAlmostEqual(angle, 0.0, delta=0.5)

    def test_single_edge_large_tilt_is_left_alone_not_forced(self):
        # 30 degrees over a 40px edge implies ~20px offset -- exceeds a 15px budget,
        # so it's correctly left un-snapped rather than distorted toward cardinal.
        pts = _tilted_chain((30, 30), tilt_deg=30, n_segments=1, jitter=0.0)
        heatmaps = _heatmaps_from_points([(x, y, 2) for x, y in pts])
        result = build_wall_network(heatmaps, point_threshold=0.3, min_length_px=3, snap_axis_tolerance_px=15)
        angle = _segment_angles_deg(result["wall_segments"])[0]
        self.assertAlmostEqual(angle, 30.0, delta=1.0)
        self.assertGreater(abs(angle - 0.0), 5.0)

    def test_well_evidenced_tilt_is_recognized_and_cleanly_snapped_to_its_own_angle(self):
        for tilt in (15, 30):
            with self.subTest(tilt=tilt):
                pts = _tilted_chain((30, 30), tilt, n_segments=3)  # 4 points, 3 segments, 6 candidate pairs
                heatmaps = _heatmaps_from_points([(x, y, 2) for x, y in pts])

                candidates = generate_candidates(
                    extract_wall_points(heatmaps, threshold=0.3, min_separation_px=15), min_length_px=3
                )
                dominant = _dominant_angles(candidates)
                self.assertTrue(
                    any(abs(((d - tilt + 90) % 180) - 90) < 3 for d in dominant),
                    f"tilt={tilt} should be detected as a dominant angle, got {dominant}",
                )

                result = build_wall_network(heatmaps, point_threshold=0.3, min_length_px=3, snap_axis_tolerance_px=3)
                angles = [round(a, 2) for a in _segment_angles_deg(result["wall_segments"])]
                self.assertEqual(len(set(angles)), 1, f"all segments should share one snapped angle, got {angles}")
                self.assertAlmostEqual(angles[0], tilt, delta=3.0)


class TestEvidenceBasedSelection(unittest.TestCase):
    """select_wall_edges_by_evidence / build_wall_network_evidence: the alternate
    principle where segmentation directly decides topology (accept a pair iff the
    line between them is wall-covered) instead of a confidence/length/angle scoring
    race for a shared arity budget. No arity, no angle bonus, no crossing check --
    only wall-pixel evidence and the same bypass rule as the scored path."""

    def test_requires_room_seg(self):
        pts = [(20, 20, 2), (100, 20, 2)]
        heatmaps = _heatmaps_from_points(pts, size=128)
        with self.assertRaises(ValueError):
            build_wall_network_evidence(heatmaps, room_seg=None, point_threshold=0.3, min_length_px=3)

    def test_rectangle_with_T_junction_matches_scored_baseline(self):
        pts = [(20, 20, 2), (100, 20, 2), (100, 100, 2), (20, 100, 2), (60, 20, 3), (60, 70, 1)]
        heatmaps = _heatmaps_from_points(pts, size=128)
        walls = [
            ((20, 20), (100, 20)), ((100, 20), (100, 100)),
            ((100, 100), (20, 100)), ((20, 100), (20, 20)),
            ((60, 20), (60, 70)),
        ]
        room_seg = _room_seg_from_walls(walls, size=128)
        result = build_wall_network_evidence(
            heatmaps, room_seg, point_threshold=0.3, min_length_px=3, min_wall_fraction=0.5,
        )
        # Same 6 as the scored baseline: 4 sides (top split in two by the T point) + 1 stub.
        self.assertEqual(len(result["wall_segments"]), 6)

    def test_phantom_diagonal_rejected_for_lack_of_evidence(self):
        pts = [(20, 20, 2), (100, 20, 3), (180, 20, 2), (130, 90, 2)]
        heatmaps = _heatmaps_from_points(pts, size=200)
        room_seg = np.full((200, 200), 3, dtype=np.int64)
        room_seg[18:23, :] = 2  # the only real wall band -- no evidence for the diagonal
        result = build_wall_network_evidence(
            heatmaps, room_seg, point_threshold=0.3, min_length_px=3, wall_class_id=2,
        )
        self.assertEqual(len(result["wall_segments"]), 2)

    def test_bypass_over_a_real_wall_still_rejected(self):
        # Three collinear points, ALL on real wall pixels (one continuous painted
        # line) -- evidence alone can't tell the long bypass from the two atomic
        # sub-segments, since the underlying pixels support both equally. The
        # _passes_through_other_point rule (kept from the scored path) is what
        # keeps this from producing a redundant long edge on top of the two short
        # ones connecting the same three points.
        points = [
            {"x": 20.0, "y": 50.0, "arity": 2, "conf": 0.9},
            {"x": 100.0, "y": 50.0, "arity": 3, "conf": 0.9},
            {"x": 180.0, "y": 50.0, "arity": 2, "conf": 0.9},
        ]
        room_seg = _room_seg_from_walls([((20, 50), (180, 50))], size=220)
        accepted, segs, _ = select_wall_edges_by_evidence(points, room_seg, min_length_px=3)
        self.assertEqual(len(segs), 2)
        pairs = {frozenset((c["i"], c["j"])) for c in accepted}
        self.assertEqual(pairs, {frozenset((0, 1)), frozenset((1, 2))})

    def test_arity_caps_a_point_with_too_many_evidence_valid_arms(self):
        # A '+' of 4 arms around a center point predicted arity=2: segmentation
        # alone can't tell a genuine 2-way point from a 4-way one (thick walls /
        # segmentation noise can put wall pixels in all 4 directions), so all 4
        # arms individually clear min_wall_fraction. Arity caps the center at 2
        # edges, and wall_fraction (the only signal this method has) breaks the
        # tie -- the two most solidly evidenced arms (west/east, fully painted)
        # must win over the two weaker ones (north/south, each with a gap near
        # the center that lowers their fraction, but not below the accept bar).
        center = {"x": 100.0, "y": 100.0, "arity": 2, "conf": 0.9}
        west = {"x": 20.0, "y": 100.0, "arity": 2, "conf": 0.9}
        east = {"x": 180.0, "y": 100.0, "arity": 2, "conf": 0.9}
        north = {"x": 100.0, "y": 20.0, "arity": 2, "conf": 0.9}
        south = {"x": 100.0, "y": 180.0, "arity": 2, "conf": 0.9}
        points = [center, west, east, north, south]

        room_seg = _room_seg_from_walls(
            [
                ((20, 100), (100, 100)),  # west arm: fully painted -> fraction ~1.0
                ((100, 100), (180, 100)),  # east arm: fully painted -> fraction ~1.0
                ((100, 20), (100, 90)),  # north arm: painted only 70/80 of the way -> ~0.875
                ((100, 130), (100, 180)),  # south arm: painted only 50/80 of the way -> ~0.625
            ],
            size=220,
        )
        accepted, segs, _ = select_wall_edges_by_evidence(points, room_seg, min_wall_fraction=0.5, min_length_px=3)
        self.assertEqual(len(segs), 2)
        pairs = {frozenset((c["i"], c["j"])) for c in accepted}
        self.assertEqual(pairs, {frozenset((0, 1)), frozenset((0, 2))})  # center-west, center-east only


if __name__ == "__main__":
    unittest.main(verbosity=2)
