from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.movie_time_ranges import merge_overlapping_movie_ranges


class MovieTimeRangeMergeTests(unittest.TestCase):
    def test_merges_same_shot_overlapping_ranges(self) -> None:
        merged = merge_overlapping_movie_ranges(
            [
                {"start": 500.5, "end": 504.5, "movie_shot_ids": ["movie_shot_000113"], "match_score": 0.81},
                {"start": 502.5, "end": 507.8, "movie_shot_ids": ["movie_shot_000113"], "match_score": 0.93},
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["start"], 500.5)
        self.assertEqual(merged[0]["end"], 507.8)
        self.assertEqual(merged[0]["match_score"], 0.93)

    def test_merges_cross_shot_overlapping_ranges(self) -> None:
        merged = merge_overlapping_movie_ranges(
            [
                {
                    "start": 500.5,
                    "end": 505.685,
                    "source_ref_shot_id": "ref_shot_017",
                    "movie_shot_ids": ["movie_shot_000112"],
                    "match_score": 0.6681,
                    "confidence": "medium",
                },
                {
                    "start": 500.5,
                    "end": 507.847,
                    "source_ref_shot_id": "ref_shot_019",
                    "movie_shot_ids": ["movie_shot_000113"],
                    "match_score": 0.9371,
                    "confidence": "high",
                },
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["start"], 500.5)
        self.assertEqual(merged[0]["end"], 507.847)
        self.assertEqual(merged[0]["source_ref_shot_id"], "ref_shot_019")
        self.assertEqual(merged[0]["movie_shot_ids"], ["movie_shot_000112", "movie_shot_000113"])

    def test_keeps_non_overlapping_ranges(self) -> None:
        merged = merge_overlapping_movie_ranges(
            [
                {"start": 387.5, "end": 394.8, "movie_shot_ids": ["movie_shot_000092"], "match_score": 0.93},
                {"start": 500.5, "end": 507.8, "movie_shot_ids": ["movie_shot_000113"], "match_score": 0.94},
            ]
        )
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
