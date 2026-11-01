"""Tests for row-level precision and recall."""
from __future__ import annotations

from dal_lab.workload import row_scores


def test_row_scores_separate_missing_from_invented_rows():
    gold = [(1, "a"), (2, "b"), (3, "c"), (4, "d")]
    assert row_scores(gold[:2], gold) == (1.0, 0.5)            # truncated
    assert row_scores(gold + [(9, "z")], gold) == (0.8, 1.0)    # one invented row
    assert row_scores([], gold) == (1.0, 0.0)


def test_row_scores_tolerate_float_rounding():
    assert row_scores([(1, 2.000001)], [(1, 2.0)]) == (1.0, 1.0)
