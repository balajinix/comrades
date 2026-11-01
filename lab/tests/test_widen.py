"""Tests for the synthetically widened TPC-H schema."""
from __future__ import annotations

import pytest

from dal_lab.widen import padding

BASE = {"customer": 8, "orders": 9, "lineitem": 16, "part": 9, "supplier": 7,
        "partsupp": 5, "nation": 4, "region": 3}


def test_padding_reaches_the_requested_width():
    pads = padding(500, BASE)
    assert sum(BASE.values()) + sum(len(c) for c in pads.values()) == 500


def test_padding_names_are_unique_and_prefixed():
    for table, cols in padding(2000, BASE).items():
        names = [n for n, _ in cols]
        assert len(names) == len(set(names))
        assert all(n.startswith(("c_", "o_", "l_", "p_", "s_", "ps_", "n_", "r_"))
                   for n in names)


def test_padding_is_deterministic():
    assert padding(250, BASE) == padding(250, BASE)


def test_padding_rejects_narrowing():
    with pytest.raises(ValueError):
        padding(10, BASE)
