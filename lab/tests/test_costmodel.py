"""Tests for the analytic cost model."""
from __future__ import annotations

import math

from dal_lab.costmodel import (ShippingModel, cumulative_input, cumulative_uniform,
                               handle_crossover_sf)


def test_cumulative_input_matches_closed_form_without_cache():
    # Arrange
    base, s, calls = 1000, 200, 7
    # Act
    got = cumulative_input(base, [s] * calls)
    # Assert
    assert got == cumulative_uniform(base, s, calls)


def test_cumulative_uniform_is_quadratic_in_calls():
    # Arrange / Act
    c10 = cumulative_uniform(0, 100, 10)
    c20 = cumulative_uniform(0, 100, 20)
    # Assert: doubling calls roughly quadruples the step term
    assert 3.5 < c20 / c10 < 4.0


def test_cumulative_input_cache_lowers_cost():
    # Arrange
    steps = [500] * 10
    # Act
    plain = cumulative_input(2000, steps)
    cached = cumulative_input(2000, steps, cache_hit=0.9, cache_discount=0.1)
    # Assert
    assert cached < 0.5 * plain


def test_shipping_model_calibrate_separates_constant_tables():
    # Arrange
    rows = {"0:nation": 25, "1:orders": 150}
    tpr = {"nation": 20.0, "orders": 40.0}
    # Act
    m = ShippingModel.calibrate(3000, rows, tpr, sf=0.01)
    # Assert
    assert m.const_tokens == 500
    assert math.isclose(m.linear_tokens_per_sf, 600_000)


def test_overflow_sf_is_where_context_reaches_budget():
    # Arrange
    m = ShippingModel(base=3000, const_tokens=0, linear_tokens_per_sf=1_000_000,
                      linear_rows_per_sf=25_000, page=1000, call_overhead=0)
    # Act
    sf_star = m.overflow_sf(128_000)
    # Assert
    assert math.isclose(sf_star, 0.125)
    assert m.context_tokens(sf_star * 1.01) > 128_000 > m.context_tokens(sf_star * 0.99)


def test_handle_crossover_sf_infinite_for_constant_answers():
    # Arrange / Act / Assert
    assert math.isinf(handle_crossover_sf(500, 0))
    assert handle_crossover_sf(500, 10_000) == 0.05
