"""Tests for the Smoothed Shift forecaster."""

from datetime import UTC, datetime, timedelta

import pytest

from custom_components.hafo.forecasters.historical_shift import ForecastPoint
from custom_components.hafo.forecasters.smoothed_shift import (
    NEAR_TERM_OFFSETS_MIN,
    add_near_term_points,
    blend_forecast_with_recent,
    interpolate_to_quarter_hour,
)

_BASE = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)


def _make_forecast(values: list[float], base: datetime = _BASE, step: timedelta = timedelta(hours=1)) -> list[ForecastPoint]:
    return [ForecastPoint(time=base + i * step, value=v) for i, v in enumerate(values)]


# --- interpolate_to_quarter_hour ---


def test_interpolate_empty_returns_empty() -> None:
    assert interpolate_to_quarter_hour([]) == []


def test_interpolate_single_point_returned_as_is() -> None:
    pts = _make_forecast([100.0])
    result = interpolate_to_quarter_hour(pts)
    assert result == pts


def test_interpolate_two_hourly_points_produces_five_quarterly() -> None:
    """2 hourly points → 4 sub-steps from first + final point = 5 total."""
    pts = _make_forecast([0.0, 40.0])
    result = interpolate_to_quarter_hour(pts)
    assert len(result) == 5
    assert result[0].time == _BASE
    assert result[1].time == _BASE + timedelta(minutes=15)
    assert result[4].time == _BASE + timedelta(hours=1)


def test_interpolate_values_are_linearly_interpolated() -> None:
    pts = _make_forecast([0.0, 40.0])
    result = interpolate_to_quarter_hour(pts)
    assert result[0].value == pytest.approx(0.0)
    assert result[1].value == pytest.approx(10.0)
    assert result[2].value == pytest.approx(20.0)
    assert result[3].value == pytest.approx(30.0)
    assert result[4].value == pytest.approx(40.0)


def test_interpolate_negative_values_floored_at_zero() -> None:
    pts = _make_forecast([10.0, -10.0])
    result = interpolate_to_quarter_hour(pts)
    for point in result:
        assert point.value >= 0.0


def test_interpolate_n_hourly_produces_correct_count() -> None:
    """n hourly points → 4*(n-1)+1 quarterly points."""
    pts = _make_forecast([0.0, 10.0, 20.0])
    result = interpolate_to_quarter_hour(pts)
    assert len(result) == 4 * (3 - 1) + 1


# --- add_near_term_points ---


def _make_15min_forecast(n: int, base: datetime = _BASE) -> list[ForecastPoint]:
    step = timedelta(minutes=15)
    return [ForecastPoint(time=base + i * step, value=float(i * 10)) for i in range(n)]


def test_add_near_term_offsets_are_present() -> None:
    now = _BASE
    forecast = _make_15min_forecast(20, base=now)
    result = add_near_term_points(forecast, now)
    times = {(p.time - now).seconds // 60 for p in result}
    for offset in NEAR_TERM_OFFSETS_MIN:
        assert offset in times


def test_add_near_term_drops_points_within_cutoff() -> None:
    now = _BASE
    forecast = _make_15min_forecast(20, base=now)
    result = add_near_term_points(forecast, now)
    cutoff = now + timedelta(minutes=max(NEAR_TERM_OFFSETS_MIN))
    within = [p for p in result if p.time <= cutoff]
    # Should be exactly the near-term offsets, nothing extra
    assert len(within) == len(NEAR_TERM_OFFSETS_MIN)


def test_add_near_term_preserves_points_beyond_cutoff() -> None:
    now = _BASE
    forecast = _make_15min_forecast(20, base=now)
    beyond_before = [p for p in forecast if p.time > now + timedelta(minutes=max(NEAR_TERM_OFFSETS_MIN))]
    result = add_near_term_points(forecast, now)
    beyond_after = [p for p in result if p.time > now + timedelta(minutes=max(NEAR_TERM_OFFSETS_MIN))]
    assert beyond_after == beyond_before


def test_add_near_term_result_is_sorted() -> None:
    now = _BASE
    forecast = _make_15min_forecast(20, base=now)
    result = add_near_term_points(forecast, now)
    times = [p.time for p in result]
    assert times == sorted(times)


def test_add_near_term_interpolates_values() -> None:
    """Values at near-term points should be interpolated from the base forecast."""
    now = _BASE
    # Flat forecast at 100.0 — interpolated values should all be 100.0
    forecast = [ForecastPoint(time=now + timedelta(hours=i), value=100.0) for i in range(5)]
    result = add_near_term_points(forecast, now)
    near = [p for p in result if (p.time - now).total_seconds() <= max(NEAR_TERM_OFFSETS_MIN) * 60]
    for p in near:
        assert p.value == pytest.approx(100.0)


def test_add_near_term_empty_forecast_returns_unchanged() -> None:
    result = add_near_term_points([], _BASE)
    assert result == []


# --- blend_forecast_with_recent ---


def test_blend_empty_forecast_returns_empty() -> None:
    assert blend_forecast_with_recent([], recent_value=100.0, now=_BASE) == []


def test_blend_at_now_is_100pct_recent() -> None:
    """A point exactly at `now` should equal the recent anchor."""
    pts = [ForecastPoint(time=_BASE, value=500.0)]
    result = blend_forecast_with_recent(pts, recent_value=50.0, now=_BASE)
    assert result[0].value == pytest.approx(50.0)


def test_blend_beyond_duration_is_pure_historical() -> None:
    """Points at or beyond blend_duration are untouched historical values."""
    now = _BASE
    pts = [ForecastPoint(time=now + timedelta(minutes=30), value=200.0)]
    result = blend_forecast_with_recent(pts, recent_value=50.0, now=now, blend_duration=timedelta(minutes=30))
    assert result[0].value == pytest.approx(200.0)


def test_blend_midpoint_is_50pct_each() -> None:
    now = _BASE
    pts = [ForecastPoint(time=now + timedelta(minutes=15), value=200.0)]
    result = blend_forecast_with_recent(pts, recent_value=100.0, now=now, blend_duration=timedelta(minutes=30))
    assert result[0].value == pytest.approx(0.5 * 100.0 + 0.5 * 200.0)


def test_blend_timestamps_preserved() -> None:
    now = _BASE
    pts = _make_forecast([10.0, 20.0, 30.0], base=now, step=timedelta(minutes=10))
    result = blend_forecast_with_recent(pts, recent_value=5.0, now=now)
    for original, blended in zip(pts, result):
        assert blended.time == original.time


def test_blend_negative_result_floored_at_zero() -> None:
    now = _BASE
    pts = [ForecastPoint(time=now, value=0.0)]
    result = blend_forecast_with_recent(pts, recent_value=-100.0, now=now)
    assert result[0].value >= 0.0


def test_blend_weight_decreases_over_time() -> None:
    """Points further in time should have a smaller recent-anchor contribution."""
    now = _BASE
    pts = [ForecastPoint(time=now + timedelta(minutes=m), value=0.0) for m in [5, 10, 20]]
    result = blend_forecast_with_recent(pts, recent_value=100.0, now=now, blend_duration=timedelta(minutes=30))
    # With hist=0, result equals w_recent * 100; w_recent should decrease with time
    assert result[0].value > result[1].value > result[2].value
