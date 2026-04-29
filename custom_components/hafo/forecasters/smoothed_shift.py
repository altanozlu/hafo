"""Smoothed shift forecaster.

Like the historical shift forecaster, but adds a dense set of near-term forecast
points (at 1, 2, 5, 10, 15, 20, 25, 30 minute offsets from now) and blends those
toward the current observed value. The blend decays linearly over 30 minutes so
the recent anchor has zero influence on points further than 30 min away.
"""

import logging
from datetime import datetime, timedelta

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from custom_components.hafo.const import CONF_HISTORY_DAYS, CONF_SOURCE_ENTITY, DEFAULT_HISTORY_DAYS, DOMAIN

from .historical_shift import ForecastPoint, ForecastResult, get_statistics_for_sensor, shift_history_to_forecast

_LOGGER = logging.getLogger(__name__)

_QUARTER_HOUR = timedelta(minutes=15)

# Near-term forecast offsets (minutes from now) inserted at full resolution
NEAR_TERM_OFFSETS_MIN: list[int] = [1, 2, 5, 10, 15, 20, 25, 30]

# The blend fades to zero at this distance from now
_BLEND_DURATION = timedelta(minutes=30)


def interpolate_to_quarter_hour(hourly: list[ForecastPoint]) -> list[ForecastPoint]:
    """Linearly interpolate hourly forecast points to 15-minute resolution.

    For each pair of consecutive hourly points produces 4 quarterly sub-steps via
    linear interpolation, then appends the final hourly point as-is.
    """
    if len(hourly) < 2:
        return list(hourly)
    result: list[ForecastPoint] = []
    for i in range(len(hourly) - 1):
        a, b = hourly[i], hourly[i + 1]
        for q in range(4):
            t = a.time + q * _QUARTER_HOUR
            v = a.value + (b.value - a.value) * (q / 4)
            result.append(ForecastPoint(time=t, value=max(0.0, v)))
    last = hourly[-1]
    result.append(ForecastPoint(time=last.time, value=max(0.0, last.value)))
    return result


def _interpolate_value_at(forecast: list[ForecastPoint], t: datetime) -> float:
    """Return the linearly interpolated value from `forecast` at time `t`."""
    if not forecast:
        return 0.0
    if t <= forecast[0].time:
        return forecast[0].value
    if t >= forecast[-1].time:
        return forecast[-1].value
    for i in range(len(forecast) - 1):
        a, b = forecast[i], forecast[i + 1]
        if a.time <= t <= b.time:
            span = (b.time - a.time).total_seconds()
            frac = (t - a.time).total_seconds() / span if span else 0.0
            return max(0.0, a.value + (b.value - a.value) * frac)
    return forecast[-1].value  # unreachable but satisfies type checker


def add_near_term_points(
    forecast: list[ForecastPoint],
    now: datetime,
    offsets_min: list[int] = NEAR_TERM_OFFSETS_MIN,
) -> list[ForecastPoint]:
    """Replace the first 30 minutes of `forecast` with dense near-term points.

    Existing forecast points within [now, now + max(offsets)] are dropped and
    replaced by points at the given minute offsets, with values interpolated from
    the remaining (15-min) forecast series.
    """
    if not forecast or not offsets_min:
        return forecast
    cutoff = now + timedelta(minutes=max(offsets_min))
    beyond = [p for p in forecast if p.time > cutoff]
    near_term = [
        ForecastPoint(
            time=now + timedelta(minutes=m),
            value=_interpolate_value_at(forecast, now + timedelta(minutes=m)),
        )
        for m in offsets_min
    ]
    return sorted(near_term + beyond, key=lambda p: p.time)


def blend_forecast_with_recent(
    forecast: list[ForecastPoint],
    recent_value: float,
    now: datetime,
    blend_duration: timedelta = _BLEND_DURATION,
) -> list[ForecastPoint]:
    """Blend forecast values toward `recent_value` with time-based linear decay.

    Points at `now` receive 100 % of `recent_value`; points at or beyond
    `now + blend_duration` are pure historical. The blend is proportional to
    how far the point is within the blend window.
    """
    if not forecast:
        return []
    blend_seconds = blend_duration.total_seconds()
    blended = []
    for point in forecast:
        elapsed = (point.time - now).total_seconds()
        w_recent = max(0.0, 1.0 - elapsed / blend_seconds) if blend_seconds > 0 else 0.0
        value = w_recent * recent_value + (1.0 - w_recent) * point.value
        blended.append(ForecastPoint(time=point.time, value=max(0.0, value)))
    return blended


class SmoothedShiftForecaster(DataUpdateCoordinator[ForecastResult | None]):
    """Forecaster that shifts historical statistics forward and smooths the transition.

    Fetches hourly recorder statistics, shifts them forward by N days, interpolates
    to 15-minute resolution, then prepends a dense set of near-term points (at 1,
    2, 5, 10, 15, 20, 25, 30 minute offsets). The whole near-term window is blended
    toward the current entity state with a linear decay over 30 minutes.
    """

    UPDATE_INTERVAL = timedelta(hours=1)

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self._entry = entry
        self._source_entity: str = entry.data[CONF_SOURCE_ENTITY]
        self._history_days: int = int(
            entry.options.get(CONF_HISTORY_DAYS, entry.data.get(CONF_HISTORY_DAYS, DEFAULT_HISTORY_DAYS))
        )
        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN}_{entry.entry_id}",
            update_interval=self.UPDATE_INTERVAL,
            config_entry=entry,
        )

    @property
    def source_entity(self) -> str:
        return self._source_entity

    @property
    def history_days(self) -> int:
        return self._history_days

    @property
    def entry(self) -> ConfigEntry:
        return self._entry

    async def _async_update_data(self) -> ForecastResult:
        result = await self._generate_forecast()
        _LOGGER.debug(
            "Generated smoothed forecast for %s with %d points",
            self._source_entity,
            len(result.forecast),
        )
        return result

    async def _generate_forecast(self) -> ForecastResult:
        now = dt_util.now()
        start_time = now - timedelta(days=self._history_days)

        statistics = await get_statistics_for_sensor(self.hass, self._source_entity, start_time, now)

        if not statistics:
            msg = f"No historical data available for {self._source_entity}"
            raise ValueError(msg)

        hourly_forecast = shift_history_to_forecast(statistics, self._history_days)

        if not hourly_forecast:
            msg = f"No valid forecast points generated for {self._source_entity}"
            raise ValueError(msg)

        forecast = interpolate_to_quarter_hour(hourly_forecast)
        forecast = add_near_term_points(forecast, now)

        state = self.hass.states.get(self._source_entity)
        try:
            recent_anchor = max(0.0, float(state.state)) if state else 0.0
        except (ValueError, TypeError):
            recent_anchor = 0.0

        forecast = blend_forecast_with_recent(forecast, recent_anchor, now)

        return ForecastResult(
            forecast=forecast,
            source_entity=self._source_entity,
            history_days=self._history_days,
            generated_at=now,
        )

    def cleanup(self) -> None:
        _LOGGER.debug("Cleaning up smoothed forecaster for %s", self._source_entity)
