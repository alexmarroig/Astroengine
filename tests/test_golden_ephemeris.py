import math
from datetime import datetime, timedelta

import swisseph as swe

from astro.ephemeris import compute_chart, compute_moon_only
from main import _tz_offset_for


def _julian_day(utc_dt: datetime) -> float:
    return swe.julday(
        utc_dt.year,
        utc_dt.month,
        utc_dt.day,
        utc_dt.hour + utc_dt.minute / 60.0 + utc_dt.second / 3600.0,
    )


def _normalize(angle: float) -> float:
    return angle % 360.0


def _golden_moon_longitude(utc_dt: datetime) -> float:
    jd = _julian_day(utc_dt)
    result, _ = swe.calc_ut(jd, swe.MOON)
    return _normalize(result[0])


GOLDEN_CASES = [
    {
        "label": "UTC baseline",
        "local_dt": datetime(2024, 1, 1, 12, 0, 0),
        "timezone": "Etc/UTC",
        "lat": 0.0,
        "lng": 0.0,
    },
    {
        "label": "São Paulo DST-aware",
        "local_dt": datetime(2024, 7, 1, 12, 0, 0),
        "timezone": "America/Sao_Paulo",
        "lat": -23.55,
        "lng": -46.63,
    },
]


def test_compute_chart_matches_swiss_ephemeris_moon_longitude():
    """Validate moon longitude against Swiss Ephemeris for fixed golden cases."""

    for case in GOLDEN_CASES:
        offset = _tz_offset_for(case["local_dt"], case["timezone"], fallback_minutes=None)
        utc_dt = case["local_dt"] - timedelta(minutes=offset)

        golden_moon_lon = _golden_moon_longitude(utc_dt)

        chart = compute_chart(
            year=case["local_dt"].year,
            month=case["local_dt"].month,
            day=case["local_dt"].day,
            hour=case["local_dt"].hour,
            minute=case["local_dt"].minute,
            second=case["local_dt"].second,
            lat=case["lat"],
            lng=case["lng"],
            tz_offset_minutes=offset,
        )

        moon_payload_lon = chart["planets"]["Moon"]["lon"]
        assert math.isclose(moon_payload_lon, golden_moon_lon, rel_tol=0, abs_tol=0.01), case["label"]


def test_compute_moon_only_sign_matches_chart_sign():
    """compute_moon_only should align with full chart for the same date/offset."""

    for case in GOLDEN_CASES:
        offset = _tz_offset_for(case["local_dt"], case["timezone"], fallback_minutes=None)

        moon_only = compute_moon_only(
            case["local_dt"].strftime("%Y-%m-%d"), tz_offset_minutes=offset
        )

        chart = compute_chart(
            year=case["local_dt"].year,
            month=case["local_dt"].month,
            day=case["local_dt"].day,
            hour=case["local_dt"].hour,
            minute=case["local_dt"].minute,
            second=case["local_dt"].second,
            lat=case["lat"],
            lng=case["lng"],
            tz_offset_minutes=offset,
        )

        assert moon_only["moon_sign"] == chart["planets"]["Moon"]["sign"], case["label"]
