from datetime import datetime, timedelta
from typing import Literal, Optional

import swisseph as swe

from astro.utils import to_julian_day, deg_to_sign

# garante funcionamento em cloud mesmo sem ephemeris externa
swe.set_ephe_path(".")

PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mercury": swe.MERCURY,
    "Venus": swe.VENUS,
    "Mars": swe.MARS,
    "Jupiter": swe.JUPITER,
    "Saturn": swe.SATURN,
    "Uranus": swe.URANUS,
    "Neptune": swe.NEPTUNE,
    "Pluto": swe.PLUTO
}

HOUSE_SYSTEMS = {
    'P': 'Placidus',
    'K': 'Koch',
    'O': 'Porphyrius',
    'R': 'Regiomontanus',
    'C': 'Campanus',
    'E': 'Equal',
    'W': 'Whole Sign'
}


AYANAMSA_MAP = {
    "lahiri": swe.SIDM_LAHIRI,
    "krishnamurti": swe.SIDM_KRISHNAMURTI,
    "ramey": swe.SIDM_RAMAN,
    "fagan_bradley": swe.SIDM_FAGAN_BRADLEY,
}


def compute_chart(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
    lat: float,
    lng: float,
    tz_offset_minutes: int = 0,
    house_system: str = 'P',
    zodiac_type: Literal['tropical', 'sidereal'] = 'tropical',
    ayanamsa: Optional[str] = None,
) -> dict:
    local_dt = datetime(year, month, day, hour, minute, second)
    utc_dt = local_dt - timedelta(minutes=tz_offset_minutes)

    jd_ut = to_julian_day(utc_dt)

    # casas: fallback seguro
    warning = None
    house_system_code = house_system[0].upper() if house_system else "P"
    house_system_bytes = house_system_code.encode("ascii")

    try:
        cusps, ascmc = swe.houses(jd_ut, lat, lng, house_system_bytes)
    except Exception:
        # fallback para Placidus
        warning = "Sistema de casas ajustado automaticamente para Placidus por segurança."
        cusps, ascmc = swe.houses(jd_ut, lat, lng, b'P')
        house_system_code = "P"

    if zodiac_type == "sidereal":
        swe.set_sid_mode(AYANAMSA_MAP.get((ayanamsa or "lahiri").lower(), swe.SIDM_LAHIRI))

    houses_data = {
        "system": HOUSE_SYSTEMS.get(house_system_code, house_system_code),
        "cusps": [round(c, 6) for c in cusps],  # cusps[0]..cusps[11] (12 valores)
        "asc": round(ascmc[0], 6),
        "mc": round(ascmc[1], 6)
    }

    planets_data = {}
    for name, planet_id in PLANETS.items():
        result, _ = swe.calc_ut(jd_ut, planet_id)
        lon = result[0] % 360.0
        sign_info = deg_to_sign(lon)
        speed = result[3] if len(result) > 3 else None
        planets_data[name] = {
            "lon": round(lon, 6),
            "sign": sign_info["sign"],
            "deg_in_sign": round(sign_info["deg_in_sign"], 4),
            "speed": round(speed, 6) if speed is not None else None,
            "retrograde": bool(speed is not None and speed < 0),
        }

    payload = {
        "utc_datetime": utc_dt.isoformat(),
        "jd_ut": round(jd_ut, 8),
        "houses": houses_data,
        "planets": planets_data
    }
    if warning:
        payload["warning"] = warning

    return payload


def compute_transits(
    target_year: int,
    target_month: int,
    target_day: int,
    lat: float,
    lng: float,
    tz_offset_minutes: int = 0,
    zodiac_type: Literal['tropical', 'sidereal'] = 'tropical',
    ayanamsa: Optional[str] = None,
) -> dict:
    # 12:00 local como referência (estável)
    return compute_chart(
        year=target_year,
        month=target_month,
        day=target_day,
        hour=12,
        minute=0,
        second=0,
        lat=lat,
        lng=lng,
        tz_offset_minutes=tz_offset_minutes,
        house_system='P',
        zodiac_type=zodiac_type,
        ayanamsa=ayanamsa,
    )


def compute_moon_only(date_yyyy_mm_dd: str, tz_offset_minutes: int = 0) -> dict:
    """
    Retorna dados mínimos da Lua para Cosmic Weather:
    - longitude
    - signo (pt)
    - ângulo de fase Sol–Lua (0..360)

    Usa 12:00 local como referência para estabilidade diária.
    """
    try:
        year, month, day = map(int, date_yyyy_mm_dd.split("-"))
    except Exception:
        raise ValueError("Data inválida. Use YYYY-MM-DD.")

    # 12:00 local → UTC
    local_dt = datetime(year, month, day, 12, 0, 0)
    utc_dt = local_dt - timedelta(minutes=tz_offset_minutes)

    jd_ut = to_julian_day(utc_dt)

    moon_res, _ = swe.calc_ut(jd_ut, swe.MOON)
    moon_lon = moon_res[0] % 360.0

    sun_res, _ = swe.calc_ut(jd_ut, swe.SUN)
    sun_lon = sun_res[0] % 360.0

    phase_angle = (moon_lon - sun_lon) % 360.0

    sign_info = deg_to_sign(moon_lon)

    return {
        "utc_datetime": utc_dt.isoformat(),
        "moon_lon": round(moon_lon, 6),
        "moon_sign": sign_info["sign"],
        "deg_in_sign": round(sign_info["deg_in_sign"], 4),
        "phase_angle_deg": round(phase_angle, 4)
    }
