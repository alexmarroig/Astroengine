from datetime import datetime, timedelta
import swisseph as swe

from astro.utils import to_julian_day, deg_to_sign

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
    house_system: str = 'P'
) -> dict:
    local_dt = datetime(year, month, day, hour, minute, second)
    utc_dt = local_dt - timedelta(minutes=tz_offset_minutes)
    
    jd_ut = to_julian_day(utc_dt)
    
    house_system_bytes = house_system[0].upper().encode('ascii')
    cusps, ascmc = swe.houses(jd_ut, lat, lng, house_system_bytes)
    
    houses_data = {
        "system": HOUSE_SYSTEMS.get(house_system, house_system),
        "cusps": [round(c, 4) for c in cusps],
        "asc": round(ascmc[0], 4),
        "mc": round(ascmc[1], 4)
    }
    
    planets_data = {}
    for name, planet_id in PLANETS.items():
        result, ret_flag = swe.calc_ut(jd_ut, planet_id)
        lon = result[0]
        sign_info = deg_to_sign(lon)
        planets_data[name] = {
            "lon": round(lon, 4),
            "sign": sign_info["sign"],
            "deg_in_sign": sign_info["deg_in_sign"]
        }
    
    return {
        "utc_datetime": utc_dt.isoformat(),
        "jd_ut": round(jd_ut, 6),
        "houses": houses_data,
        "planets": planets_data
    }


def compute_transits(
    target_year: int,
    target_month: int,
    target_day: int,
    lat: float,
    lng: float,
    tz_offset_minutes: int = 0
) -> dict:
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
        house_system='P'
    )
