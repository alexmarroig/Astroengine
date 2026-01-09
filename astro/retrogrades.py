from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import swisseph as swe

from astro.utils import to_julian_day

swe.set_ephe_path(".")

PLANET_CODES = {
    "mercury": swe.MERCURY,
    "venus": swe.VENUS,
    "mars": swe.MARS,
    "jupiter": swe.JUPITER,
    "saturn": swe.SATURN,
    "uranus": swe.URANUS,
    "neptune": swe.NEPTUNE,
    "pluto": swe.PLUTO,
}

MEANINGS_PT = {
    "mercury": "Durante este período, temas de comunicação, revisão e logística tendem a pedir mais cuidado e checagem.",
    "venus": "Durante este período, relações e valores pessoais pedem mais revisão e escolhas conscientes.",
    "mars": "Durante este período, a energia de ação pode desacelerar e pedir estratégia antes de agir.",
    "jupiter": "Durante este período, planos de expansão pedem ajustes e revisão de expectativas.",
    "saturn": "Durante este período, responsabilidades e estrutura pedem reavaliação cuidadosa.",
    "uranus": "Durante este período, mudanças internas pedem ritmo e adaptação gradual.",
    "neptune": "Durante este período, intuição e sensibilidade pedem clareza para evitar confusões.",
    "pluto": "Durante este período, processos de transformação pedem paciência e profundidade.",
}


def _speed_at(dt: datetime, planet_id: int) -> float:
    jd_ut = to_julian_day(dt)
    result, _ = swe.calc_ut(jd_ut, planet_id)
    return result[3]


def _station_search(start: datetime, planet_id: int, direction: int, max_days: int = 400) -> datetime:
    current = start
    for _ in range(max_days):
        speed = _speed_at(current, planet_id)
        if speed >= 0:
            return current
        current += timedelta(days=direction)
    return current


def retrograde_window(date_local: datetime, planet_id: int) -> Dict[str, Optional[datetime]]:
    speed = _speed_at(date_local, planet_id)
    if speed >= 0:
        return {"is_active": False, "start": None, "end": None}

    start = _station_search(date_local, planet_id, direction=-1)
    end = _station_search(date_local, planet_id, direction=1)
    return {"is_active": True, "start": start, "end": end}


def retrograde_alerts(date_local: datetime) -> List[Dict[str, Optional[str]]]:
    alerts = []
    for planet, planet_id in PLANET_CODES.items():
        window = retrograde_window(date_local, planet_id)
        if not window["is_active"]:
            continue
        alerts.append(
            {
                "planet": planet,
                "is_active": True,
                "start_date": window["start"].strftime("%Y-%m-%d") if window["start"] else None,
                "end_date": window["end"].strftime("%Y-%m-%d") if window["end"] else None,
                "shadow_start": None,
                "shadow_end": None,
                "meaning": MEANINGS_PT.get(planet, "Período de revisão e ajustes."),
            }
        )
    return alerts
