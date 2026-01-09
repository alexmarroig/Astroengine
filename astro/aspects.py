from typing import List, Dict
from astro.utils import angle_diff

ASPECTS = {
    "conjunction": {"angle": 0, "orb": 6, "influence": "intense"},
    "opposition": {"angle": 180, "orb": 6, "influence": "challenging"},
    "square": {"angle": 90, "orb": 5, "influence": "challenging"},
    "trine": {"angle": 120, "orb": 5, "influence": "supportive"},
    "sextile": {"angle": 60, "orb": 4, "influence": "supportive"},
}


def compute_transit_aspects(
    transit_planets: Dict[str, dict],
    natal_planets: Dict[str, dict]
) -> List[dict]:
    aspects_found = []
    
    for t_name, t_data in transit_planets.items():
        t_lon = t_data["lon"]
        
        for n_name, n_data in natal_planets.items():
            n_lon = n_data["lon"]
            
            separation = angle_diff(t_lon, n_lon)
            
            for aspect_name, aspect_info in ASPECTS.items():
                target_angle = aspect_info["angle"]
                max_orb = aspect_info["orb"]
                
                orb = abs(separation - target_angle)
                
                if orb <= max_orb:
                    aspects_found.append({
                        "transit_planet": t_name,
                        "natal_planet": n_name,
                        "aspect": aspect_name,
                        "exact_angle": target_angle,
                        "actual_angle": round(separation, 4),
                        "orb": round(orb, 4),
                        "influence": aspect_info["influence"],
                    })
    
    aspects_found.sort(key=lambda x: x["orb"])
    
    return aspects_found
