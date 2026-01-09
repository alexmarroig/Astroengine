from datetime import datetime

ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer",
    "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

ZODIAC_SIGNS_PT = [
    "Áries", "Touro", "Gêmeos", "Câncer",
    "Leão", "Virgem", "Libra", "Escorpião",
    "Sagitário", "Capricórnio", "Aquário", "Peixes"
]

SIGN_PT = dict(zip(ZODIAC_SIGNS, ZODIAC_SIGNS_PT))



def to_julian_day(dt: datetime) -> float:
    year = dt.year
    month = dt.month
    day = dt.day + (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0

    if month <= 2:
        year -= 1
        month += 12

    A = int(year / 100)
    B = 2 - A + int(A / 4)

    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    return jd


def deg_to_sign(lon: float) -> dict:
    lon = lon % 360
    sign_index = int(lon / 30)
    deg_in_sign = lon % 30
    return {
        "sign": ZODIAC_SIGNS[sign_index],
        "deg_in_sign": round(deg_in_sign, 4)
    }


def sign_to_pt(sign: str) -> str:
    return SIGN_PT.get(sign, sign)


def angle_diff(a: float, b: float) -> float:
    diff = abs(a - b) % 360
    if diff > 180:
        diff = 360 - diff
    return round(diff, 4)
