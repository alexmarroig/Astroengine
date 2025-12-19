import os
import time
import uuid
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any, Dict, Literal, List
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict, model_validator
from openai import OpenAI

from astro.ephemeris import compute_chart, compute_transits, compute_moon_only
from astro.aspects import compute_transit_aspects
from ai.prompts import build_cosmic_chat_messages

from core.security import require_api_key_and_user
from core.cache import cache
from core.plans import is_trial_or_premium

# -----------------------------
# Load env
# -----------------------------
load_dotenv()

# -----------------------------
# Logging (structured)
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("astro-api")
logger.setLevel(LOG_LEVEL)
handler = logging.StreamHandler()
handler.setLevel(LOG_LEVEL)
logger.propagate = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "ts": datetime.utcnow().isoformat() + "Z",
            "msg": record.getMessage(),
        }
        for k in ("request_id", "path", "status", "latency_ms", "user_id"):
            if hasattr(record, k):
                payload[k] = getattr(record, k)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


handler.setFormatter(JsonFormatter())
logger.handlers = [handler]


def _log(level: str, message: str, **extra: Any) -> None:
    """Small wrapper to keep structured logging consistent."""
    log_method = getattr(logger, level)
    log_method(message, extra=extra)

# -----------------------------
# App
# -----------------------------
app = FastAPI(
    title="Premium Astrology API",
    description="Accurate astrological calculations using Swiss Ephemeris with AI-powered cosmic insights",
    version="1.1.1",
)

# -----------------------------
# CORS
# -----------------------------
origins = os.getenv("ALLOWED_ORIGINS", "*")
allowed = [o.strip() for o in origins.split(",")] if origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  # Authorization + X-User-Id
)

# -----------------------------
# Middleware: request_id + logging
# -----------------------------
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    request.state.request_id = request_id

    start = time.time()
    try:
        response = await call_next(request)
        latency_ms = int((time.time() - start) * 1000)

        extra = {
            "request_id": request_id,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": latency_ms,
        }
        _log("info", "request", **extra)

        response.headers["X-Request-Id"] = request_id
        return response

    except Exception:
        latency_ms = int((time.time() - start) * 1000)
        extra = {
            "request_id": request_id,
            "path": request.url.path,
            "status": 500,
            "latency_ms": latency_ms,
        }
        logger.error("unhandled_exception", exc_info=True, extra=extra)
        return JSONResponse(
            status_code=500,
            content={"detail": "Erro interno no servidor.", "request_id": request_id},
        )

# -----------------------------
# Exception handler: HTTPException
# -----------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", None)
    extra = {
        "request_id": request_id,
        "path": request.url.path,
        "status": exc.status_code,
        "latency_ms": None,
    }
    _log("warning", "http_exception", **extra)
    payload = {"detail": exc.detail}
    if request_id:
        payload["request_id"] = request_id
    return JSONResponse(status_code=exc.status_code, content=payload)

# -----------------------------
# Auth dependency
# -----------------------------
def get_auth(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
):
    auth = require_api_key_and_user(
        authorization=authorization,
        x_user_id=x_user_id,
        request_path=request.url.path,
    )
    return auth

# -----------------------------
# Models
# -----------------------------
class HouseSystem(str, Enum):
    PLACIDUS = "P"
    KOCH = "K"
    REGIOMONTANUS = "R"


class ZodiacType(str, Enum):
    TROPICAL = "tropical"
    SIDEREAL = "sidereal"


class NatalChartRequest(BaseModel):
    year: int = Field(..., ge=1800, le=2100)
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)
    hour: int = Field(..., ge=0, le=23)
    minute: int = Field(0, ge=0, le=59)
    second: int = Field(0, ge=0, le=59)
    lat: float = Field(..., ge=-89.9999, le=89.9999)
    lng: float = Field(..., ge=-180, le=180)
    tz_offset_minutes: Optional[int] = Field(
        None, ge=-840, le=840, description="Minutos de offset para o fuso. Se vazio, usa timezone."
    )
    timezone: Optional[str] = Field(
        None,
        description="Timezone IANA (ex.: America/Sao_Paulo). Se preenchido, substitui tz_offset_minutes",
    )
    house_system: HouseSystem = Field(default=HouseSystem.PLACIDUS)
    zodiac_type: ZodiacType = Field(default=ZodiacType.TROPICAL)
    ayanamsa: Optional[str] = Field(
        default=None, description="Opcional para zodíaco sideral (ex.: lahiri, fagan_bradley)",
    )
    strict_timezone: bool = Field(
        default=False,
        description="Quando true, rejeita horários ambíguos em transições de DST para evitar datas erradas.",
    )

    @model_validator(mode="after")
    def validate_tz(self):
        if self.tz_offset_minutes is None and not self.timezone:
            raise HTTPException(
                status_code=400,
                detail="Informe timezone IANA ou tz_offset_minutes para calcular o mapa.",
            )
        return self

class TransitsRequest(BaseModel):
    natal_year: int = Field(..., ge=1800, le=2100)
    natal_month: int = Field(..., ge=1, le=12)
    natal_day: int = Field(..., ge=1, le=31)
    natal_hour: int = Field(..., ge=0, le=23)
    natal_minute: int = Field(0, ge=0, le=59)
    natal_second: int = Field(0, ge=0, le=59)
    lat: float = Field(..., ge=-89.9999, le=89.9999)
    lng: float = Field(..., ge=-180, le=180)
    tz_offset_minutes: Optional[int] = Field(
        None, ge=-840, le=840, description="Minutos de offset para o fuso. Se vazio, usa timezone."
    )
    timezone: Optional[str] = Field(
        None,
        description="Timezone IANA (ex.: America/Sao_Paulo). Se preenchido, substitui tz_offset_minutes",
    )
    target_date: str = Field(..., description="YYYY-MM-DD")
    zodiac_type: ZodiacType = Field(default=ZodiacType.TROPICAL)
    ayanamsa: Optional[str] = Field(
        default=None, description="Opcional para zodíaco sideral (ex.: lahiri, fagan_bradley)",
    )
    strict_timezone: bool = Field(
        default=False,
        description="Quando true, rejeita horários ambíguos em transições de DST para evitar datas erradas.",
    )

    @model_validator(mode="after")
    def validate_tz(self):
        if self.tz_offset_minutes is None and not self.timezone:
            raise HTTPException(
                status_code=400,
                detail="Informe timezone IANA ou tz_offset_minutes para calcular trânsitos.",
            )
        return self

class CosmicChatRequest(BaseModel):
    user_question: str = Field(..., min_length=1)
    astro_payload: Dict[str, Any] = Field(...)
    tone: Optional[str] = None
    language: str = Field("pt-BR")

class CosmicWeatherResponse(BaseModel):
    date: str
    moon_phase: str
    moon_sign: str
    headline: str
    text: str


class CosmicWeatherRangeResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(alias="from")
    to: str
    items: List[CosmicWeatherResponse]

class RenderDataRequest(BaseModel):
    year: int
    month: int
    day: int
    hour: int
    minute: int = 0
    second: int = 0
    lat: float = Field(..., ge=-89.9999, le=89.9999)
    lng: float = Field(..., ge=-180, le=180)
    tz_offset_minutes: Optional[int] = Field(
        None, ge=-840, le=840, description="Minutos de offset para o fuso. Se vazio, usa timezone."
    )
    timezone: Optional[str] = Field(
        None,
        description="Timezone IANA (ex.: America/Sao_Paulo). Se preenchido, substitui tz_offset_minutes",
    )
    house_system: HouseSystem = Field(default=HouseSystem.PLACIDUS)
    zodiac_type: ZodiacType = Field(default=ZodiacType.TROPICAL)
    ayanamsa: Optional[str] = Field(
        default=None, description="Opcional para zodíaco sideral (ex.: lahiri, fagan_bradley)",
    )
    strict_timezone: bool = Field(
        default=False,
        description="Quando true, rejeita horários ambíguos em transições de DST para evitar datas erradas.",
    )

    @model_validator(mode="after")
    def validate_tz(self):
        if self.tz_offset_minutes is None and not self.timezone:
            raise HTTPException(
                status_code=400,
                detail="Informe timezone IANA ou tz_offset_minutes para renderização.",
            )
        return self


class TimezoneResolveRequest(BaseModel):
    datetime_local: datetime = Field(..., description="Data/hora local, ex.: 2025-12-19T14:30:00")
    timezone: str = Field(..., description="Timezone IANA, ex.: America/Sao_Paulo")
    strict_birth: bool = Field(
        default=False,
        description="Quando true, acusa horários ambíguos em transições de DST para dados de nascimento.",
    )


class SystemAlert(BaseModel):
    id: str
    severity: Literal["low", "medium", "high"]
    title: str
    body: str
    technical: Dict[str, Any] = Field(default_factory=dict)


class SystemAlertsResponse(BaseModel):
    date: str
    alerts: List[SystemAlert]


class NotificationsDailyResponse(BaseModel):
    date: str
    items: List[Dict[str, Any]]

# -----------------------------
# Helpers
# -----------------------------
def _parse_date_yyyy_mm_dd(s: str) -> tuple[int, int, int]:
    try:
        parsed = datetime.strptime(s, "%Y-%m-%d")
        return parsed.year, parsed.month, parsed.day
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato inválido de data. Use YYYY-MM-DD.")

def _moon_phase_4(phase_angle_deg: float) -> str:
    a = phase_angle_deg % 360
    if a < 45 or a >= 315:
        return "Nova"
    if 45 <= a < 135:
        return "Crescente"
    if 135 <= a < 225:
        return "Cheia"
    return "Minguante"

def _cw_text(phase: str, sign: str) -> str:
    options = [
        "O dia tende a favorecer mais presença emocional e escolhas com calma. Ajustes pequenos podem ter efeito grande.",
        "Pode ser um dia de observação interna. Priorize o essencial e evite decidir no pico da emoção.",
        "A energia pode ficar mais intensa em alguns momentos. Pausas curtas e ritmo consistente ajudam.",
    ]
    return options[hash(phase + sign) % len(options)]

def _now_yyyy_mm_dd() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _git_commit_hash() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _mercury_alert_for(date: str, lat: float, lng: float, tz_offset_minutes: int) -> Optional[SystemAlert]:
    """Return a Mercury retrograde alert for a validated YYYY-MM-DD date.

    The date is parsed through ``_parse_date_yyyy_mm_dd`` to guarantee a clear
    400 error for bad inputs (instead of silent ValueErrors) and to avoid
    duplicating manual ``split("-")`` logic. Keeping a single, validated parse
    path is the correct resolution for merge conflicts that show a second
    ``compute_transits`` call using ``date.split("-")``: we only need one
    transit computation after validation.
    """

    y, m, d = _parse_date_yyyy_mm_dd(date)
    chart = compute_transits(
        target_year=y,
        target_month=m,
        target_day=d,
        lat=lat,
        lng=lng,
        tz_offset_minutes=tz_offset_minutes,
    )
    mercury = chart.get("planets", {}).get("Mercury")
    if not mercury or mercury.get("speed") is None:
        return None

    retro = mercury.get("retrograde")
    if retro:
        return SystemAlert(
            id="mercury_retrograde",
            severity="medium",
            title="Mercúrio retrógrado",
            body="Mercúrio está em retrogradação. Revise comunicações e contratos com atenção.",
            technical={"mercury_speed": mercury.get("speed"), "mercury_lon": mercury.get("lon")},
        )
    return None


def _daily_notifications_payload(date: str, lat: float, lng: float, tz_offset_minutes: int) -> NotificationsDailyResponse:
    moon = compute_moon_only(date, tz_offset_minutes=tz_offset_minutes)
    phase = _moon_phase_4(moon["phase_angle_deg"])
    sign = moon["moon_sign"]

    items: List[Dict[str, Any]] = [
        {
            "type": "cosmic_weather",
            "title": f"Lua {phase} em {sign}",
            "body": _cw_text(phase, sign),
        }
    ]

    mercury_alert = _mercury_alert_for(date, lat, lng, tz_offset_minutes)
    if mercury_alert:
        items.append(
            {
                "type": "system_alert",
                "title": mercury_alert.title,
                "body": mercury_alert.body,
                "technical": mercury_alert.technical,
            }
        )

    return NotificationsDailyResponse(date=date, items=items)


def _tz_offset_for(
    date_time: datetime, timezone: Optional[str], fallback_minutes: Optional[int], strict: bool = False
) -> int:
    """Resolve timezone: prefer IANA name; fallback to explicit offset or UTC.

    When ``strict`` is True, detect ambiguous DST transitions and reject them with a
    helpful error so birth datetimes não fiquem "um dia antes" por causa de fuso mal
    resolvido.
    """

    if timezone:
        try:
            tzinfo = ZoneInfo(timezone)
        except ZoneInfoNotFoundError:
            raise HTTPException(status_code=400, detail=f"Timezone inválido: {timezone}")

        offset_fold0 = date_time.replace(tzinfo=tzinfo, fold=0).utcoffset()
        offset_fold1 = date_time.replace(tzinfo=tzinfo, fold=1).utcoffset()

        # Escolhe o offset padrão (compatível com o comportamento anterior)
        offset = offset_fold0 or offset_fold1
        if offset is None:
            raise HTTPException(status_code=400, detail=f"Timezone sem offset disponível: {timezone}")

        if strict and offset_fold0 and offset_fold1 and offset_fold0 != offset_fold1:
            # horário ambíguo na virada de DST
            opts = sorted({int(offset_fold0.total_seconds() // 60), int(offset_fold1.total_seconds() // 60)})
            raise HTTPException(
                status_code=400,
                detail={
                    "detail": "Horário ambíguo na transição de horário de verão.",
                    "offset_options_minutes": opts,
                    "hint": "Envie tz_offset_minutes explicitamente ou ajuste o horário local.",
                },
            )

        return int(offset.total_seconds() // 60)

    if fallback_minutes is not None:
        return fallback_minutes

    return 0


# -----------------------------
# Cache TTLs
# -----------------------------
TTL_NATAL_SECONDS = 30 * 24 * 3600
TTL_TRANSITS_SECONDS = 6 * 3600
TTL_RENDER_SECONDS = 30 * 24 * 3600
TTL_COSMIC_WEATHER_SECONDS = 6 * 3600


def _cosmic_weather_payload(
    date_str: str, timezone: Optional[str], tz_offset_minutes: Optional[int], user_id: str
) -> Dict[str, Any]:
    """Compute (or fetch) the cosmic weather payload for a single day."""

    _parse_date_yyyy_mm_dd(date_str)
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=12, minute=0, second=0)
    resolved_offset = _tz_offset_for(dt, timezone, tz_offset_minutes)

    cache_key = f"cw:{user_id}:{date_str}:{timezone}:{resolved_offset}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    moon = compute_moon_only(date_str, tz_offset_minutes=resolved_offset)
    phase = _moon_phase_4(moon["phase_angle_deg"])
    sign = moon["moon_sign"]

    payload = {
        "date": date_str,
        "moon_phase": phase,
        "moon_sign": sign,
        "headline": f"Lua {phase} em {sign}",
        "text": _cw_text(phase, sign),
    }

    cache.set(cache_key, payload, ttl_seconds=TTL_COSMIC_WEATHER_SECONDS)
    return payload

ROADMAP_FEATURES = {
    "notifications": {"status": "beta", "notes": "feed diário via API; push aguardando provedor"},
    "mercury_retrograde_alert": {
        "status": "beta",
        "notes": "alertas sistêmicos quando Mercúrio entrar/saír de retrogradação",
    },
    "life_cycles": {"status": "planned", "notes": "mapear ciclos de retorno e progressões"},
    "auto_timezone": {"status": "beta", "notes": "usa timezone IANA no payload ou resolver via endpoint"},
    "tests": {"status": "in_progress", "notes": "priorizar casos críticos de cálculo"},
}

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def root():
    """Lightweight root endpoint for uptime checks and Render probes."""
    return {
        "ok": True,
        "service": "astroengine",
        "version": app.version,
        "commit": _git_commit_hash(),
        "env": {"openai": bool(os.getenv("OPENAI_API_KEY")), "log_level": LOG_LEVEL},
    }


@app.get("/health")
async def health_check():
    return {"ok": True}


@app.get("/v1/system/roadmap")
async def roadmap():
    """Visão rápida do andamento das próximas funcionalidades."""
    return {"features": ROADMAP_FEATURES}


@app.post("/v1/time/resolve-tz")
async def resolve_timezone(body: TimezoneResolveRequest):
    resolved_offset = _tz_offset_for(
        body.datetime_local, body.timezone, fallback_minutes=None, strict=body.strict_birth
    )
    return {"tz_offset_minutes": resolved_offset}

@app.post("/v1/chart/natal")
async def natal(body: NatalChartRequest, request: Request, auth=Depends(get_auth)):
    try:
        dt = datetime(
            year=body.year,
            month=body.month,
            day=body.day,
            hour=body.hour,
            minute=body.minute,
            second=body.second,
        )
        tz_offset_minutes = _tz_offset_for(
            dt, body.timezone, body.tz_offset_minutes, strict=body.strict_timezone
        )

        cache_key = f"natal:{auth['user_id']}:{hash(body.model_dump_json())}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        chart = compute_chart(
            year=body.year,
            month=body.month,
            day=body.day,
            hour=body.hour,
            minute=body.minute,
            second=body.second,
            lat=body.lat,
            lng=body.lng,
            tz_offset_minutes=tz_offset_minutes,
            house_system=body.house_system.value,
            zodiac_type=body.zodiac_type.value,
            ayanamsa=body.ayanamsa,
        )

        cache.set(cache_key, chart, ttl_seconds=TTL_NATAL_SECONDS)
        return chart
    except Exception as e:
        logger.error(
            "natal_error",
            exc_info=True,
            extra={"request_id": getattr(request.state, "request_id", None), "path": request.url.path},
        )
        raise HTTPException(status_code=500, detail=f"Erro ao calcular mapa natal: {str(e)}")

@app.post("/v1/chart/transits")
async def transits(body: TransitsRequest, request: Request, auth=Depends(get_auth)):
    y, m, d = _parse_date_yyyy_mm_dd(body.target_date)

    try:
        natal_dt = datetime(
            year=body.natal_year,
            month=body.natal_month,
            day=body.natal_day,
            hour=body.natal_hour,
            minute=body.natal_minute,
            second=body.natal_second,
        )
        tz_offset_minutes = _tz_offset_for(natal_dt, body.timezone, body.tz_offset_minutes)

        cache_key = f"transits:{auth['user_id']}:{body.target_date}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        natal_chart = compute_chart(
            year=body.natal_year,
            month=body.natal_month,
            day=body.natal_day,
            hour=body.natal_hour,
            minute=body.natal_minute,
            second=body.natal_second,
            lat=body.lat,
            lng=body.lng,
            tz_offset_minutes=tz_offset_minutes,
            house_system="P",
            zodiac_type=body.zodiac_type.value,
            ayanamsa=body.ayanamsa,
        )

        transit_chart = compute_transits(
            target_year=y,
            target_month=m,
            target_day=d,
            lat=body.lat,
            lng=body.lng,
            tz_offset_minutes=tz_offset_minutes,
            zodiac_type=body.zodiac_type.value,
            ayanamsa=body.ayanamsa,
        )

        aspects = compute_transit_aspects(
            transit_planets=transit_chart["planets"],
            natal_planets=natal_chart["planets"],
        )

        moon = compute_moon_only(body.target_date, tz_offset_minutes=tz_offset_minutes)
        phase = _moon_phase_4(moon["phase_angle_deg"])
        sign = moon["moon_sign"]

        response = {
            "date": body.target_date,
            "cosmic_weather": {
                "moon_phase": phase,
                "moon_sign": sign,
                "headline": f"Lua {phase} em {sign}",
                "text": _cw_text(phase, sign),
            },
            "natal": natal_chart,
            "transits": transit_chart,
            "aspects": aspects,
        }

        cache.set(cache_key, response, ttl_seconds=TTL_TRANSITS_SECONDS)
        return response

    except Exception as e:
        logger.error(
            "transits_error",
            exc_info=True,
            extra={"request_id": getattr(request.state, "request_id", None), "path": request.url.path},
        )
        raise HTTPException(status_code=500, detail=f"Erro ao calcular trânsitos: {str(e)}")

@app.get("/v1/cosmic-weather", response_model=CosmicWeatherResponse)
async def cosmic_weather(
    request: Request,
    date: Optional[str] = None,
    timezone: Optional[str] = Query(None, description="Timezone IANA"),
    tz_offset_minutes: Optional[int] = Query(
        None, ge=-840, le=840, description="Offset manual em minutos; ignorado se timezone for enviado."
    ),
    auth=Depends(get_auth),
):
    d = date or _now_yyyy_mm_dd()
    payload = _cosmic_weather_payload(d, timezone, tz_offset_minutes, auth["user_id"])
    return CosmicWeatherResponse(**payload)


@app.get("/v1/cosmic-weather/range", response_model=CosmicWeatherRangeResponse)
async def cosmic_weather_range(
    request: Request,
    from_: str = Query(..., alias="from", description="Data inicial no formato YYYY-MM-DD"),
    to: str = Query(..., description="Data final no formato YYYY-MM-DD"),
    timezone: Optional[str] = Query(None, description="Timezone IANA"),
    tz_offset_minutes: Optional[int] = Query(
        None, ge=-840, le=840, description="Offset manual em minutos; ignorado se timezone for enviado."
    ),
    auth=Depends(get_auth),
):
    start_y, start_m, start_d = _parse_date_yyyy_mm_dd(from_)
    end_y, end_m, end_d = _parse_date_yyyy_mm_dd(to)

    start_date = datetime(year=start_y, month=start_m, day=start_d)
    end_date = datetime(year=end_y, month=end_m, day=end_d)

    if end_date < start_date:
        raise HTTPException(status_code=400, detail="Parâmetro 'from' deve ser anterior ou igual a 'to'.")

    interval_days = (end_date - start_date).days + 1
    if interval_days > 90:
        raise HTTPException(status_code=400, detail="Intervalo máximo permitido é de 90 dias.")

    items: List[CosmicWeatherResponse] = []
    current = start_date
    for _ in range(interval_days):
        date_str = current.strftime("%Y-%m-%d")
        payload = _cosmic_weather_payload(date_str, timezone, tz_offset_minutes, auth["user_id"])
        items.append(CosmicWeatherResponse(**payload))
        current += timedelta(days=1)

    return CosmicWeatherRangeResponse(from_=from_, to=to, items=items)

@app.post("/v1/chart/render-data")
async def render_data(body: RenderDataRequest, request: Request, auth=Depends(get_auth)):
    dt = datetime(
        year=body.year,
        month=body.month,
        day=body.day,
        hour=body.hour,
        minute=body.minute,
        second=body.second,
    )
    tz_offset_minutes = _tz_offset_for(dt, body.timezone, body.tz_offset_minutes)

    cache_key = f"render:{auth['user_id']}:{hash(body.model_dump_json())}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    natal = compute_chart(
        year=body.year,
        month=body.month,
        day=body.day,
        hour=body.hour,
        minute=body.minute,
        second=body.second,
        lat=body.lat,
        lng=body.lng,
        tz_offset_minutes=tz_offset_minutes,
        house_system=body.house_system.value,
        zodiac_type=body.zodiac_type.value,
        ayanamsa=body.ayanamsa,
    )

    cusps = natal.get("houses", {}).get("cusps")
    if not cusps or len(cusps) < 12:
        raise HTTPException(status_code=500, detail="Cálculo não retornou houses.cusps (12 valores).")

    houses = []
    for i in range(12):
        start = float(cusps[i])
        end = float(cusps[(i + 1) % 12])
        if end < start:
            end += 360.0
        houses.append({"house": i + 1, "start_deg": start, "end_deg": end})

    planets = []
    # seu compute_chart retorna planets como dict -> converte em lista útil pro front
    for name, p in natal.get("planets", {}).items():
        planets.append({
            "name": name,
            "sign": p.get("sign"),
            "deg_in_sign": p.get("deg_in_sign"),
            "angle_deg": p.get("lon"),
        })

    resp = {
        "zodiac": ["Áries","Touro","Gêmeos","Câncer","Leão","Virgem","Libra","Escorpião","Sagitário","Capricórnio","Aquário","Peixes"],
        "houses": houses,
        "planets": planets,
        "premium_aspects": [] if is_trial_or_premium(auth["plan"]) else None,
    }

    cache.set(cache_key, resp, ttl_seconds=TTL_RENDER_SECONDS)
    return resp

@app.post("/v1/ai/cosmic-chat")
async def cosmic_chat(body: CosmicChatRequest, request: Request, auth=Depends(get_auth)):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY não configurada no servidor.")

    language = body.language or "pt-BR"
    tone = body.tone or "calmo, adulto, tecnológico"

    try:
        client = OpenAI(api_key=api_key)

        messages = build_cosmic_chat_messages(
            user_question=body.user_question,
            astro_payload=body.astro_payload,
            tone=tone,
            language=language,
        )

        max_tokens_free = int(os.getenv("OPENAI_MAX_TOKENS_FREE", "600"))
        max_tokens_paid = int(os.getenv("OPENAI_MAX_TOKENS_PAID", "1100"))
        max_tokens = max_tokens_free if auth["plan"] == "free" else max_tokens_paid

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )

        return {
            "response": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    except Exception as e:
        logger.error(
            "cosmic_chat_error",
            exc_info=True,
            extra={"request_id": getattr(request.state, "request_id", None), "path": request.url.path},
        )
        raise HTTPException(status_code=500, detail=f"Erro no processamento de IA: {str(e)}")


@app.get("/v1/alerts/system", response_model=SystemAlertsResponse)
async def system_alerts(
    date: str,
    lat: float = Query(..., ge=-89.9999, le=89.9999),
    lng: float = Query(..., ge=-180, le=180),
    timezone: Optional[str] = Query(None, description="Timezone IANA"),
    tz_offset_minutes: Optional[int] = Query(None, ge=-840, le=840),
    auth=Depends(get_auth),
):
    _parse_date_yyyy_mm_dd(date)
    dt = datetime.strptime(date, "%Y-%m-%d").replace(hour=12, minute=0, second=0)
    resolved_offset = _tz_offset_for(dt, timezone, tz_offset_minutes)
    alerts: List[SystemAlert] = []

    mercury = _mercury_alert_for(date, lat, lng, resolved_offset)
    if mercury:
        alerts.append(mercury)

    return SystemAlertsResponse(date=date, alerts=alerts)


@app.get("/v1/notifications/daily", response_model=NotificationsDailyResponse)
async def notifications_daily(
    date: Optional[str] = None,
    lat: float = Query(..., ge=-89.9999, le=89.9999),
    lng: float = Query(..., ge=-180, le=180),
    timezone: Optional[str] = Query(None, description="Timezone IANA"),
    tz_offset_minutes: Optional[int] = Query(None, ge=-840, le=840),
    auth=Depends(get_auth),
):
    d = date or _now_yyyy_mm_dd()
    _parse_date_yyyy_mm_dd(d)
    dt = datetime.strptime(d, "%Y-%m-%d").replace(hour=12, minute=0, second=0)
    resolved_offset = _tz_offset_for(dt, timezone, tz_offset_minutes)

    cache_key = f"notif:{auth['user_id']}:{d}:{lat}:{lng}:{timezone}:{resolved_offset}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    payload = _daily_notifications_payload(d, lat, lng, resolved_offset)
    cache.set(cache_key, payload.model_dump(), ttl_seconds=TTL_COSMIC_WEATHER_SECONDS)
    return payload
