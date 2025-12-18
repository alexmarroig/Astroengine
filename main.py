import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

from astro.ephemeris import compute_chart, compute_transits
from astro.aspects import compute_transit_aspects
from ai.prompts import build_cosmic_chat_messages

load_dotenv()

app = FastAPI(
    title="Premium Astrology API",
    description="Accurate astrological calculations using Swiss Ephemeris with AI-powered cosmic insights",
    version="1.0.0"
)


class NatalChartRequest(BaseModel):
    year: int = Field(..., ge=1800, le=2100, description="Birth year")
    month: int = Field(..., ge=1, le=12, description="Birth month")
    day: int = Field(..., ge=1, le=31, description="Birth day")
    hour: int = Field(..., ge=0, le=23, description="Birth hour (24h format)")
    minute: int = Field(0, ge=0, le=59, description="Birth minute")
    second: int = Field(0, ge=0, le=59, description="Birth second")
    lat: float = Field(..., ge=-90, le=90, description="Latitude of birth location")
    lng: float = Field(..., ge=-180, le=180, description="Longitude of birth location")
    tz_offset_minutes: int = Field(0, description="Timezone offset from UTC in minutes")
    house_system: str = Field("P", min_length=1, max_length=1, description="House system code (P=Placidus, K=Koch, etc.)")


class TransitsRequest(BaseModel):
    natal_year: int = Field(..., ge=1800, le=2100, description="Birth year")
    natal_month: int = Field(..., ge=1, le=12, description="Birth month")
    natal_day: int = Field(..., ge=1, le=31, description="Birth day")
    natal_hour: int = Field(..., ge=0, le=23, description="Birth hour")
    natal_minute: int = Field(0, ge=0, le=59, description="Birth minute")
    natal_second: int = Field(0, ge=0, le=59, description="Birth second")
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lng: float = Field(..., ge=-180, le=180, description="Longitude")
    tz_offset_minutes: int = Field(0, description="Timezone offset in minutes")
    target_date: str = Field(..., description="Target date for transits (YYYY-MM-DD)")


class CosmicChatRequest(BaseModel):
    user_question: str = Field(..., min_length=1, description="User's question")
    astro_payload: dict = Field(..., description="Astrological data (natal, transits, aspects)")
    tone: Optional[str] = Field(None, description="Optional tone modifier")
    language: str = Field("English", description="Response language")


@app.get("/health")
async def health_check():
    return {"ok": True}


@app.post("/v1/chart/natal")
async def calculate_natal_chart(request: NatalChartRequest):
    try:
        chart = compute_chart(
            year=request.year,
            month=request.month,
            day=request.day,
            hour=request.hour,
            minute=request.minute,
            second=request.second,
            lat=request.lat,
            lng=request.lng,
            tz_offset_minutes=request.tz_offset_minutes,
            house_system=request.house_system
        )
        return chart
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart calculation error: {str(e)}")


@app.post("/v1/chart/transits")
async def calculate_transits(request: TransitsRequest):
    try:
        target_parts = request.target_date.split("-")
        target_year = int(target_parts[0])
        target_month = int(target_parts[1])
        target_day = int(target_parts[2])
    except (ValueError, IndexError):
        raise HTTPException(status_code=400, detail="Invalid target_date format. Use YYYY-MM-DD")
    
    try:
        natal_chart = compute_chart(
            year=request.natal_year,
            month=request.natal_month,
            day=request.natal_day,
            hour=request.natal_hour,
            minute=request.natal_minute,
            second=request.natal_second,
            lat=request.lat,
            lng=request.lng,
            tz_offset_minutes=request.tz_offset_minutes,
            house_system='P'
        )
        
        transit_chart = compute_transits(
            target_year=target_year,
            target_month=target_month,
            target_day=target_day,
            lat=request.lat,
            lng=request.lng,
            tz_offset_minutes=request.tz_offset_minutes
        )
        
        aspects = compute_transit_aspects(
            transit_planets=transit_chart["planets"],
            natal_planets=natal_chart["planets"]
        )
        
        return {
            "natal": natal_chart,
            "transits": transit_chart,
            "aspects": aspects
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transit calculation error: {str(e)}")


@app.post("/v1/ai/cosmic-chat")
async def cosmic_chat(request: CosmicChatRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        client = OpenAI(api_key=api_key)
        
        messages = build_cosmic_chat_messages(
            user_question=request.user_question,
            astro_payload=request.astro_payload,
            tone=request.tone,
            language=request.language
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        return {
            "response": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI processing error: {str(e)}")
