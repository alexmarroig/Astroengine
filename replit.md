# Premium Astrology API

## Overview
A Python backend API for a premium astrology app using FastAPI and Swiss Ephemeris (pyswisseph) for accurate astronomical calculations. The API provides natal chart calculations, transit analysis with aspect detection, and AI-powered cosmic insights via OpenAI.

## Project Structure
```
├── main.py                 # FastAPI application with all routes
├── requirements.txt        # Python dependencies
├── astro/
│   ├── __init__.py
│   ├── utils.py           # Julian day conversion, sign calculations, angle utilities
│   ├── ephemeris.py       # Swiss Ephemeris chart calculations
│   └── aspects.py         # Transit-to-natal aspect calculations
└── ai/
    ├── __init__.py
    └── prompts.py         # OpenAI message building with safety guidelines
```

## API Endpoints

### Health Check
- `GET /health` - Returns `{"ok": true}`

### Natal Chart
- `POST /v1/chart/natal` - Calculate natal chart with planetary positions and house cusps
  - Required: year, month, day, hour, lat, lng
  - Optional: minute, second, tz_offset_minutes, house_system

### Transits
- `POST /v1/chart/transits` - Calculate transits with aspects to natal chart
  - Required: natal birth data, lat, lng, target_date (YYYY-MM-DD)

### AI Cosmic Chat
- `POST /v1/ai/cosmic-chat` - AI-powered astrological guidance
  - Required: user_question, astro_payload
  - Optional: tone, language

### API Documentation
- `GET /docs` - Interactive Swagger documentation

## Technical Details

### Swiss Ephemeris
- Uses pyswisseph for precise planetary calculations
- Supports multiple house systems (Placidus, Koch, Whole Sign, etc.)
- Calculates Sun through Pluto positions

### Aspect Detection
- Major aspects: Conjunction, Opposition, Square, Trine, Sextile
- Default orbs: 6°, 6°, 5°, 5°, 4° respectively
- Influence classification: Intense, Challenging, Fluid

### AI Guidelines
- Non-deterministic language
- No fear-based predictions
- No medical/legal claims
- Multiple interpretations for ambiguous symbolism

## Environment Variables
- `OPENAI_API_KEY` - Required for the cosmic-chat endpoint

## Running the Server
The server runs on port 5000 using uvicorn:
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 5000
```
