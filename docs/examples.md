# AstroAPI Examples

## Natal chart

```bash
curl -X POST "$API_URL/v1/chart/natal" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -H "X-User-Id: user_123" \
  -d '{
    "natal_year": 1995,
    "natal_month": 11,
    "natal_day": 7,
    "natal_hour": 22,
    "natal_minute": 56,
    "natal_second": 0,
    "lat": -23.5505,
    "lng": -46.6333,
    "timezone": "America/Sao_Paulo",
    "house_system": "P",
    "zodiac_type": "tropical"
  }'
```

## Transits

```bash
curl -X POST "$API_URL/v1/chart/transits" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -H "X-User-Id: user_123" \
  -d '{
    "natal_year": 1995,
    "natal_month": 11,
    "natal_day": 7,
    "natal_hour": 22,
    "natal_minute": 56,
    "natal_second": 0,
    "lat": -23.5505,
    "lng": -46.6333,
    "timezone": "America/Sao_Paulo",
    "target_date": "2026-01-09",
    "house_system": "P",
    "zodiac_type": "tropical"
  }'
```

## Retrogrades alerts

```bash
curl -X GET "$API_URL/v1/alerts/retrogrades?date=2024-01-01&timezone=Etc/UTC"
```
