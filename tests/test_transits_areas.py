import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    yield


def _auth_headers():
    return {"Authorization": "Bearer test-key", "X-User-Id": "u1"}


def test_transits_includes_areas_activated():
    client = TestClient(main.app)
    payload = {
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
    }

    resp = client.post("/v1/chart/transits", json=payload, headers=_auth_headers())
    assert resp.status_code == 200
    body = resp.json()
    areas = body["areas_activated"]
    assert len(areas) == 4
    for item in areas:
        assert item["level"] in {"low", "medium", "high", "intense"}
        assert 0 <= item["score"] <= 100
        assert item["reason"]
