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


def _transits_payload():
    return {
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


def test_mercury_retrograde_endpoint():
    client = TestClient(main.app)
    payload = {
        "target_date": "2026-01-09",
        "lat": -23.5505,
        "lng": -46.6333,
        "timezone": "America/Sao_Paulo",
    }

    resp = client.post("/v1/insights/mercury-retrograde", json=payload, headers=_auth_headers())
    assert resp.status_code == 200
    body = resp.json()
    assert body["planet"] == "Mercury"
    assert isinstance(body["retrograde"], bool)


def test_dominant_theme_endpoint():
    client = TestClient(main.app)
    resp = client.post(
        "/v1/insights/dominant-theme",
        json=_transits_payload(),
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "theme" in body
    assert "summary" in body


def test_areas_activated_endpoint():
    client = TestClient(main.app)
    resp = client.post(
        "/v1/insights/areas-activated",
        json=_transits_payload(),
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body


def test_care_suggestion_endpoint():
    client = TestClient(main.app)
    resp = client.post(
        "/v1/insights/care-suggestion",
        json=_transits_payload(),
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "suggestion" in body
    assert "moon_phase" in body


def test_life_cycles_endpoint():
    client = TestClient(main.app)
    resp = client.post(
        "/v1/insights/life-cycles",
        json=_transits_payload(),
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "age_years" in body
    assert "items" in body
