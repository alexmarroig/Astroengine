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


def test_resolve_timezone_utc():
    client = TestClient(main.app)
    payload = {
        "year": 2024,
        "month": 12,
        "day": 1,
        "hour": 12,
        "minute": 0,
        "second": 0,
        "timezone": "Etc/UTC",
    }
    resp = client.post("/v1/time/resolve-tz", json=payload)
    assert resp.status_code == 200
    assert resp.json()["tz_offset_minutes"] == 0


def test_resolve_timezone_dst_difference():
    """Ensure DST-aware offsets come from timezone data, not hardcoded minutes."""
    client = TestClient(main.app)
    winter = {
        "year": 2024,
        "month": 1,
        "day": 15,
        "hour": 12,
        "minute": 0,
        "second": 0,
        "timezone": "America/New_York",
    }
    summer = {
        "year": 2024,
        "month": 7,
        "day": 15,
        "hour": 12,
        "minute": 0,
        "second": 0,
        "timezone": "America/New_York",
    }

    resp_winter = client.post("/v1/time/resolve-tz", json=winter)
    resp_summer = client.post("/v1/time/resolve-tz", json=summer)

    assert resp_winter.status_code == 200
    assert resp_summer.status_code == 200
    assert resp_winter.json()["tz_offset_minutes"] == -300  # UTC-5
    assert resp_summer.json()["tz_offset_minutes"] == -240  # UTC-4 (DST)


def test_cosmic_weather_accepts_timezone():
    client = TestClient(main.app)
    resp = client.get(
        "/v1/cosmic-weather",
        params={"date": "2024-01-01", "timezone": "Etc/UTC"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["date"] == "2024-01-01"
    assert body["moon_sign"]
    assert body["moon_phase"]
