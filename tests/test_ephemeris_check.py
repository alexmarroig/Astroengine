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


def test_ephemeris_check_compares_against_swiss_ephemeris():
    client = TestClient(main.app)
    payload = {
        "datetime_local": "2024-01-01T12:00:00",
        "timezone": "Etc/UTC",
        "lat": 0.0,
        "lng": 0.0,
    }

    resp = client.post("/v1/diagnostics/ephemeris-check", json=payload, headers=_auth_headers())
    assert resp.status_code == 200

    body = resp.json()
    assert body["tz_offset_minutes"] == 0
    assert body["items"], "payload must list planets compared"

    for item in body["items"]:
        assert item["delta_deg_abs"] < 0.05, item


def test_ephemeris_check_rejects_invalid_timezone():
    client = TestClient(main.app)
    payload = {
        "datetime_local": "2024-01-01T12:00:00",
        "timezone": "Mars/Crater",
        "lat": 0.0,
        "lng": 0.0,
    }

    resp = client.post("/v1/diagnostics/ephemeris-check", json=payload, headers=_auth_headers())
    assert resp.status_code == 400
