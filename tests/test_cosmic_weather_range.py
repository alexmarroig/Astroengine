import sys
from pathlib import Path
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))
import main  # noqa: E402


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    yield


def _auth_headers():
    return {"Authorization": "Bearer test-key", "X-User-Id": "u1"}


def test_range_returns_expected_length():
    client = TestClient(main.app)
    resp = client.get(
        "/v1/cosmic-weather/range",
        params={"from": "2024-01-01", "to": "2024-01-03", "timezone": "Etc/UTC"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["from"] == "2024-01-01"
    assert data["to"] == "2024-01-03"
    assert len(data["items"]) == 3
    assert [item["date"] for item in data["items"]] == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_range_rejects_invalid_date_format():
    client = TestClient(main.app)
    resp = client.get(
        "/v1/cosmic-weather/range",
        params={"from": "2024-13-01", "to": "2024-01-02", "timezone": "Etc/UTC"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    assert "Formato inválido" in resp.json()["detail"]


def test_range_rejects_from_after_to():
    client = TestClient(main.app)
    resp = client.get(
        "/v1/cosmic-weather/range",
        params={"from": "2024-02-01", "to": "2024-01-01", "timezone": "Etc/UTC"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 400


def test_range_rejects_over_90_days():
    client = TestClient(main.app)
    resp = client.get(
        "/v1/cosmic-weather/range",
        params={"from": "2024-01-01", "to": "2024-04-05", "timezone": "Etc/UTC"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 400


def test_range_accepts_manual_tz_offset():
    client = TestClient(main.app)
    resp = client.get(
        "/v1/cosmic-weather/range",
        params={"from": "2024-01-01", "to": "2024-01-02", "tz_offset_minutes": 0},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 2


def test_range_accepts_timezone_name():
    client = TestClient(main.app)
    resp = client.get(
        "/v1/cosmic-weather/range",
        params={"from": "2024-01-01", "to": "2024-01-02", "timezone": "America/Sao_Paulo"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 2


def test_range_requires_auth_headers():
    client = TestClient(main.app)
    resp = client.get(
        "/v1/cosmic-weather/range",
        params={"from": "2024-01-01", "to": "2024-01-02", "timezone": "Etc/UTC"},
    )
    assert resp.status_code == 401
