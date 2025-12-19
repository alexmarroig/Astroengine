import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))
import main


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    yield


def _auth_headers():
    return {"Authorization": "Bearer test-key", "X-User-Id": "u1"}


def test_system_alerts_rejects_invalid_date():
    client = TestClient(main.app)
    resp = client.get(
        "/v1/alerts/system",
        params={
            "date": "2024-13-40",  # invalid, should trigger the shared parser
            "lat": 0,
            "lng": 0,
            "timezone": "Etc/UTC",
        },
        headers=_auth_headers(),
    )

    assert resp.status_code == 400
    assert "Formato inválido" in resp.json()["detail"]


def test_system_alerts_works_for_valid_date():
    client = TestClient(main.app)
    resp = client.get(
        "/v1/alerts/system",
        params={"date": "2024-01-01", "lat": 0, "lng": 0, "timezone": "Etc/UTC"},
        headers=_auth_headers(),
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["date"] == "2024-01-01"
    assert isinstance(body["alerts"], list)
