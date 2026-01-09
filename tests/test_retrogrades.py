import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    yield


def test_retrogrades_endpoint_schema():
    client = TestClient(main.app)
    resp = client.get(
        "/v1/alerts/retrogrades",
        params={"date": "2024-01-01", "timezone": "Etc/UTC"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "retrogrades" in body
    for item in body["retrogrades"]:
        assert set(item.keys()) == {
            "planet",
            "is_active",
            "start_date",
            "end_date",
            "shadow_start",
            "shadow_end",
            "meaning",
        }
