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


def test_natal_chart_test_vector_pt_br():
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
    }

    resp = client.post(
        "/v1/chart/natal?lang=pt-BR",
        json=payload,
        headers=_auth_headers(),
    )
    assert resp.status_code == 200

    chart = resp.json()
    sun = chart["planets"]["Sun"]
    moon = chart["planets"]["Moon"]

    assert sun["sign"] == "Escorpião"
    assert sun["sign_pt"] == "Escorpião"
    assert sun["deg_in_sign"] == pytest.approx(15.14, abs=0.5)

    assert moon["sign"] == "Touro"
    assert moon["sign_pt"] == "Touro"
