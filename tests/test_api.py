
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.main import app, load_model, model_state

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"


@pytest.fixture(scope="module", autouse=True)
def _ensure_model_loaded():
    """Make sure the model is loaded before any test runs."""
    load_model(MODEL_PATH)
    assert model_state["model"] is not None, (
        "Model failed to load; run `python scripts/train_model.py` first."
    )
    yield
    load_model(MODEL_PATH)  # restore after any test that mutated state


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)



def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}



def test_ready_when_model_loaded(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True
    assert body["features"] == ["wt", "hp"]


def test_ready_returns_503_when_model_missing(client: TestClient) -> None:
    saved = model_state["model"]
    saved_error = model_state["error"]
    model_state["model"] = None
    model_state["error"] = "Model not loaded (simulated for test)"
    try:
        response = client.get("/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["model_loaded"] is False
        assert "simulated" in body["error"]
    finally:
        model_state["model"] = saved
        model_state["error"] = saved_error



def test_predict_returns_reasonable_mpg(client: TestClient) -> None:
   
    response = client.post("/predict", json={"wt": 2.62, "hp": 110})
    assert response.status_code == 200
    body = response.json()
    assert "predicted_mpg" in body
    assert 15 < body["predicted_mpg"] < 28
    assert body["features_used"] == ["wt", "hp"]
    assert set(body["model_coefficients"].keys()) == {"wt", "hp"}


def test_predict_heavy_car_lower_mpg_than_light_car(client: TestClient) -> None:
    light = client.post("/predict", json={"wt": 1.8, "hp": 100}).json()["predicted_mpg"]
    heavy = client.post("/predict", json={"wt": 5.0, "hp": 100}).json()["predicted_mpg"]
    assert heavy < light, "Heavier car should be predicted to have lower mpg"


def test_predict_rejects_missing_field(client: TestClient) -> None:
    response = client.post("/predict", json={"wt": 2.62})  # hp missing
    assert response.status_code == 422
    body = response.json()
    assert any("hp" in str(err.get("loc", [])) for err in body["detail"])


def test_predict_rejects_wrong_type(client: TestClient) -> None:
    response = client.post("/predict", json={"wt": "heavy", "hp": 110})
    assert response.status_code == 422


def test_predict_rejects_negative_weight(client: TestClient) -> None:
    response = client.post("/predict", json={"wt": -1.0, "hp": 110})
    assert response.status_code == 422


def test_predict_503_when_model_missing(client: TestClient) -> None:
    saved = model_state["model"]
    model_state["model"] = None
    model_state["error"] = "Model not loaded (simulated)"
    try:
        response = client.post("/predict", json={"wt": 2.62, "hp": 110})
        assert response.status_code == 503
    finally:
        model_state["model"] = saved
        model_state["error"] = None


def test_app_metadata() -> None:
    assert app_main.app.title == "MTCARS FastAPI"
