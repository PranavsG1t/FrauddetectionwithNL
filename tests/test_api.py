"""Step 2 tests. Replaces the step-1 scaffolding tests (the /demo/event
endpoint they covered no longer exists)."""
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from src.api.alert_service import AlertService, fallback_explanation
from src.api.main import PLACEHOLDER_EXPLANATION, app
from src.api.model_service import PROJECT_ROOT, default_model_dir
from src.api.schemas import TransactionFeatures
from src.data.preprocess import FEATURE_COLUMNS
from src.schema.fraud_event import FraudEvent

TEST_KEY = "test-key-123"
AUTH = {"X-API-Key": TEST_KEY}
ROUTINE, SUSPICIOUS = TransactionFeatures.model_config["json_schema_extra"]["examples"]

needs_models = pytest.mark.skipif(
    not (default_model_dir() / "lightgbm.joblib").exists(),
    reason="model artifacts not found",
)


@pytest.fixture(scope="module")
def client():
    mp = pytest.MonkeyPatch()
    mp.setenv("FRAUD_API_KEY", TEST_KEY)
    mp.setenv("ENABLE_ALERTS", "0")  # no Gemini in tests; fakes are injected below
    # `with` is what triggers lifespan (model loading). Without it, the
    # app starts "cold" with no models.
    with TestClient(app) as c:
        yield c
    mp.undo()


# --- public / infrastructure -------------------------------------------------

def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_cors_allows_dashboard_origin(client):
    r = client.options("/transactions/score", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_blocks_unknown_origin(client):
    r = client.options("/transactions/score", headers={
        "Origin": "https://evil.example",
        "Access-Control-Request-Method": "POST",
    })
    assert "access-control-allow-origin" not in r.headers


# --- auth + validation (don't need real models) ------------------------------

def test_missing_key_is_401(client):
    assert client.post("/transactions/score", json=ROUTINE).status_code == 401


def test_wrong_key_is_401(client):
    r = client.post("/transactions/score", json=ROUTINE, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_missing_field_is_422(client):
    bad = {k: v for k, v in ROUTINE.items() if k != "amt"}
    assert client.post("/transactions/score", json=bad, headers=AUTH).status_code == 422


def test_out_of_range_is_422(client):
    assert client.post("/transactions/score", json={**ROUTINE, "hour": 25}, headers=AUTH).status_code == 422
    assert client.post("/transactions/score", json={**ROUTINE, "category_fraud_rate": 1.5}, headers=AUTH).status_code == 422


def test_unknown_field_is_422(client):
    assert client.post("/transactions/score", json={**ROUTINE, "amount": 5}, headers=AUTH).status_code == 422


# --- real scoring ------------------------------------------------------------

@needs_models
def test_score_returns_full_fraud_event(client):
    r = client.post("/transactions/score", json=ROUTINE, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"event_type", "risk_score", "source_features",
                         "explanation_text", "recommended_action", "timestamp"}
    assert body["event_type"] == "transaction_fraud"
    assert 0.0 <= body["risk_score"] <= 1.0
    assert set(body["source_features"]) == set(FEATURE_COLUMNS)  # one SHAP value per feature


@needs_models
def test_timestamp_is_utc(client):
    ts = client.post("/transactions/score", json=ROUTINE, headers=AUTH).json()["timestamp"]
    assert ts.endswith("Z") or ts.endswith("+00:00")


@needs_models
def test_suspicious_scores_higher_than_routine(client):
    # Sanity check on the real model, not a law: if this fails, look at the
    # example values in schemas.py before blaming the code.
    lo = client.post("/transactions/score", json=ROUTINE, headers=AUTH).json()["risk_score"]
    hi = client.post("/transactions/score", json=SUSPICIOUS, headers=AUTH).json()["risk_score"]
    assert hi > lo


# --- pure logic --------------------------------------------------------------

def test_thresholds_map_to_actions():
    for score, action in {0.99: "block", 0.8: "manual_review", 0.1: "monitor"}.items():
        ev = FraudEvent.from_transaction({}, score, "x")
        assert ev.recommended_action == action


# --- Gemini alert wiring (all hermetic: fake generators, no network) ---------

def _fake_ok(shap, risk):
    return FraudEvent.from_transaction(shap, risk, "FAKE ALERT")


def _fake_down(shap, risk):
    raise RuntimeError("503 model overloaded")


@needs_models
def test_alerts_disabled_returns_placeholder(client):
    body = client.post("/transactions/score", json=ROUTINE, headers=AUTH).json()
    assert body["explanation_text"] == PLACEHOLDER_EXPLANATION


@needs_models
def test_explanation_comes_from_alert_service(client, monkeypatch):
    monkeypatch.setattr(app.state, "alerts", AlertService(_fake_ok))
    body = client.post("/transactions/score", json=ROUTINE, headers=AUTH).json()
    assert body["explanation_text"] == "FAKE ALERT"


@needs_models
def test_gemini_failure_degrades_to_shap_text_not_500(client, monkeypatch):
    monkeypatch.setattr(app.state, "alerts", AlertService(_fake_down))
    r = client.post("/transactions/score", json=SUSPICIOUS, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["explanation_text"].startswith("Automated explanation unavailable")
    assert 0.0 <= body["risk_score"] <= 1.0 and body["recommended_action"]


def test_fallback_text_names_top_features_by_impact():
    text = fallback_explanation({"a": 0.1, "b": -2.0, "c": 1.0, "d": 0.5})
    assert text.index("b (-2.00)") < text.index("c (+1.00)") < text.index("d (+0.50)")
    assert "a (" not in text  # only the top 3


def test_importing_main_does_not_create_gemini_client():
    # Load-order guard: alert_generator builds the Gemini client at import,
    # so importing the API module must never import it.
    code = "import sys, src.api.main; assert 'src.rag.alert_generator' not in sys.modules"
    r = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr