"""
Unit tests for FastAPI endpoints and ModelManager.
Run: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def mock_model_manager():
    mm = MagicMock()
    mm.loaded_versions.return_value = ["stable", "canary"]
    mm.ab_config = {"stable": 0.8, "canary": 0.2}
    mm.route_ab.return_value = "stable"
    mm.predict.return_value = ("landscape", 0.8734)
    return mm


@pytest.fixture
def client(mock_model_manager):
    from app.main import app
    app.state.model_manager = mock_model_manager
    with TestClient(app) as c:
        yield c


# ── Health endpoint ───────────────────────────────────────────────────────────
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "stable" in data["model_versions"]
    assert "canary" in data["model_versions"]
    assert data["ab_split"]["stable"] == 0.8


# ── Predict endpoint ──────────────────────────────────────────────────────────
def test_predict_returns_label(client):
    resp = client.post("/predict", json={
        "image_url": "https://example.com/test.jpg",
        "user_id": "user_123",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] == "landscape"
    assert 0 <= data["confidence"] <= 1
    assert data["model_version"] in ("stable", "canary")
    assert data["latency_ms"] >= 0


def test_predict_force_version(client, mock_model_manager):
    resp = client.post("/predict", json={
        "image_url": "https://example.com/img.jpg",
        "user_id":   "user_xyz",
        "force_version": "canary",
    })
    assert resp.status_code == 200
    # model_manager.predict was called with "canary"
    args = mock_model_manager.predict.call_args[0]
    assert args[0] == "canary"


def test_predict_error_handling(client, mock_model_manager):
    mock_model_manager.predict.side_effect = ValueError("Bad image URL")
    resp = client.post("/predict", json={
        "image_url": "not-a-real-url",
        "user_id": "user_err",
    })
    assert resp.status_code == 500
    assert "Bad image URL" in resp.json()["detail"]


# ── Rollback endpoint ─────────────────────────────────────────────────────────
def test_rollback(client, mock_model_manager):
    resp = client.post("/rollback")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ab_split"]["stable"] == 1.0
    assert data["ab_split"]["canary"] == 0.0


# ── Metrics endpoint ──────────────────────────────────────────────────────────
def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "predictions_total" in resp.text


# ── ModelManager A/B routing ─────────────────────────────────────────────────
def test_ab_routing_deterministic():
    """Same user_id should always hit the same model version."""
    from app.model_manager import ModelManager
    mm = ModelManager()
    mm.ab_config = {"stable": 0.8, "canary": 0.2}

    results = [mm.route_ab("consistent_user") for _ in range(20)]
    assert len(set(results)) == 1, "Same user must always hit same version"


def test_ab_routing_split():
    """Traffic split should be roughly 80/20 over many users."""
    from app.model_manager import ModelManager
    mm = ModelManager()
    mm.ab_config = {"stable": 0.8, "canary": 0.2}

    import random, string
    results = [mm.route_ab("user_" + "".join(random.choices(string.ascii_lowercase, k=8)))
               for _ in range(1000)]
    stable_pct = results.count("stable") / 1000
    assert 0.70 <= stable_pct <= 0.90, f"Expected ~80% stable, got {stable_pct:.0%}"


# ── Drift detector (smoke test) ───────────────────────────────────────────────
def test_drift_detector_runs():
    """Smoke test: drift_detector completes without raising an exception."""
    from monitoring.drift_detector import load_reference_data, load_current_data, run_drift_analysis
    ref  = load_reference_data("nonexistent_path.parquet")
    curr = load_current_data("nonexistent_path.parquet")
    results = run_drift_analysis(ref, curr)
    assert "dataset_drift_detected" in results
    assert "share_drifted_features" in results
    assert isinstance(results["share_drifted_features"], float)
