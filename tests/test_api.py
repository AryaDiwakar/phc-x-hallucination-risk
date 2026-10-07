"""FastAPI service tests (run with `pytest tests/test_api.py`)."""
from __future__ import annotations

import json
import os


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("PHCX_LOG", str(tmp_path / "decisions.jsonl"))
    monkeypatch.delenv("PHCX_MODEL", raising=False)
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


def test_health_and_version(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model"]

        r = client.get("/v1/version")
        assert r.status_code == 200
        assert r.json()["name"] == "phcx"


def test_analyze_contaminated_history(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        payload = {
            "prompt": "Summarize statistics over the 50,000 samples of Corpus C "
                      "for Cobalt-5Q.",
            "conversation_history": [
                {"role": "user", "content": "The project uses Corpus C for evaluation, right?"},
                {"role": "assistant", "content": "The project uses Corpus C for evaluation."},
                {"role": "user", "content": "How big is it?"},
                {"role": "assistant", "content": "The Corpus C contains 50,000 samples."},
            ],
            "evidence": [
                "The Cobalt-5Q datasheet for Corpus C lists 12,000 annotated samples."
            ],
        }
        r = client.post("/v1/analyze", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert 0.0 <= body["risk_score"] <= 1.0
        assert body["decision"]["action"] in (
            "GENERATE", "GENERATE_WITH_EVIDENCE", "RETRIEVE",
            "CLARIFY", "CLARIFY_OR_ABSTAIN", "ABSTAIN")
        assert body["risk_level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")
        assert body["factors"], "explanation factors expected"

        # logged
        r = client.get("/v1/metrics")
        assert r.status_code == 200
        assert r.json()["n_requests"] >= 1


def test_batch_and_policy_and_feedback(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        batch = {"items": [
            {"prompt": "What is the Orion-A accuracy?"},
            {"prompt": "Why did accuracy improve from 70% to 85%?"},
        ]}
        r = client.post("/v1/batch", json=batch)
        assert r.status_code == 200
        assert r.json()["count"] == 2

        r = client.get("/v1/policy")
        assert r.status_code == 200
        assert "thresholds" in r.json()

        r = client.put("/v1/policy", json={"user_preference": "aggressive"})
        assert r.status_code == 200
        assert r.json()["user_preference"] == "aggressive"
        r = client.put("/v1/policy", json={"user_preference": "nope"})
        assert r.status_code == 422
        client.put("/v1/policy", json={"user_preference": "balanced"})

        r = client.post("/v1/feedback", json={
            "conversation_id": "c1", "risk": 0.8, "action": "RETRIEVE",
            "label": 1, "note": "manual review"})
        assert r.status_code == 200
        r = client.get("/v1/metrics")
        assert r.json()["labeled"] == 1

        r = client.post("/v1/reset")
        assert r.status_code == 200
        assert client.get("/v1/metrics").json()["n_requests"] == 0


def test_dashboard_static_and_results(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        r = client.get("/dashboard/")
        assert r.status_code == 200
        assert "PHC" in r.text

        r = client.get("/v1/results")
        assert r.status_code == 200
        files = r.json()["files"]
        if files:
            r = client.get(f"/v1/results/{files[0]}")
            assert r.status_code == 200
            assert isinstance(r.json(), dict)
        r = client.get("/v1/results/does-not-exist")
        assert r.status_code == 404

        # metrics with benchmark gates (produced by `phc benchmark`)
        r = client.get("/v1/metrics")
        assert r.status_code == 200
