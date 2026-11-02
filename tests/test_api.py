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
        assert body["version"] == "0.2.0"
        assert body["model"]
        assert not body["model"].startswith("rf:"), (
            "the raw synthetic-trained checkpoint should not be the default scorer"
        )

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


def test_prompt_enhancement_is_local_grounded_and_reviewable(tmp_path, monkeypatch):
    import api.main as api_main

    monkeypatch.setattr(api_main, "_local_enhancer", lambda: None)
    with _client(tmp_path, monkeypatch) as client:
        response = client.post("/v1/enhance", json={
            "prompt": "How many participants were in the Northstar pilot?",
            "conversation_history": [
                {"role": "user", "content": "The Northstar pilot had 120 participants."},
                {"role": "assistant", "content": "The Northstar pilot had 120 participants."},
            ],
            "conversation_id": "enhance-test",
        })
        assert response.status_code == 200, response.text
        result = response.json()
        assert "Northstar pilot" in result["enhanced_prompt"]
        assert "120 participants" in result["enhanced_prompt"]
        assert result["context_support"]["source_role"] == "user"
        assert "estimated_risk" in result and "risk_reduced" in result
        assert "does not prove" in result["note"]

        math_result = client.post("/v1/enhance", json={
            "prompt": "What is 2 + 2?",
            "conversation_id": "enhance-math-test",
        }).json()
        assert math_result["estimated_risk"] == 0.0
        assert "Question: What is 2 + 2?" in math_result["enhanced_prompt"]
        assert "arithmetic" in math_result["enhanced_prompt"]

        conflict_result = client.post("/v1/enhance", json={
            "prompt": "For this chat, the Northstar pilot notes say 180 participants.",
            "conversation_history": [
                {"role": "user", "content": "For this chat, the Northstar pilot notes say 120 participants."},
                {"role": "assistant", "content": "The Northstar pilot had 120 participants."},
            ],
            "conversation_id": "enhance-conflict-test",
        }).json()
        assert "conflicting statements" in conflict_result["enhanced_prompt"]
        assert "Do not choose one as true" in conflict_result["enhanced_prompt"]
        assert conflict_result["context_support"]["context_conflict"]["new_values"] == ["180"]


def test_local_enhancer_instruction_is_fact_limited(monkeypatch):
    import api.main as api_main

    class FakeEnhancer:
        def __init__(self, content):
            self.content = content

        def generate(self, _prompt, max_tokens=80):
            return self.content

    diagnosis = {"direct_user_context_support": {"supported": True}}
    monkeypatch.setattr(
        api_main, "_local_enhancer",
        lambda: FakeEnhancer('{"instruction":"Use only supplied evidence and state uncertainty."}'),
    )
    assert api_main._validated_local_instruction(diagnosis)

    monkeypatch.setattr(
        api_main, "_local_enhancer",
        lambda: FakeEnhancer('{"instruction":"Use the 120 participant result."}'),
    )
    assert api_main._validated_local_instruction(diagnosis) is None


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
