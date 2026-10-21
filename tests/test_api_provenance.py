"""Tests for the REST API's provenance fields.

The API's Turn model used to carry only role and content, so a caller could
not say "this assistant turn came from a verified source".  Every assistant
turn therefore looked model-generated and unverified, which silently
disabled the provenance machinery the whole framework is built on.
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.main as main_module

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    with TestClient(main_module.app) as c:
        yield c


def test_turn_model_accepts_provenance():
    turn = main_module.Turn(role="assistant", content="Atlas A has 12,000 rows.",
                            source="doc:orion-spec", verified=True)
    assert turn.source == "doc:orion-spec"
    assert turn.verified is True
    assert turn.retrieved is False


def test_history_turns_keep_their_provenance_through_the_api(client):
    """A verified turn must not be treated as an unverified model claim."""
    body = {
        "prompt": "How many samples does Atlas A contain?",
        "conversation_history": [
            {"role": "user", "content": "How big is Atlas A?"},
            {"role": "assistant", "content": "Atlas A contains 50,000 samples.",
             "source": "doc:orion-spec", "verified": True},
        ],
        "evidence": ["Atlas A evaluates the Orion model."],
    }
    res = client.post("/v1/analyze", json=body)
    assert res.status_code == 200, res.text
    payload = res.json()
    # the verified turn must not be reported as a contamination source
    assert payload["contamination"]["score"] == 0.0, (
        f"a source-verified turn was treated as unverified model memory: "
        f"{payload['contamination']}"
    )


def test_history_turn_without_provenance_is_flagged_as_unverified(client):
    """The same conversation without the source *is* contamination."""
    body = {
        "prompt": "How many samples does Atlas A contain?",
        "conversation_history": [
            {"role": "user", "content": "How big is Atlas A?"},
            {"role": "assistant", "content": "Atlas A contains 50,000 samples."},
        ],
        "evidence": ["Atlas A evaluates the Orion model."],
    }
    res = client.post("/v1/analyze", json=body)
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["contamination"]["score"] > 0.0, (
        f"an unsourced model claim was not reported as contamination: "
        f"{payload['contamination']}"
    )
