import json
from pathlib import Path

import pytest

from phcx.experiments.external import (
    load_diahalu,
    load_halluhard_seed,
    load_halueval_dialogue,
)
from phcx.experiments.llm_baselines import semantic_entropy_lexical_score, selfcheck_lexical_score

ROOT = Path(__file__).resolve().parent.parent


def test_diahalu_adapter_preserves_labels_and_turns():
    path = ROOT / "data/raw/diahalu_bench.jsonl"
    if not path.exists():
        pytest.skip("external snapshot not downloaded")
    rows = load_diahalu(path, limit=4)
    assert rows
    assert all(r["label"] in (0, 1) for r in rows)
    assert all(r["prompt"] and r["answer"] for r in rows)
    assert all(isinstance(r["conversation_history"], list) for r in rows)


def test_halueval_adapter_creates_paired_labels():
    path = ROOT / "data/raw/halueva_dialogue.json"
    if not path.exists():
        pytest.skip("external snapshot not downloaded")
    rows = load_halueval_dialogue(path, limit=2)
    assert len(rows) == 4
    assert {r["label"] for r in rows} == {0, 1}
    assert {r["metadata"]["pair_id"] for r in rows} == {0, 1}


def test_halluhard_adapter_is_explicitly_unlabeled(tmp_path):
    path = tmp_path / "halluhard.jsonl"
    path.write_text(json.dumps({"research_question": "What is tested?", "abstract": "A paper."}) + "\n")
    rows = load_halluhard_seed(path)
    assert len(rows) == 1
    assert rows[0]["label"] is None


def test_lexical_baseline_scores_are_bounded():
    assert 0.0 <= selfcheck_lexical_score("same answer", ["same answer", "same answer"]) <= 1.0
    assert 0.0 <= semantic_entropy_lexical_score(["one answer", "different response"]) <= 1.0
