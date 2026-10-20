"""Tests for the human-validation analysis and the synthetic-annotator simulator.

The simulator is a rehearsal, not a study: it exists so that the analysis
pipeline is exercised (and its weaknesses found) before real annotators
exist.  These tests pin that it is deterministic, blind, and correctly
labelled -- and that the analysis it feeds works.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from phcx.experiments.human_validation import (
    RUBRIC_QUESTIONS,
    analyze_annotations,
    krippendorff_alpha,
    simulate_annotations,
)

ROOT = Path(__file__).resolve().parents[1]
SIM_SCRIPT = ROOT / "scripts" / "simulate_human_annotations.py"


# ------------------------------------------------------------ krippendorff
def test_krippendorff_alpha_is_one_for_perfect_agreement():
    # two distinct values, each rated identically by every rater
    ratings = {i: [1, 1, 1] for i in range(10)}
    ratings.update({i: [2, 2, 2] for i in range(10, 20)})
    assert krippendorff_alpha(ratings) == pytest.approx(1.0, abs=1e-6)


def test_krippendorff_alpha_is_low_for_inconsistent_ratings():
    ratings = {i: ([1, 1, 1] if i % 2 else [2, 2, 2]) for i in range(20)}
    noisy = {i: ([1, 2, 3, 1, 2, 3] if i % 2 else [1, 1, 2, 1, 2, 1]) for i in range(20)}
    assert krippendorff_alpha(noisy) < krippendorff_alpha(ratings)


def test_krippendorff_alpha_is_undefined_for_a_single_value():
    """With only one value in play there is no disagreement to measure."""
    assert krippendorff_alpha({i: [1, 1, 1] for i in range(10)}) is None


def test_krippendorff_alpha_returns_none_without_repeated_ratings():
    assert krippendorff_alpha({"a": [1], "b": [2]}) is None


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def packet_and_key(tmp_path_factory):
    """A small packet and key, built by the real preparation script."""
    base = tmp_path_factory.mktemp("hvsim")
    packet = base / "packet.jsonl"
    key = base / "key.jsonl"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "prepare_human_validation.py"),
         "--out", str(packet), "--key-out", str(key),
         "--n", "30", "--duplicates", "5", "--pool-limit", "120"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in packet.read_text().splitlines() if line.strip()]
    keys = [json.loads(line) for line in key.read_text().splitlines() if line.strip()]
    if keys and "annotation_id" not in keys[0]:
        keys = keys[1:]
    return rows, keys


# -------------------------------------------------------------- simulation
def test_simulation_produces_the_requested_number_of_annotations(packet_and_key):
    packet, key = packet_and_key
    ann = simulate_annotations(packet, key, n_annotators=20, n_annotations=200, seed=42)
    assert len(ann) == 200
    assert len({a["annotator_code"] for a in ann}) > 1


def test_simulation_is_deterministic(packet_and_key):
    packet, key = packet_and_key
    a = simulate_annotations(packet, key, n_annotators=10, n_annotations=120, seed=7)
    b = simulate_annotations(packet, key, n_annotators=10, n_annotations=120, seed=7)
    assert a == b


def test_simulation_answers_every_rubric_question(packet_and_key):
    packet, key = packet_and_key
    ann = simulate_annotations(packet, key, n_annotators=10, n_annotations=150, seed=1)
    for a in ann:
        for q in RUBRIC_QUESTIONS:
            assert a.get(q) is not None
        assert a["supported_by_context"] in ("yes", "no", "uncertain")
        assert a["would_allow_generation"] in ("allow", "retrieve", "clarify", "abstain")
        assert 1 <= a["confidence"] <= 5
        assert a["synthetic"] is True


def test_simulation_never_leaks_the_gold_label(packet_and_key):
    packet, key = packet_and_key
    ann = simulate_annotations(packet, key, n_annotators=10, n_annotations=120, seed=3)
    gold_keys = {"label", "gold_label", "risk_score", "action", "action_group"}
    for a in ann:
        assert not (set(a.keys()) & gold_keys)


def test_simulation_gives_every_item_several_labels(packet_and_key):
    """Agreement is only computable if items are rated more than once."""
    packet, key = packet_and_key
    ann = simulate_annotations(packet, key, n_annotators=20, n_annotations=400, seed=5)
    counts = {}
    for a in ann:
        counts[a["annotation_id"]] = counts.get(a["annotation_id"], 0) + 1
    assert min(counts.values()) >= 2


# ---------------------------------------------------------------- analysis
def test_analysis_produces_every_field_the_protocol_asks_for(packet_and_key):
    packet, key = packet_and_key
    ann = simulate_annotations(packet, key, n_annotators=20, n_annotations=400, seed=11)
    report = analyze_annotations(ann, key)

    assert report["n_annotations"] == 400
    assert report["n_items"] > 0
    for q in ("supported_by_context", "contradicts_history",
              "presupposition_or_drift", "would_allow_generation"):
        assert q in report["agreement"]
        assert "krippendorff_alpha" in report["agreement"][q]
    assert "human_vs_model" in report
    assert "calibration_by_band" in report
    assert "agreement_by_stratum" in report
    assert "action_agreement" in report
    assert "false_block_rate" in report["action_agreement"]
    assert "SYNTHETIC" in report["warning"].upper()


def test_analysis_flags_the_rubric_reachability_problem(packet_and_key):
    """The support-based rubric cannot reach reasoning errors.

    The dry run exists to surface this, so the diagnostic must be present.
    """
    packet, key = packet_and_key
    ann = simulate_annotations(packet, key, n_annotators=20, n_annotations=400, seed=13)
    report = analyze_annotations(ann, key)
    reach = report.get("rubric_reachability")
    assert reach is not None
    assert "reasoning_errors" in reach
    assert "note" in reach


def test_analysis_handles_an_empty_annotation_set(packet_and_key):
    _, key = packet_and_key
    report = analyze_annotations([], key)
    assert report["n_annotations"] == 0
    assert report["n_items"] == 0


# ----------------------------------------------------------------- script
def test_simulation_script_writes_labelled_artifacts(tmp_path):
    packet = tmp_path / "packet.jsonl"
    key = tmp_path / "key.jsonl"
    out = tmp_path / "ann.jsonl"
    report = tmp_path / "report.json"
    prep = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "prepare_human_validation.py"),
         "--out", str(packet), "--key-out", str(key),
         "--n", "20", "--duplicates", "3", "--pool-limit", "80"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900,
    )
    assert prep.returncode == 0, prep.stderr
    proc = subprocess.run(
        [sys.executable, str(SIM_SCRIPT), "--packet", str(packet), "--key", str(key),
         "--out", str(out), "--report", str(report),
         "--annotators", "10", "--n-annotations", "60", "--seed", "42"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr
    assert "simulated" in proc.stdout.lower()

    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert "SYNTHETIC" in lines[0]["warning"].upper()
    assert lines[0]["synthetic"] is True
    assert all(a.get("synthetic") is True for a in lines[1:])

    payload = json.loads(report.read_text())
    assert payload["synthetic"] is True
    assert "SYNTHETIC" in payload["warning"].upper()
    assert payload["n_annotations"] == 60
