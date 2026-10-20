"""Tests for the human-validation packet builder.

The packet has to be *blind* (no label, score or stratum reaches the
annotator) and *stratified* (the protocol asks for equal coverage of the
risk categories, the domains, the action groups and the score bands).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_human_validation.py"

#: fields that must never reach the annotator
LEAKY_FIELDS = {
    "label", "gold_label", "risk", "risk_score", "risk_level", "level",
    "action", "action_group", "score_band", "category", "domain",
    "stratum_dimension", "category_or_domain", "duplicate_of",
}


def _run(tmp_path, *extra):
    out = tmp_path / "packet.jsonl"
    key = tmp_path / "key.jsonl"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), "--key-out", str(key),
         "--n", "24", "--duplicates", "4", "--pool-limit", "250", *extra],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, f"script failed:\n{proc.stdout}\n{proc.stderr}"
    return out, key


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    base = tmp_path_factory.mktemp("hv")
    return _run(base)


def test_packet_is_blinded(packet):
    out, _ = packet
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert rows
    for row in rows:
        assert not (set(row.keys()) & LEAKY_FIELDS), (
            f"leaking field reached the annotator: "
            f"{sorted(set(row.keys()) & LEAKY_FIELDS)}"
        )
        assert row["annotator_label"] is None
        assert row["rationale"] is None
        assert row["rubric"], "the packet must carry the rubric"
        assert row["prompt"] is not None


def test_packet_contains_the_requested_number_of_duplicates(packet):
    out, key = packet
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    keys = [json.loads(line) for line in key.read_text().splitlines() if line.strip()]
    header, entries = keys[0], keys[1:]
    dups = [k for k in entries if k.get("duplicate_of")]
    assert len(dups) == header["n_duplicates"] == 4
    # the duplicate must be a copy of an existing item
    by_id = {r["annotation_id"]: r for r in rows}
    for d in dups:
        original = by_id[d["duplicate_of"]]
        assert original["prompt"] == by_id[d["annotation_id"]]["prompt"]
        assert original["proposed_answer"] == by_id[d["annotation_id"]]["proposed_answer"]


def test_packet_is_stratified_across_the_protocol_dimensions(packet):
    out, key = packet
    keys = [json.loads(line) for line in key.read_text().splitlines() if line.strip()]
    entries = [k for k in keys[1:] if not k.get("duplicate_of")]
    assert entries
    for dim in ("action_group", "score_band", "source_dataset"):
        values = {k[dim] for k in entries}
        assert len(values) >= 2, f"{dim} is not stratified: {values}"
    categories = {k["category_or_domain"] for k in entries}
    assert len(categories) >= 2, f"strata are not covered: {categories}"


def test_key_file_carries_the_analysis_metadata(packet):
    _, key = packet
    keys = [json.loads(line) for line in key.read_text().splitlines() if line.strip()]
    header = keys[0]
    assert header["analysis_seed"] == 42
    assert header["rubric_version"]
    assert "never" in header["note"].lower()
