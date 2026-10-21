"""Tests for the annotation session -- the tool a real person uses."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from phcx.annotate import (
    AnnotationSession,
    format_item,
    new_annotator_code,
    validate_answers,
)

ROOT = Path(__file__).resolve().parents[1]

VALID = {
    "supported_by_context": "no",
    "contradicts_history": "no",
    "presupposition_or_drift": "yes",
    "would_allow_generation": "retrieve",
    "confidence": 4,
    "rationale": "The answer names a dataset that appears nowhere.",
}


@pytest.fixture()
def packet(tmp_path):
    p = tmp_path / "packet.jsonl"
    rows = [
        {
            "annotation_id": f"hv_{i:04d}",
            "source_dataset": "PHC-Conversation",
            "prompt": "Which dataset does the Aurora-12A project use?",
            "conversation_history": [
                {"role": "user", "content": "We use Atlas A."},
                {"role": "assistant", "content": "Noted, Atlas A."},
            ],
            "evidence": [{"text": "The Aurora-12A project evaluates on Atlas A."}],
            "proposed_answer": "We use Corpus M for evaluation.",
            "annotator_label": None,
        }
        for i in range(5)
    ]
    with p.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def test_annotator_codes_are_opaque():
    codes = {new_annotator_code() for _ in range(20)}
    assert len(codes) > 1
    for c in codes:
        assert c.startswith("A") and c[1:].isdigit()


def test_validate_answers_rejects_incomplete_or_invalid():
    assert validate_answers(VALID) == []
    bad = dict(VALID, supported_by_context="maybe")
    assert validate_answers(bad)
    bad2 = dict(VALID, confidence=9)
    assert validate_answers(bad2)
    bad3 = dict(VALID, would_allow_generation="skip")
    assert validate_answers(bad3)
    missing = {k: v for k, v in VALID.items() if k != "confidence"}
    assert validate_answers(missing)


def test_session_records_and_appends(tmp_path, packet):
    out = tmp_path / "annotations.jsonl"
    s = AnnotationSession.from_packet(packet, out, code="A1111", shuffle=False)
    assert s.n_total == 5
    assert s.n_remaining == 5
    item = s.take()
    rec = s.record(item, VALID)
    assert rec["annotator_code"] == "A1111"
    assert rec["human"] is True
    assert rec["confidence"] == 4
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["annotation_id"] == item["annotation_id"]


def test_session_resumes(tmp_path, packet):
    out = tmp_path / "annotations.jsonl"
    s = AnnotationSession.from_packet(packet, out, code="A1111", shuffle=False)
    s.record(s.take(), VALID)
    s2 = AnnotationSession.from_packet(packet, out, code="A1111", shuffle=False)
    assert s2.n_remaining == 4
    # a different annotator does not inherit the progress
    s3 = AnnotationSession.from_packet(packet, out, code="A2222", shuffle=False)
    assert s3.n_remaining == 5


def test_session_refuses_to_record_invalid_answers(tmp_path, packet):
    out = tmp_path / "annotations.jsonl"
    s = AnnotationSession.from_packet(packet, out, code="A1111", shuffle=False)
    item = s.take()
    with pytest.raises(ValueError):
        s.record(item, dict(VALID, confidence="high"))
    assert not out.exists() or out.read_text().strip() == ""


def test_format_item_shows_every_part(tmp_path, packet):
    s = AnnotationSession.from_packet(packet, tmp_path / "a.jsonl", code="A1111")
    text = format_item(s.take())
    for marker in ("PROMPT", "CONVERSATION SO FAR", "EVIDENCE", "PROPOSED ANSWER"):
        assert marker in text
    assert "Atlas A" in text
    assert "Corpus M" in text


def test_annotate_cli_dry_run(packet, tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "phcx.cli.main", "annotate",
         "--packet", str(packet), "--out", str(tmp_path / "a.jsonl"), "--dry-run"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PROPOSED ANSWER" in proc.stdout
    # the rubric is shown in human-readable form
    assert "Is the answer factually supported" in proc.stdout
    assert "allow/retrieve/clarify/abstain" in proc.stdout


def test_analyze_annotations_cli_accepts_real_annotations(tmp_path, packet):
    out = tmp_path / "annotations.jsonl"
    s = AnnotationSession.from_packet(packet, out, code="A1111", shuffle=False)
    for _ in range(3):
        s.record(s.take(), VALID)
    report = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, "-m", "phcx.cli.main", "analyze-annotations",
         "--annotations", str(out), "--out", str(report)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(report.read_text())
    assert payload["n_annotations"] == 3
    assert payload["n_annotators"] == 1
    assert "agreement" in payload
