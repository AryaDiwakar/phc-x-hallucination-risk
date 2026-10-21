"""The annotation session: the part of the human-validation study a person
actually touches.

The study was blocked on one thing: there was no way for an annotator to
do the work.  This module is that missing piece.  It owns the item order,
the progress file and the annotation record; the CLI in
``phcx.cli.main`` owns the terminal.

Two design points matter for the study's validity:

* **The intra-rater duplicates are not marked.**  Forty items appear twice
  in the packet.  The session presents them like any other item, so a
  duplicate is only identifiable afterwards by its content -- which is what
  makes the intra-rater consistency measurement meaningful.
* **Annotator codes are opaque.**  A code is generated locally and never
  linked to a name, so the finished file can be shared without identifying
  anybody.
"""
from __future__ import annotations

import json
import random
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .experiments.human_validation import ACTIONS, RUBRIC_QUESTIONS, YES_NO_UNCERTAIN

#: the three yes/no/uncertain questions, in the order they are asked
YES_NO_QUESTIONS = (
    "supported_by_context",
    "contradicts_history",
    "presupposition_or_drift",
)


def new_annotator_code() -> str:
    """An opaque, non-identifying code for an annotator."""
    return f"A{secrets.randbelow(9000) + 1000}"


@dataclass
class AnnotationSession:
    """A resumable annotation session over a packet."""

    packet: List[Dict[str, Any]]
    out_path: Path
    code: str
    seed: int = 42
    shuffle: bool = True
    records: List[Dict[str, Any]] = field(default_factory=list)
    _order: List[int] = field(default_factory=list)
    _done: set = field(default_factory=set)

    @classmethod
    def from_packet(
        cls,
        packet_path: str | Path,
        out_path: str | Path,
        code: Optional[str] = None,
        seed: int = 42,
        shuffle: bool = True,
    ) -> "AnnotationSession":
        packet: List[Dict[str, Any]] = []
        with Path(packet_path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    packet.append(json.loads(line))
        session = cls(
            packet=packet,
            out_path=Path(out_path),
            code=code or new_annotator_code(),
            seed=seed,
            shuffle=shuffle,
        )
        session._prepare()
        return session

    # ------------------------------------------------------------------ order
    def _prepare(self) -> None:
        rng = random.Random(self.seed)
        order = list(range(len(self.packet)))
        if self.shuffle:
            rng.shuffle(order)
        self._order = order
        # resume: skip what has already been recorded for this code
        done = set()
        if self.out_path.exists():
            for rec in self._read_records():
                if rec.get("annotator_code") == self.code:
                    done.add(rec.get("annotation_id"))
        self._done = done

    def _read_records(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not self.out_path.exists():
            return rows
        with self.out_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    # ------------------------------------------------------------------ items
    def remaining(self) -> List[Dict[str, Any]]:
        return [self.packet[i] for i in self._order
                if self.packet[i].get("annotation_id") not in self._done]

    @property
    def n_remaining(self) -> int:
        return len(self.remaining())

    @property
    def n_total(self) -> int:
        return len(self.packet)

    def take(self) -> Optional[Dict[str, Any]]:
        """The next unannotated item, or None when the packet is exhausted."""
        for i in self._order:
            item = self.packet[i]
            if item.get("annotation_id") not in self._done:
                return item
        return None

    # --------------------------------------------------------------- recording
    def record(
        self,
        item: Dict[str, Any],
        answers: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate and append one annotation."""
        problems = validate_answers(answers)
        if problems:
            raise ValueError("; ".join(problems))
        rec = {
            "annotation_id": item.get("annotation_id"),
            "annotator_code": self.code,
            "round": 0,
            **{q: answers[q] for q in RUBRIC_QUESTIONS},
            "rationale": answers.get("rationale", ""),
            "human": True,
        }
        self.records.append(rec)
        self._done.add(item.get("annotation_id"))
        self._append(rec)
        return rec

    def _append(self, rec: Dict[str, Any]) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def close(self) -> Dict[str, Any]:
        return {
            "annotator_code": self.code,
            "n_recorded": len(self.records),
            "n_total": self.n_total,
            "n_remaining": self.n_remaining,
            "out_path": str(self.out_path),
        }


def validate_answers(answers: Dict[str, Any]) -> List[str]:
    """Check that an answer set is complete and in range."""
    problems: List[str] = []
    for q in YES_NO_QUESTIONS:
        v = answers.get(q)
        if v not in YES_NO_UNCERTAIN:
            problems.append(f"{q} must be one of {YES_NO_UNCERTAIN}")
    action = answers.get("would_allow_generation")
    if action not in ACTIONS:
        problems.append(f"would_allow_generation must be one of {ACTIONS}")
    conf = answers.get("confidence")
    try:
        c = int(conf)
        if not 1 <= c <= 5:
            problems.append("confidence must be between 1 and 5")
    except (TypeError, ValueError):
        problems.append("confidence must be an integer between 1 and 5")
    return problems


def format_item(item: Dict[str, Any]) -> str:
    """Render an item for a human to read."""
    lines: List[str] = []
    lines.append("-" * 72)
    lines.append(f"item {item.get('annotation_id')}   [{item.get('source_dataset', '?')}]")
    lines.append("-" * 72)
    lines.append("PROMPT")
    lines.append(f"  {item.get('prompt', '')}")
    history = item.get("conversation_history") or []
    if history:
        lines.append("CONVERSATION SO FAR")
        for turn in history:
            role = str(turn.get("role", "?")).upper()
            text = str(turn.get("content", "")).replace("\n", " ")
            lines.append(f"  {role}: {text}")
    evidence = item.get("evidence") or []
    if evidence:
        lines.append("EVIDENCE")
        for doc in evidence:
            text = (doc.get("text", "") if isinstance(doc, dict) else str(doc))
            lines.append(f"  - {str(text).replace(chr(10), ' ')}")
    lines.append("PROPOSED ANSWER")
    lines.append(f"  {item.get('proposed_answer', '')}")
    lines.append("-" * 72)
    return "\n".join(lines)


RUBRIC_TEXT = (
    "1. Is the answer factually supported by the supplied context? [yes/no/uncertain]\n"
    "2. Does it contradict the conversation history? [yes/no/uncertain]\n"
    "3. Does it contain an unsupported presupposition or entity/time drift? [yes/no/uncertain]\n"
    "4. Would you allow generation without retrieval or clarification? "
    "[allow/retrieve/clarify/abstain]\n"
    "5. Confidence 1-5, then a short rationale (one line):"
)
