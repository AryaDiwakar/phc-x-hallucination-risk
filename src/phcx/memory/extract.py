"""Claim extraction from raw utterances.

Rule-based and deterministic: PHC-X never treats an assistant utterance as a
set of automatically verified facts.  Extraction only proposes candidate
claims; provenance classification assigns their epistemic status afterwards.
"""
from __future__ import annotations

import re
from typing import List

from ..utils.text import content_tokens, sent_split, tokenize
from .types import Claim, Turn

_QUESTION_RE = re.compile(r"\?\s*$")
_FACTUAL_MARKERS = re.compile(
    r"\b(is|are|was|were|has|have|had|contains?|consists?|uses?|used|produces?|"
    r"achieves?|reports?|equals?|=|located|founded|released|amounted?|totaled?|"
    r"increased?|decreased?|improved?|declined?|according)\b",
    re.I,
)
_NUMBER_RE = re.compile(r"\b\d[\d,\.]*%?\b")
_STRUCTURED_FACT_RE = re.compile(r"^\s*(?:[-*]\s*)?([A-Za-z][\w -]{1,40}):\s*(\S.*)$")
_META_REPLY_RE = re.compile(
    r"^\s*(?:understood|got it|noted|okay|ok)\b|"
    r"^\s*(?:i|we)\s+(?:(?:will|shall|can|cannot|can't|won't|'ll|’ll)\s+)?"
    r"(?:use|treat|distinguish|keep|avoid|remember|help|answer|cite)\b",
    re.I,
)


def is_factual_candidate(sentence: str) -> bool:
    s = sentence.strip()
    # Section labels such as "Northstar details are:" are not claims.  Treating
    # them as model facts causes any later reference to the topic to look like
    # contamination even when the actual value came from the user.
    if not s or s.endswith(":") or len(tokenize(s)) < 3 or _META_REPLY_RE.search(s):
        return False
    if _QUESTION_RE.search(s):
        return False
    if _FACTUAL_MARKERS.search(s) or _NUMBER_RE.search(s):
        return True
    # Declarative sentences with a proper noun are usually claims.
    if re.match(r"^(?:[A-Z][\w\-']*)(?:\s+\w+){1,}", s) and not s.startswith(("Yes", "No", "I ", "You ")):
        return True
    return False


def extract_claims(turn: Turn, max_claims: int = 8) -> List[Claim]:
    """Extract candidate factual claims from a turn."""
    claims: List[Claim] = []
    structured_lines = set()
    # Chat UIs and assistants often render factual answers as label/value
    # lines ("Participants: 120"). Keep the label attached to the value so
    # provenance matching doesn't collapse into a context-free numeric claim.
    for line in turn.text.splitlines():
        match = _STRUCTURED_FACT_RE.match(line)
        if not match or len(claims) >= max_claims:
            continue
        structured_lines.add(line.strip())
        label, value = match.groups()
        claims.append(Claim.create(
            text=f"{label.strip()}: {value.strip()}",
            turn_id=turn.turn_id,
            speaker=turn.speaker,
            timestamp=turn.timestamp,
        ))
    for sent in sent_split(turn.text):
        if len(claims) >= max_claims:
            break
        if sent.strip() in structured_lines:
            continue
        if not is_factual_candidate(sent):
            continue
        claims.append(
            Claim.create(
                text=sent,
                turn_id=turn.turn_id,
                speaker=turn.speaker,
                timestamp=turn.timestamp,
            )
        )
    # Always keep at least one claim per non-trivial assistant turn so that
    # provenance tracking never loses model statements.
    if (
        not claims and turn.speaker == "assistant"
        and len(content_tokens(turn.text)) >= 4
        and not _META_REPLY_RE.search(turn.text)
    ):
        claims.append(
            Claim.create(
                text=turn.text.strip(),
                turn_id=turn.turn_id,
                speaker=turn.speaker,
                timestamp=turn.timestamp,
                metadata={"fallback": True},
            )
        )
    for c in claims:
        c.entities = _naive_entities(c.text)
    return claims


def _naive_entities(text: str) -> List[str]:
    ents = []
    for m in re.finditer(r"\b([A-Z][a-z0-9]+(?:[\s\-][A-Z][a-z0-9]+)*)\b", text):
        e = m.group(1)
        if e.lower() in {"the", "a", "an", "i", "we", "it", "in", "on", "at", "to"}:
            continue
        if len(e) > 1:
            ents.append(e)
    # dedupe preserving order
    seen, out = set(), []
    for e in ents:
        k = e.lower()
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out
