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


def is_factual_candidate(sentence: str) -> bool:
    s = sentence.strip()
    if not s or len(tokenize(s)) < 3:
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
    for sent in sent_split(turn.text):
        if len(claims) >= max_claims:
            break
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
    if not claims and turn.speaker == "assistant" and len(content_tokens(turn.text)) >= 4:
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
