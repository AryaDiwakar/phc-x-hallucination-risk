"""Entity tracking and entity-drift detection (spec section 7).

Detects entity replacement, alias/type mismatches, ambiguous references and
disambiguation failures across conversation turns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..memory import ConversationMemory, Turn
from ..utils.text import content_tokens, similarity

_ENT_RE = re.compile(
    r"\b("
    r"[A-Z][A-Za-z0-9]+(?:[\s\-][A-Z][A-Za-z0-9]*)*"     # Capitalized / CamelCase / mixed case
    r"|\d[\d,]*\.?\d*\s*(?:%|percent|samples?|rows?|points?)"  # numeric quantities
    r")\b"
)
_GENERIC = {
    "the", "a", "an", "i", "we", "it", "they", "you", "he", "she", "this", "that",
    "our", "my", "your", "yes", "no", "ok", "great", "sure", "thanks", "please",
    "what", "how", "why", "when", "where", "who", "which", "based", "using", "used",
    "and", "or", "but", "let", "us", "proceed", "great", "thanks", "noted", "accuracy",
}

_REPLACE_RE = re.compile(
    r"\b(?:switch(?:ed)?|chang(?:ed|es|ing)?|migrat(?:ed|es|ing)?|replac(?:ed|es|ing)?|"
    r"updat(?:ed|es|ing)?|mov(?:ed|es|ing)|shift(?:ed|s)?|swapp(?:ed|s)?)\b"
    r"[^.!?]{0,60}?\bto\s+((?:[A-Z][\w\-]*)(?:\s+[A-Z][\w\-]*)*)",
    re.I,
)

# Semantic type hints
_TYPE_HINTS = {
    "dataset": ("dataset", "corpus", "data set", "split", "train set", "test set"),
    "model": ("model", "bert", "roberta", "gpt", "llama", "classifier", "network", "transformer"),
    "metric": ("accuracy", "f1", "precision", "recall", "auc", "score", "%", "percent"),
    "person": ("professor", "dr", "researcher", "author"),
    "org": ("inc", "ltd", "university", "lab", "company", "google", "openai"),
    "date": ("2020", "2021", "2022", "2023", "2024", "2025", "2026"),
}


@dataclass
class EntityMention:
    entity: str
    turn_id: int
    speaker: str
    context: str
    etype: str = "unknown"


@dataclass
class EntityDriftResult:
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    entities: Dict[str, List[int]] = field(default_factory=dict)
    ambiguous_references: List[Dict[str, Any]] = field(default_factory=list)
    replacements: List[Dict[str, Any]] = field(default_factory=list)
    type_mismatches: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "reasons": self.reasons,
            "entities": {k: v[:10] for k, v in self.entities.items()},
            "ambiguous_references": self.ambiguous_references[:6],
            "replacements": self.replacements[:6],
            "type_mismatches": self.type_mismatches[:6],
        }


def _classify_type(entity: str, context: str) -> str:
    low = (entity + " " + context).lower()
    for t, hints in _TYPE_HINTS.items():
        if any(h in low for h in hints):
            return t
    if re.match(r"^\d", entity):
        return "quantity"
    if entity[:1].isupper() and " " not in entity:
        return "proper"
    return "unknown"


class EntityTracker:
    def extract(self, text: str, turn_id: int, speaker: str) -> List[EntityMention]:
        out: List[EntityMention] = []
        seen = set()
        for m in _ENT_RE.finditer(text or ""):
            e = m.group(1).strip()
            if e.lower() in _GENERIC or len(e) < 2:
                continue
            key = e.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(EntityMention(entity=e, turn_id=turn_id, speaker=speaker, context=text,
                                     etype=_classify_type(e, text)))
        return out

    def track(self, memory: ConversationMemory, prompt: Optional[str] = None) -> EntityDriftResult:
        res = EntityDriftResult()
        mentions: List[EntityMention] = []
        for t in memory.turns:
            mentions.extend(self.extract(t.text, t.turn_id, t.speaker))
        if prompt:
            mentions.extend(self.extract(prompt, -1, "user"))

        # group by canonical alias (case-insensitive, prefix-compatible)
        groups: Dict[str, List[EntityMention]] = {}
        canon: Dict[str, str] = {}
        for m in mentions:
            key = m.entity.lower()
            host = None
            for k in canon:
                if key.startswith(k) or k.startswith(key):
                    host = k
                    break
            if host is None:
                canon[key] = key
                host = key
            groups.setdefault(host, []).append(m)

        for g, ms in groups.items():
            turns = sorted({m.turn_id for m in ms})
            res.entities[g] = turns

        # 2) entity replacement: same role slot ("dataset", "model") with
        #    different entities introduced at different turns
        by_type: Dict[str, Dict[str, List[int]]] = {}
        for g, ms in groups.items():
            t = _classify_type(g, " ".join(m.context for m in ms))
            by_type.setdefault(t, {}).setdefault(g, [])
            by_type[t][g].extend(m.turn_id for m in ms)
        # explicit replacement language ("switched to X", "migrated to Y"):
        # the new entity adopts the role of the active typed slot
        for t in memory.turns:
            for m in _REPLACE_RE.finditer(t.text or ""):
                ent = (m.group(1) or "").strip()
                if not ent or len(ent) < 2:
                    continue
                slot = next(
                    (tt for tt in ("dataset", "model", "metric") if by_type.get(tt)), None
                )
                if slot is None:
                    continue
                key = ent.lower()
                groups.setdefault(key, []).append(
                    EntityMention(entity=ent, turn_id=t.turn_id, speaker=t.speaker,
                                  context=t.text, etype=slot)
                )
                by_type[slot].setdefault(key, []).append(t.turn_id)

        for t, ents in by_type.items():
            if t in ("dataset", "model", "metric") and len(ents) >= 2:
                names = list(ents)
                res.replacements.append(
                    {"type": t, "entities": [{"entity": n, "turns": sorted(set(ents[n]))} for n in names]}
                )
                res.reasons.append(
                    f"multiple {t} entities referenced across turns: {', '.join(names[:4])}"
                )

        # 3) type mismatch: same surface form used as different types
        for g, ms in groups.items():
            types = {m.etype for m in ms}
            if len(types) > 1 and "unknown" not in types:
                res.type_mismatches.append({"entity": g, "types": sorted(types)})

        # 4) definite reference to a role with >1 candidate in prompt
        role_ambiguity = 0.0
        replacement_score = min(1.0, 0.45 * len(res.replacements)) if res.replacements else 0.0
        mismatch_score = min(1.0, 0.3 * len(res.type_mismatches)) if res.type_mismatches else 0.0
        res.score = float(min(1.0, max(role_ambiguity, replacement_score, mismatch_score)))
        return res
