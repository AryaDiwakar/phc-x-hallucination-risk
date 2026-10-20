"""Entity tracking and entity-drift detection (spec section 7).

Detects entity replacement, alias/type mismatches, ambiguous references and
disambiguation failures across conversation turns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..memory import ConversationMemory, Turn
from ..utils.text import content_tokens, sent_split, similarity

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

# The verb is matched case-insensitively, but the *entity* is matched
# case-sensitively: a case-insensitive capture swallowed trailing lowercase
# words ("Corpus B now") and invented entities out of them.
_REPLACE_RE = re.compile(
    r"\b(?i:switch(?:ed)?|chang(?:ed|es|ing)?|migrat(?:ed|es|ing)?|replac(?:ed|es|ing)?|"
    r"updat(?:ed|es|ing)?|mov(?:ed|es|ing)|shift(?:ed|s)?|swapp(?:ed|s)?)\b"
    r"[^.!?]{0,60}?\bto\s+((?:[A-Z][\w\-]*)(?:\s+[A-Z][\w\-]*)*)"
)

# Definite role references: "the dataset", "the model", ...
_ROLE_REF_RE = re.compile(
    r"\bthe\s+(dataset|corpus|data\s+set|model|metric|network|classifier)\b", re.I
)
_ROLE_SLOT = {
    "dataset": "dataset", "corpus": "dataset", "data set": "dataset",
    "model": "model", "network": "model", "classifier": "model",
    "metric": "metric",
}

# Semantic type hints
_TYPE_HINTS = {
    "dataset": ("dataset", "corpus", "data set", "split", "train set", "test set"),
    "model": ("model", "bert", "roberta", "gpt", "llama", "classifier", "network", "transformer"),
    "metric": ("accuracy", "f1", "precision", "recall", "auc", "score", "%", "percent"),
    "person": ("professor", "dr", "researcher", "author"),
    "org": ("inc", "ltd", "university", "lab", "company", "google", "openai"),
    "date": ("2020", "2021", "2022", "2023", "2024", "2025", "2026"),
}

# Name-driven typing.  The *name* of an entity is what identifies its role,
# not the sentence it happens to sit in: a turn that merely mentions "corpus"
# must not turn every capitalized word in it into a dataset.  The vocabulary
# covers the entity names PHC-X itself produces plus the generic role words.
_NAME_TYPE_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("dataset", re.compile(
        r"\b(corpus|corpusx|atlas|bench|archive|pool|dataset|data set|train set|test set)\b",
        re.I)),
    ("model", re.compile(
        r"\b(bert|roberta|distilbert|xlnet|electra|deberta|albert|longformer|bigbird|"
        r"convbert|model|classifier|network|transformer|llm|gpt)\b", re.I)),
    ("metric", re.compile(
        r"\b(accuracy|f1|precision|recall|auc|auroc|bleu|metric|score)\b", re.I)),
    ("person", re.compile(r"\b(professor|dr|researcher|author)\b", re.I)),
    ("org", re.compile(
        r"\b(inc|ltd|university|lab|company|google|openai)\b", re.I)),
]
_DATE_RE = re.compile(r"^(19|20)\d{2}$")


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


def _is_word_suffix(text: str, suffix: str) -> bool:
    """True when ``suffix`` is a whole-word suffix of ``text``."""
    if not suffix or not text:
        return False
    return bool(re.search(rf"(?:^|\s){re.escape(suffix)}$", text))


def _sentence_around(context: str, entity: str) -> str:
    """The sentence of ``context`` that mentions ``entity`` (fallback: context)."""
    ent = (entity or "").lower()
    if not ent:
        return context or ""
    for s in sent_split(context or ""):
        if ent in s.lower():
            return s
    return context or ""


def _name_type(entity: str) -> str:
    """Type of an entity derived from its *name* only.

    This is the type used for role slots (dataset / model / metric): it is
    stable regardless of the sentence the entity appears in, so a common
    capitalized word such as "Okay" is never mistaken for a dataset.
    """
    e = (entity or "").strip()
    if not e:
        return "unknown"
    for t, pat in _NAME_TYPE_PATTERNS:
        if pat.search(e):
            return t
    if _DATE_RE.match(e):
        return "date"
    if re.match(r"^\d", e):
        return "quantity"
    if e[:1].isupper():
        return "proper"
    return "unknown"


def _context_type(entity: str, context: str) -> str:
    """Type inferred from the surrounding text (used for mismatch detection)."""
    local = _sentence_around(context, entity).lower()
    for t, hints in _TYPE_HINTS.items():
        if any(h in local for h in hints):
            return t
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
                                     etype=_name_type(e)))
        return out

    def track(self, memory: ConversationMemory, prompt: Optional[str] = None) -> EntityDriftResult:
        res = EntityDriftResult()
        mentions: List[EntityMention] = []
        for t in memory.turns:
            mentions.extend(self.extract(t.text, t.turn_id, t.speaker))
        if prompt:
            mentions.extend(self.extract(prompt, -1, "user"))

        # group by canonical alias (case-insensitive, prefix- and suffix-compatible)
        groups: Dict[str, List[EntityMention]] = {}
        canon: Dict[str, str] = {}
        raw_keys: List[str] = []
        for m in mentions:
            key = m.entity.lower()
            if key not in raw_keys:
                raw_keys.append(key)
            host = None
            for k in canon:
                if key.startswith(k) or k.startswith(key):
                    host = k
                    break
            if host is None:
                canon[key] = key
                host = key
            groups.setdefault(host, []).append(m)

        # A mention may carry a leading qualifier that the entity itself does
        # not have ("Summarize Corpus B" vs "Corpus B").  Merge such keys onto
        # the shorter word-boundary suffix so they do not look like two
        # different datasets.
        suffix_of: Dict[str, str] = {}
        for key in raw_keys:
            best = key
            for other in raw_keys:
                if other == key:
                    continue
                if len(other.split()) < len(best.split()) and _is_word_suffix(best, other):
                    best = other
            suffix_of[key] = best
        merged: Dict[str, List[EntityMention]] = {}
        for key, ms in groups.items():
            merged.setdefault(suffix_of.get(key, key), []).extend(ms)
        groups = merged

        for g, ms in groups.items():
            turns = sorted({m.turn_id for m in ms})
            res.entities[g] = turns

        # 2) entity replacement: same role slot ("dataset", "model") with
        #    different entities introduced at different turns
        by_type: Dict[str, Dict[str, List[int]]] = {}
        for g, ms in groups.items():
            t = _name_type(g)
            by_type.setdefault(t, {}).setdefault(g, [])
            by_type[t][g].extend(m.turn_id for m in ms)
        # explicit replacement language ("switched to X", "migrated to Y"):
        # the new entity adopts the role of the active typed slot
        for t in memory.turns:
            for m in _REPLACE_RE.finditer(t.text or ""):
                ent = (m.group(1) or "").strip()
                if not ent or len(ent) < 2:
                    continue
                # a name that identifies its own role wins; otherwise the new
                # entity inherits the slot that is currently active
                ent_type = _name_type(ent)
                if ent_type in ("dataset", "model", "metric"):
                    slot = ent_type
                else:
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

        # 3) type mismatch: the same surface form is used in a role that
        #    contradicts its name (e.g. a dataset named like a metric)
        for g, ms in groups.items():
            name_t = _name_type(g)
            ctx_types = {
                t for t in (_context_type(g, m.context) for m in ms) if t != "unknown"
            }
            if name_t != "unknown" and ctx_types and name_t not in ctx_types:
                res.type_mismatches.append(
                    {"entity": g, "types": sorted({name_t} | ctx_types)}
                )

        # 4) definite reference to a role with >1 candidate in prompt
        role_ambiguity = 0.0
        if prompt:
            for m in _ROLE_REF_RE.finditer(prompt or ""):
                slot = _ROLE_SLOT.get(re.sub(r"\s+", " ", m.group(1).strip().lower()))
                if not slot:
                    continue
                cands = sorted(by_type.get(slot, {}))
                if len(cands) < 2:
                    continue
                role_ambiguity = max(role_ambiguity, min(1.0, 0.35 * (len(cands) - 1)))
                res.ambiguous_references.append(
                    {
                        "reference": m.group(0),
                        "role": slot,
                        "candidates": cands[:4],
                        "n_candidates": len(cands),
                    }
                )
                res.reasons.append(
                    f"ambiguous role reference {m.group(0)!r}: "
                    f"{len(cands)} {slot} candidates ({', '.join(cands[:4])})"
                )

        replacement_score = min(1.0, 0.45 * len(res.replacements)) if res.replacements else 0.0
        mismatch_score = min(1.0, 0.3 * len(res.type_mismatches)) if res.type_mismatches else 0.0
        res.score = float(min(1.0, max(role_ambiguity, replacement_score, mismatch_score)))
        return res
