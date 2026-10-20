"""Unsupported presupposition detector (spec section 9).

Detects propositions the user's question takes for granted and checks whether
the conversation or evidence actually establishes them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..memory import ConversationMemory
from ..utils.text import content_tokens, similarity

# (regex, presupposition template, whether the presupposed content is the clause)
_TRIGGERS: List[Tuple[str, str, str]] = [
    # change-of-state: presupposes the change happened
    (r"\bwhy\s+(?:did|do|does|has|have)\s+(.{2,60}?)\s+(improve|increase|rise|grow|improve[sd]?|"
     r"decrease|decline|drop|fall|recover|change|switch|replace)\b",
     "change", "X {verb}ed"),
    (r"\bhow\s+(?:did|do|does)\s+(.{2,60}?)\s+(improve|increase|rise|grow|decrease|decline|drop|fall)\b",
     "change", "X {verb}ed"),
    (r"\b(?:improve|improved|increase|increased|decrease|decreased|decline|declined|"
     r"rise|rose|fall|fell|grow|grew|drop|dropped|change|changed)\s+from\s+([\w\d\.,%]+)"
     r"\s+to\s+([\w\d\.,%]+)",
     "change_with_values", "metric changed from {0} to {1}"),
    (r"\bfrom\s+([\w\d\.,%]+)\s+to\s+([\w\d\.,%]+)\b",
     "change_with_values", "metric changed from {0} to {1}"),
    # factive verbs: presuppose the complement is true
    (r"\b(?:know|realize|discover|notice|regret|acknowledge|understand|confirm)\s+(?:that\s+)?(.{5,120})",
     "factive", "that {0}"),
    # question-form factives: "isn't it true that X", "is it that X", ...
    # These presuppose the complement even though they are phrased as
    # questions, and are the canonical false-presupposition form.
    (r"\b(?:is|was|are|were)(?:n['’]t|\s+not)\s+it\s+(?:true\s+)?(?:that\s+)?(.{5,120})",
     "factive", "that {0}"),
    (r"\b(?:is|was|are|were)\s+it\s+(?:true\s+)?(?:that\s+)?(.{5,120})",
     "factive", "that {0}"),
    # "don't you think that X" also presupposes the complement
    (r"\bdon['’]t\s+you\s+(?:think|believe|agree|reckon)\s+(?:that\s+)?(.{5,120})",
     "factive", "that {0}"),
    # iterative: presupposes it happened before
    (r"\b(?:again|still|continue[sd]?|resume[sd]?|another|repeat(?:ed)?)\b",
     "iterative", "event occurred previously"),
    # cleft / presuppositional definite descriptions
    (r"\bthe\s+(?:best|worst|top|highest|lowest|fastest|largest|only|first|last)\s+"
     r"([\w\s]{2,40}?)\s+(?:is|was|were|are)\b",
     "superlative", "there exists a unique {0}"),
    # "back": presupposes prior state
    (r"\b(?:bring|get|put|take|going)\s+.*\bback\b",
     "reversion", "state existed before"),
    # comparative presupposition
    (r"\bbetter than\b|\bworse than\b|\bmore \w+ than\b|\bless \w+ than\b",
     "comparative", "comparison class exists"),
]

# presupposition cues that are safe when grounded
_SAFE_IF_SUPPORTED = ("know", "realize", "discover", "notice", "understand", "confirm")


@dataclass
class Presupposition:
    text: str
    trigger: str
    kind: str
    supported: bool
    support_score: float
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "presupposed_claim": self.text,
            "trigger": self.trigger,
            "kind": self.kind,
            "supported": self.supported,
            "support_score": round(self.support_score, 3),
            "evidence": self.evidence,
        }


@dataclass
class PresuppositionResult:
    score: float = 0.0
    count: int = 0
    unsupported_count: int = 0
    presuppositions: List[Presupposition] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "count": self.count,
            "unsupported_count": self.unsupported_count,
            "presuppositions": [p.to_dict() for p in self.presuppositions],
            "reasons": self.reasons,
        }


def _nums(text: str) -> set:
    """Normalised numbers: '70.0%' and '70.0' are the same value."""
    return {n.rstrip("%").rstrip(",") for n in re.findall(r"\d[\d,\.]*(?:%|percent)?", text)}


def _support(presupposed: str, context_texts: List[str]) -> Tuple[float, str]:
    best, best_t = 0.0, ""
    pcontent = content_tokens(presupposed)
    if not pcontent:
        return 0.0, ""
    for t in context_texts:
        s = similarity(presupposed, t)
        # numeric agreement strengthens support substantially
        pn = _nums(presupposed)
        tn = _nums(t)
        if pn and tn:
            if pn <= tn:
                s = min(1.0, s + 0.35)      # every presupposed number is attested
            elif pn & tn:
                s = s * 0.7                 # partial overlap -> some values unsupported
            else:
                s = min(1.0, s * 0.5)       # no numeric overlap -> weak
        if s > best:
            best, best_t = s, t
    return best, best_t


class PresuppositionEngine:
    def __init__(self, support_threshold: float = 0.45, max_props: int = 6):
        self.support_threshold = support_threshold
        self.max_props = max_props

    def extract(self, prompt: str) -> List[Presupposition]:
        found: List[Presupposition] = []
        seen = set()
        for pattern, kind, template in _TRIGGERS:
            if len(found) >= self.max_props:
                break
            for m in re.finditer(pattern, prompt, re.I | re.S):
                groups = m.groups()
                if kind == "change" and len(groups) >= 2:
                    subj, verb = groups[0].strip(), groups[1].strip()
                    text = f"{subj} {verb} (change occurred)"
                    trigger = m.group(0)
                elif kind == "change_with_values" and len(groups) >= 2:
                    text = f"metric changed from {groups[0]} to {groups[1]}"
                    trigger = m.group(0)
                elif kind == "factive" and groups:
                    text = f"that {groups[0].strip()}"
                    trigger = m.group(0)
                elif groups:
                    text = template.format(*groups)
                    trigger = m.group(0)
                else:
                    text = template
                    trigger = m.group(0)
                key = (kind, text.lower()[:80])
                if key in seen:
                    continue
                seen.add(key)
                found.append(Presupposition(text=text, trigger=trigger[:120], kind=kind,
                                            supported=False, support_score=0.0))
        return found

    def analyze(
        self,
        prompt: str,
        memory: Optional[ConversationMemory] = None,
        evidence_texts: Optional[List[str]] = None,
    ) -> PresuppositionResult:
        res = PresuppositionResult()
        props = self.extract(prompt or "")
        context: List[str] = []
        if memory is not None:
            context.extend(c.text for c in memory.all_claims)
            context.extend(t.text for t in memory.turns)
        if evidence_texts:
            context.extend(evidence_texts)

        for p in props:
            score, ev = _support(p.text, context)
            p.support_score = score
            p.supported = score >= self.support_threshold
            p.evidence = ev[:200]
            if not p.supported:
                res.unsupported_count += 1
                res.reasons.append(
                    f"SUPPORT: NOT FOUND for presupposed claim \"{p.text[:90]}\" (support={score:.2f})"
                )
        # a supported "change from A to B" entailment also grounds the bare
        # "the change happened" presupposition produced by the same trigger
        value_backed = [p for p in props
                        if p.kind == "change_with_values" and p.supported]
        if value_backed:
            for p in props:
                if p.kind == "change" and not p.supported:
                    p.supported = True
                    p.support_score = max(p.support_score,
                                          max(x.support_score for x in value_backed))
                    p.evidence = max(value_backed, key=lambda x: x.support_score).evidence

        res.presuppositions = props
        res.count = len(props)
        res.unsupported_count = sum(1 for p in props if not p.supported)
        if props:
            # risk grows with the fraction and severity of unsupported presuppositions
            frac = res.unsupported_count / max(1, len(props))
            res.score = float(min(1.0, 0.35 * min(3, res.unsupported_count) / 1.5 * (0.5 + 0.5 * frac) + 0.25 * frac))
        return res
