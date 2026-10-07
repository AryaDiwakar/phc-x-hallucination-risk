"""Temporal consistency engine (spec section 8).

Extracts temporal expressions, estimates staleness of relevant memory against
freshness demands of the prompt, and flags TEMPORAL_UNCERTAINTY when freshness
cannot be determined without external retrieval.  PHC-X does NOT claim that
temporal freshness is perfectly knowable offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..memory import ConversationMemory
from ..utils.text import similarity

_RELATIVE = {
    "today": 0, "now": 0, "currently": 0, "right now": 0, "at present": 0,
    "this year": 0, "latest": 0, "most recent": 0, "up-to-date": 0, "up to date": 0,
    "current": 0, "recently": 1, "this week": 7, "this month": 30,
    "yesterday": 1, "last week": 7, "last month": 30, "last year": 365,
    "previously": None, "earlier": None, "before": None, "at the time": None,
    "in the past": None, "originally": None, "initially": None, "then": None,
}
_FRESHNESS_CUES = (
    "current", "latest", "today", "now", "recent", "up-to-date", "up to date",
    "newest", "most recent", "currently", "this year", "version", "price",
    "status", "nowadays", "live", "today's",
)
_ABS_DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2})\b|\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b|"
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
    re.I,
)


@dataclass
class TemporalResult:
    score: float = 0.0
    freshness_demand: bool = False
    temporal_uncertainty: bool = False
    stale_memory: bool = False
    reasons: List[str] = field(default_factory=list)
    prompt_expressions: List[str] = field(default_factory=list)
    memory_expressions: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "freshness_demand": self.freshness_demand,
            "temporal_uncertainty": self.temporal_uncertainty,
            "stale_memory": self.stale_memory,
            "reasons": self.reasons,
            "prompt_expressions": self.prompt_expressions,
            "memory_expressions": self.memory_expressions[:8],
            "conflicts": self.conflicts[:6],
        }


def extract_temporal(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    low = (text or "").lower()
    for phrase, days in _RELATIVE.items():
        if phrase in low:
            out.append({"expr": phrase, "kind": "relative", "days": days})
    for m in _ABS_DATE_RE.finditer(text or ""):
        val = next(g for g in m.groups() if g)
        out.append({"expr": val, "kind": "absolute", "days": None})
    for m in re.finditer(r"\b(?:19|20)\d{2}\b", text or ""):
        out.append({"expr": m.group(0), "kind": "year", "days": None})
    # dedupe
    seen, uniq = set(), []
    for e in out:
        if e["expr"] not in seen:
            seen.add(e["expr"])
            uniq.append(e)
    return uniq


class TemporalEngine:
    def __init__(self, reference_time: Optional[float] = None):
        self.reference_time = reference_time

    def analyze(
        self,
        memory: ConversationMemory,
        prompt: str = "",
        fresh_evidence: Optional[List[str]] = None,
    ) -> TemporalResult:
        res = TemporalResult()
        pexpr = extract_temporal(prompt)
        res.prompt_expressions = [e["expr"] for e in pexpr]

        # freshness demand: prompt asks about *current* state
        low = (prompt or "").lower()
        res.freshness_demand = any(cue in low for cue in _FRESHNESS_CUES)

        # memory temporal content, linked to relevant claims
        rel = memory.relevant_claims(prompt, top_k=8, min_sim=0.08) if prompt else []
        relevant_turns = {c.turn_id for c, _ in rel}
        for t in memory.turns:
            if relevant_turns and t.turn_id not in relevant_turns:
                continue
            for e in extract_temporal(t.text):
                res.memory_expressions.append({"turn": t.turn_id, **e})

        # staleness: relevant memory only speaks in the past / has old dates
        past_cues = [
            e for e in res.memory_expressions
            if e["kind"] == "relative" and (e["days"] is None or (e["days"] or 0) >= 30)
        ]
        absolute_years = [int(e["expr"][:4]) for e in res.memory_expressions if e["kind"] == "year"]
        now_year = datetime.now(timezone.utc).year
        if self.reference_time:
            now_year = datetime.fromtimestamp(self.reference_time, tz=timezone.utc).year
        old_years = [y for y in absolute_years if now_year - y >= 2]

        # conflicting temporal statements about the same topic
        by_topic: Dict[str, List[Dict[str, Any]]] = {}
        for e in res.memory_expressions:
            if e["kind"] == "relative":
                by_topic.setdefault("relative", []).append(e)
        if len({e["expr"] for e in by_topic.get("relative", [])}) >= 3:
            res.conflicts.append(
                {"type": "multiple_relative_frames", "exprs": sorted({e["expr"] for e in by_topic["relative"]})}
            )

        # fresh external evidence can satisfy a freshness demand
        fresh_hits = [
            t for t in (fresh_evidence or [])
            if re.search(r"\b(current|latest|now|2026|2025|recent|up-to-date|up to date)\b", t, re.I)
        ]

        # age estimate of relevant memory (turn index as proxy when no clock)
        age_score = 0.0
        if res.freshness_demand and fresh_hits:
            res.reasons.append(
                f"fresh external evidence available ({len(fresh_hits)} passage(s)) — stale-memory risk reduced"
            )
            res.temporal_uncertainty = False
        elif res.freshness_demand:
            if past_cues:
                age_score = 0.6
                res.stale_memory = True
                res.reasons.append(
                    "prompt demands current information but relevant memory is framed in the past"
                )
            if old_years:
                age_score = max(age_score, 0.75)
                res.stale_memory = True
                res.reasons.append(f"relevant memory references old date(s): {sorted(set(old_years))}")
            if not res.memory_expressions:
                age_score = 0.5
                res.temporal_uncertainty = True
                res.reasons.append("freshness requested but no temporal anchor exists in memory")
            elif not past_cues and not old_years and res.memory_expressions:
                # we cannot be sure memory is fresh without external retrieval
                res.temporal_uncertainty = True
                res.reasons.append("TEMPORAL_UNCERTAINTY: freshness not verifiable offline")

        # conflicting frames -> uncertainty even without freshness demand
        if fresh_hits and age_score > 0:
            age_score = min(age_score, 0.2)
            res.stale_memory = False
        conflict_score = min(0.6, 0.3 * len(res.conflicts)) if res.conflicts else 0.0
        if res.conflicts:
            res.temporal_uncertainty = True
            res.reasons.append("conflicting temporal frames in relevant memory")

        # prompt uses vague deictic expressions
        vague = [e for e in pexpr if e["kind"] == "relative" and e["days"] is None]
        vague_score = min(0.4, 0.2 * len(vague)) if vague else 0.0
        if vague:
            res.temporal_uncertainty = True
            res.reasons.append(f"vague temporal reference(s) in prompt: {[v['expr'] for v in vague]}")

        res.score = float(min(1.0, max(age_score, conflict_score, vague_score)))
        return res
