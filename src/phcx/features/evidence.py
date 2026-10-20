"""Evidence coverage scoring (spec section 11).

Answers: does the available context actually support what the prompt asks?
- strong verified evidence    -> risk down
- contradictory evidence      -> risk up
- stale / model-only evidence -> risk up
- no evidence                 -> risk up
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from ..memory import ConversationMemory, ProvenanceType, VerificationStatus
from ..retrieval.local import RetrievedPassage
from ..utils.text import similarity, tokenize


@dataclass
class EvidenceResult:
    coverage_score: float = 0.0          # 1.0 = fully covered, 0.0 = no evidence
    retrieval_similarity: float = 0.0
    supporting_passages: int = 0
    contradictory_passages: int = 0
    stale: bool = False
    model_only: bool = False
    available: bool = False
    source_reliability: float = 0.0
    score: float = 0.0                   # evidence RISK score (0 good, 1 bad)
    reasons: List[str] = field(default_factory=list)
    passages: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage_score": round(self.coverage_score, 4),
            "retrieval_similarity": round(self.retrieval_similarity, 4),
            "supporting_passages": self.supporting_passages,
            "contradictory_passages": self.contradictory_passages,
            "stale": self.stale,
            "model_only": self.model_only,
            "available": self.available,
            "source_reliability": round(self.source_reliability, 4),
            "score": round(self.score, 4),
            "reasons": self.reasons,
            "passages": self.passages[:5],
        }


_NEG = ("not", "never", "no", "false", "incorrect", "disputed", "refuted", "denied")

# A passage is considered stale once it predates the current year by more than
# two years.  The window follows the clock so the rule cannot silently rot.
_STALE_AFTER_YEARS = 2


def _timestamp_year(value: Any) -> Optional[int]:
    """Best-effort calendar year for a passage timestamp (None if unknown)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        n = float(value)
        if n > 1e8:  # epoch seconds
            try:
                return datetime.fromtimestamp(n, tz=timezone.utc).year
            except (OverflowError, OSError, ValueError):
                return None
        return int(n) if 1 <= n <= 9999 else None
    s = str(value).strip()
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return None



class EvidenceEngine:
    def __init__(self, coverage_threshold: float = 0.35):
        self.coverage_threshold = coverage_threshold

    def analyze(
        self,
        prompt: str,
        memory: ConversationMemory,
        passages: Optional[Sequence[RetrievedPassage]] = None,
    ) -> EvidenceResult:
        res = EvidenceResult()
        passages = list(passages or [])

        # ---- retrieval evidence ------------------------------------------
        rel = [(p, similarity(prompt, p.text)) for p in passages]
        rel = [(p, s) for p, s in rel if s > 0.05]
        rel.sort(key=lambda x: -x[1])
        supporting, contradicting = [], []
        for p, s in rel:
            ptoks = set(tokenize(p.text, lowercase=True))
            neg = any(n in ptoks for n in _NEG)
            (contradicting if neg and s > 0.3 else supporting).append((p, s))

        res.supporting_passages = len(supporting)
        res.contradictory_passages = len(contradicting)
        res.retrieval_similarity = max([s for _, s in rel], default=0.0)
        res.available = bool(passages)
        res.source_reliability = (
            max([p.reliability for p, _ in rel], default=0.0) if rel else 0.0
        )
        _now_year = datetime.now(timezone.utc).year
        res.stale = any(
            (y := _timestamp_year(p.timestamp)) is not None
            and y <= _now_year - _STALE_AFTER_YEARS
            for p, _ in rel
        )
        res.passages = [p.to_dict() | {"similarity": round(s, 3)} for p, s in rel[:5]]

        # ---- memory evidence ---------------------------------------------
        mem_rel = memory.relevant_claims(prompt, top_k=6, min_sim=0.1) if prompt else []
        # Memory counts as support only when it is not model-generated: a claim
        # the model asserted earlier is never evidence for its own next turn.
        mem_model_only = bool(mem_rel) and all(c.is_model_claim for c, _ in mem_rel)
        res.model_only = mem_model_only and not supporting

        # ---- coverage ------------------------------------------------------
        best_rel = max([s for _, s in mem_rel], default=0.0)
        ret_cov = res.retrieval_similarity
        mem_cov = best_rel * (sum(c.trust for c, _ in mem_rel) / len(mem_rel) if mem_rel else 0.0)
        coverage = max(ret_cov, mem_cov)
        if supporting:
            coverage = max(coverage, 0.6 + 0.4 * min(1.0, len(supporting) / 3.0))
        res.coverage_score = float(min(1.0, coverage))
        res.available = bool(passages) or bool(mem_rel)

        # ---- risk ----------------------------------------------------------
        risk = 1.0 - res.coverage_score
        if res.contradictory_passages:
            risk = max(risk, 0.7)
            res.reasons.append(f"{res.contradictory_passages} contradictory passage(s)")
        if res.model_only:
            risk = max(risk, 0.75)
            res.reasons.append("only MODEL-GENERATED memory available as evidence")
        if res.stale:
            risk = max(risk, 0.6)
            res.reasons.append("retrieved evidence is stale")
        if not res.available:
            res.reasons.append("NO EVIDENCE available (no retrieval results, no trusted memory)")
        elif res.coverage_score < self.coverage_threshold:
            res.reasons.append(
                f"evidence coverage below threshold ({res.coverage_score:.2f} < {self.coverage_threshold})"
            )
        res.score = float(max(0.0, min(1.0, risk)))
        return res
