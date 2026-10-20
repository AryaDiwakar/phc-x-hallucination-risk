"""Post-hoc verification (spec section 17).

Multi-strategy claim verification: claim extraction -> retrieval -> lexical /
numeric / negation entailment checks -> optional transformer NLI (if a local
model is configured) -> claim-level labels.

An external LLM may be used ONLY as an additional, clearly-separated signal
(PHC_NLI_MODEL); it is never the sole source of ground truth.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..memory import ConversationMemory, ProvenanceType
from ..retrieval.local import LocalRetriever, RetrievedPassage
from ..utils.text import content_tokens, negation_present, sent_split, similarity

_NUM_RE = re.compile(r"\d[\d,]*\.?\d*\s*(?:%|percent|samples?|million|billion)?")


@dataclass
class VerifiedClaim:
    text: str
    status: str  # SUPPORTED | UNSUPPORTED | CONTRADICTED
    support_score: float = 0.0
    evidence: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.text,
            "status": self.status,
            "support_score": round(self.support_score, 4),
            "evidence": self.evidence[:200],
            "reason": self.reason,
        }


@dataclass
class VerificationResult:
    hallucinated_claims: List[VerifiedClaim] = field(default_factory=list)
    supported_claims: List[VerifiedClaim] = field(default_factory=list)
    unsupported_claims: List[VerifiedClaim] = field(default_factory=list)
    contradicted_claims: List[VerifiedClaim] = field(default_factory=list)
    label: int = 0  # 1 = at least one hallucinated (unsupported/contradicted) claim
    hallucination_rate: float = 0.0
    strategy: str = "retrieval+nli-lite"
    nli_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "hallucinated_claims": [c.to_dict() for c in self.hallucinated_claims],
            "supported_claims": [c.to_dict() for c in self.supported_claims],
            "unsupported_claims": [c.to_dict() for c in self.unsupported_claims],
            "contradicted_claims": [c.to_dict() for c in self.contradicted_claims],
            "strategy": self.strategy,
            "nli_available": self.nli_available,
        }


def extract_answer_claims(answer: str) -> List[str]:
    out = []
    for s in sent_split(answer):
        t = s.strip()
        if len(content_tokens(t)) < 3:
            continue
        if re.match(r"^(sure|certainly|of course|okay|ok|here)\b", t, re.I) and len(t.split()) < 6:
            continue
        out.append(t)
    return out[:12]


class _NLIBackend:
    """Optional local NLI model; loaded only if PHC_NLI_MODEL is set."""

    def __init__(self) -> None:
        self.available = False
        self._pipe = None
        name = os.environ.get("PHC_NLI_MODEL", "").strip()
        if name:
            try:
                from transformers import pipeline

                self._pipe = pipeline("text-classification", model=name, top_k=None)
                self.available = True
            except Exception:
                self._pipe = None
                self.available = False

    def score(self, premise: str, hypothesis: str) -> Dict[str, float]:
        if not self.available or self._pipe is None:
            return {}
        try:
            out = self._pipe(f"{premise} [SEP] {hypothesis}")[0]
            return {i["label"].lower(): float(i["score"]) for i in out}
        except Exception:
            return {}


class Verifier:
    def __init__(
        self,
        retriever: Optional[LocalRetriever] = None,
        support_threshold: float = 0.42,
        contradiction_threshold: float = 0.55,
        nli: Optional[_NLIBackend] = None,
    ):
        self.retriever = retriever or LocalRetriever()
        self.support_threshold = support_threshold
        self.contradiction_threshold = contradiction_threshold
        self.nli = nli or _NLIBackend()

    # ------------------------------------------------------------------ core
    def verify(
        self,
        answer: str,
        prompt: str = "",
        memory: Optional[ConversationMemory] = None,
        passages: Optional[Sequence[RetrievedPassage]] = None,
    ) -> VerificationResult:
        res = VerificationResult(nli_available=self.nli.available)
        claims = extract_answer_claims(answer)
        if not claims:
            return res

        # evidence pool: retrieved passages + trusted memory claims
        evidence: List[tuple[str, str]] = []
        if passages:
            evidence.extend((p.text, f"doc:{p.doc_id}") for p in passages)
        elif prompt:
            for p in self.retriever.retrieve(f"{prompt} {answer}", top_k=6):
                evidence.append((p.text, f"doc:{p.doc_id}"))
        if memory is not None:
            for c in memory.all_claims:
                # user assertions / source-verified / retrieved claims count as
                # conversation evidence; unverified model claims NEVER do.
                if (not c.is_model_claim) or c.trust >= 0.45:
                    evidence.append((c.text, f"memory:turn{c.turn_id}"))

        for claim in claims:
            vc = self._score_claim(claim, evidence)
            if vc.status == "SUPPORTED":
                res.supported_claims.append(vc)
            elif vc.status == "CONTRADICTED":
                res.contradicted_claims.append(vc)
                res.hallucinated_claims.append(vc)
            else:
                res.unsupported_claims.append(vc)
                res.hallucinated_claims.append(vc)

        res.label = 1 if res.hallucinated_claims else 0
        res.hallucination_rate = len(res.hallucinated_claims) / max(1, len(claims))
        return res

    def _score_claim(self, claim: str, evidence: Sequence[tuple[str, str]]) -> VerifiedClaim:
        best, best_src, status, reason = 0.0, "", "UNSUPPORTED", "no supporting evidence found"
        claim_nums = set(_NUM_RE.findall(claim))
        claim_neg = negation_present(claim)

        for ev, src in evidence:
            s = similarity(claim, ev)
            ev_nums = set(_NUM_RE.findall(ev))
            ev_neg = negation_present(ev)

            # numeric consistency check
            if claim_nums and ev_nums:
                if claim_nums <= ev_nums or ev_nums <= claim_nums:
                    s = min(1.0, s + 0.30)
                elif claim_nums & ev_nums:
                    s *= 0.75
                else:
                    s = min(s, 0.30)

            # negation check: same content, opposite polarity -> contradiction
            if s >= 0.35 and claim_neg != ev_neg and abs(len(claim) - len(ev)) < 200:
                if s >= self.contradiction_threshold:
                    return VerifiedClaim(claim, "CONTRADICTED", s, ev, "polarity conflict with evidence")
                s *= 0.6

            if s > best:
                best, best_src = s, ev

            # optional NLI signal
            if self.nli.available:
                scores = self.nli.score(ev, claim)
                if scores:
                    ent = scores.get("entailment", 0.0)
                    con = scores.get("contradiction", 0.0)
                    if ent >= 0.7 and ent > best:
                        best, best_src, status, reason = ent, ev, "SUPPORTED", "NLI entailment"
                    if con >= 0.7:
                        return VerifiedClaim(claim, "CONTRADICTED", con, ev, "NLI contradiction")

        if status != "SUPPORTED":
            if best >= self.support_threshold:
                status, reason = "SUPPORTED", "lexical/numeric support in evidence"
            elif best >= 0.25:
                status, reason = "UNSUPPORTED", f"weak support only ({best:.2f})"
            else:
                status, reason = "UNSUPPORTED", "no supporting evidence found"
        return VerifiedClaim(claim, status, best, best_src, reason)
