"""Cross-turn contradiction detection (spec section 6).

Combines structured claim comparison (normalized subject / lemmatized relation
/ value slots / numbers / negation) with semantic similarity.  An LLM judge is
explicitly *not* used as the sole signal; a transformer NLI model may be used
only as an optional additional signal when available.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..memory import Claim, ConversationMemory, EdgeType, VerificationStatus
from ..utils.text import content_tokens, negation_present, similarity, tokenize

_NUMBER_RE = re.compile(
    r"(?<![\w.])(\d[\d,]*\.?\d*)\s*(%|percent|k|m|b|million|billion|thousand)?", re.I
)
_VALUE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]+(?:[\s\-][A-Z][A-Za-z0-9]*)*|\d[\d,]*\.?\d*\s*(?:%|percent|samples?|rows?)?)\b"
)
_GENERIC_SPANS = {
    "the", "a", "an", "we", "it", "they", "you", "he", "she", "this", "that",
    "our", "my", "your", "what", "which", "who", "how", "why", "when", "where",
    "i", "yes", "no", "not", "in", "on", "at", "to", "for", "and", "or",
}
_LEAD_DROP = {
    "the", "a", "an", "this", "that", "these", "those", "our", "my", "your",
    "his", "her", "their", "its", "we", "you", "they", "i",
}
_WH = {"what", "which", "who", "how", "why", "when", "where", "whose", "whatever"}

_RELATION_LEMMA = {
    "uses": "use", "use": "use", "used": "use", "using": "use",
    "is": "be", "are": "be", "was": "be", "were": "be", "be": "be", "been": "be",
    "has": "have", "have": "have", "had": "have",
    "contains": "contain", "contain": "contain", "consists": "consist",
    "equals": "equal", "equal": "equal",
    "achieves": "achieve", "achieve": "achieve", "achieved": "achieve",
    "reports": "report", "report": "report", "reported": "report",
    "produces": "produce", "produce": "produce", "produced": "produce",
    "requires": "require", "require": "require", "required": "require",
    "located": "locate", "founded": "found", "released": "release",
    "increased": "increase", "increase": "increase", "decreased": "decrease",
    "decrease": "decrease", "improved": "improve", "improve": "improve",
    "declined": "decline", "decline": "decline", "changed": "change",
    "change": "change", "runs": "run", "run": "run", "ran": "run",
    "supports": "support", "support": "support", "based": "base",
    "switched": "switch", "replaced": "replace", "became": "become",
    "become": "become", "got": "get", "got": "get",
}
_RELATION_WORDS = tuple(set(_RELATION_LEMMA) | set(_RELATION_LEMMA.values()))


@dataclass
class ContradictionPair:
    a: Dict[str, Any]
    b: Dict[str, Any]
    score: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b, "score": round(self.score, 4), "reason": self.reason}


@dataclass
class ContradictionResult:
    score: float = 0.0
    pairs: List[ContradictionPair] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    history_conflict_score: float = 0.0
    prompt_conflict_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "history_conflict_score": round(self.history_conflict_score, 4),
            "prompt_conflict_score": round(self.prompt_conflict_score, 4),
            "pairs": [p.to_dict() for p in self.pairs[:8]],
            "reasons": self.reasons[:8],
        }


def _lemma(word: str) -> str:
    return _RELATION_LEMMA.get(word, word)


def _signature(text: str) -> Dict[str, Any]:
    toks = tokenize(text, lowercase=True)
    numbers: List[float] = []
    for m in _NUMBER_RE.finditer(text):
        try:
            v = float(m.group(1).replace(",", ""))
            unit = (m.group(2) or "").lower()
            if unit in ("k", "thousand"):
                v *= 1_000
            elif unit in ("m", "million"):
                v *= 1_000_000
            elif unit in ("b", "billion"):
                v *= 1_000_000_000
            numbers.append(v)
        except ValueError:
            pass

    rel_idx = None
    relation = ""
    for i, t in enumerate(toks):
        if t in _RELATION_LEMMA:
            rel_idx, relation = i, _lemma(t)
            break
    # subject: content tokens before the relation (head noun kept)
    subject_toks = toks[:rel_idx] if rel_idx is not None else []
    subject_toks = [t for t in subject_toks if t not in _LEAD_DROP and t not in _WH]
    if not subject_toks and rel_idx is not None:
        subject_toks = toks[:rel_idx][-1:] if rel_idx else []
    subject = " ".join(subject_toks)

    values = []
    for m in _VALUE_RE.finditer(text):
        v = m.group(1).strip()
        if v.lower() in _GENERIC_SPANS or len(v) < 2:
            continue
        if v not in values:
            values.append(v)

    return {
        "subject": subject,
        "relation": relation,
        "numbers": numbers,
        "values": values,
        "negation": negation_present(text),
        "tokens": set(content_tokens(text)),
    }


def _claim_brief(c: Claim) -> Dict[str, Any]:
    return {
        "claim_id": c.claim_id,
        "turn_id": c.turn_id,
        "speaker": c.speaker,
        "text": c.text,
        "provenance": c.provenance_type.value,
        "verification": c.verification_status.value,
    }


def pair_contradiction(a: str, b: str) -> Tuple[float, str]:
    """Return (score, reason) for a potential contradiction between two statements."""
    sa, sb = _signature(a), _signature(b)
    sim = similarity(a, b)
    score, reason = 0.0, ""

    subj_a, subj_b = sa["subject"], sb["subject"]
    same_subject = bool(subj_a) and subj_a == subj_b
    # subjects of different surface form but one contains the other
    related_subject = bool(subj_a) and bool(subj_b) and (
        subj_a in subj_b or subj_b in subj_a
    )

    # 1) negation conflict on similar statements
    if sim >= 0.4 and sa["negation"] != sb["negation"]:
        score, reason = max(score, 0.75 + 0.2 * sim), "negation conflict on similar statements"

    # 2) structured comparison: same subject (or shared subject) + same relation
    if (same_subject or related_subject) and sa["relation"] and sa["relation"] == sb["relation"]:
        na, nb = sa["numbers"], sb["numbers"]
        va, vb = set(sa["values"]), set(sb["values"])
        if na and nb and not set(na) & set(nb):
            score, reason = max(score, 0.9), (
                f"same subject '{subj_a}' + relation '{sa['relation']}' with different values: "
                f"{na[0]:g} vs {nb[0]:g}"
            )
        elif va and vb and not (va & vb):
            # ignore disagreements that only stem from stopwords/generic spans
            score, reason = max(score, 0.85), (
                f"same subject '{subj_a}' + relation '{sa['relation']}' with conflicting named values: "
                f"{' '.join(sorted(va))[:50]} vs {' '.join(sorted(vb))[:50]}"
            )
        elif sim >= 0.55 and (sa["tokens"] ^ sb["tokens"]):
            diff = " ".join(sorted(sa["tokens"] ^ sb["tokens"]))[:60]
            score, reason = max(score, 0.6), f"same subject/relation, differing content ({diff})"

    # 3) explicit correction / negation markers
    if (same_subject or related_subject or sim >= 0.5) and sim >= 0.25:
        for pat, sc, why in (
            (r"\bbut (?:actually|in fact|however)\b", 0.5, "explicit correction marker"),
            (r"\bis not\b|\bdoes not\b|\bnever\b|\bno longer\b", 0.5, "explicit negation"),
            (r"\binstead of\b|\brather than\b", 0.5, "substitution marker"),
        ):
            if re.search(pat, b, re.I) or re.search(pat, a, re.I):
                score, reason = max(score, sc), why

    # 4) highly similar surface, disjoint informative content
    if sim >= 0.65 and sa["tokens"] and sb["tokens"]:
        overlap = len(sa["tokens"] & sb["tokens"]) / max(1, min(len(sa["tokens"]), len(sb["tokens"])))
        if overlap < 0.45:
            score, reason = max(score, 0.55), "high surface similarity but disjoint content"

    return float(min(1.0, score)), reason


class ContradictionEngine:
    """Detects contradictions among stored claims and against the prompt."""

    def __init__(self, min_pair_score: float = 0.5, max_pairs: int = 800):
        self.min_pair_score = min_pair_score
        self.max_pairs = max_pairs

    def analyze(
        self,
        memory: ConversationMemory,
        prompt: Optional[str] = None,
        relevant_top_k: int = 12,
    ) -> ContradictionResult:
        res = ContradictionResult()
        claims = memory.all_claims

        # ---- (a) contradictions *within* history --------------------------
        pairs: List[ContradictionPair] = []
        n = len(claims)
        budget = self.max_pairs
        for i in range(n):
            for j in range(i + 1, n):
                if budget <= 0:
                    break
                a, b = claims[i], claims[j]
                if a.turn_id == b.turn_id and a.speaker == b.speaker:
                    continue
                if not (set(content_tokens(a.text)) & set(content_tokens(b.text))):
                    continue
                budget -= 1
                sc, why = pair_contradiction(a.text, b.text)
                if sc >= self.min_pair_score:
                    pairs.append(ContradictionPair(_claim_brief(a), _claim_brief(b), sc, why))
        pairs.sort(key=lambda p: -p.score)

        hist_best = max([p.score for p in pairs], default=0.0)
        res.history_conflict_score = hist_best

        if pairs:
            memory.support_contradiction_edges(
                [(p.a["claim_id"], p.b["claim_id"], EdgeType.CONTRADICTS, p.score) for p in pairs[:20]]
            )
            by_id = {c.claim_id: c for c in claims}
            for p in pairs[:20]:
                ca, cb = by_id.get(p.a["claim_id"]), by_id.get(p.b["claim_id"])
                if ca is not None:
                    ca.contradicting_claims.append(p.b["claim_id"])
                    ca.metadata["contradiction"] = p.to_dict()
                    ca.verification_status = VerificationStatus.CONTRADICTED
                if cb is not None:
                    cb.contradicting_claims.append(p.a["claim_id"])
                    cb.metadata["contradiction"] = p.to_dict()
                    cb.verification_status = VerificationStatus.CONTRADICTED
            top = pairs[0]
            res.reasons.insert(
                0,
                f"turn {top.a['turn_id']} vs turn {top.b['turn_id']}: {top.reason} (score={top.score:.2f})",
            )

        # ---- (b) prompt vs history ---------------------------------------
        # A question does not commit to a proposition, so it cannot conflict
        # with history (its *presuppositions* are handled by the
        # presupposition engine instead).  Only assertive prompts are checked.
        prompt_best = 0.0
        if prompt and not prompt.strip().endswith("?"):
            for c, _sim in memory.relevant_claims(prompt, top_k=relevant_top_k, min_sim=0.15):
                sc, why = pair_contradiction(prompt, c.text)
                if sc > prompt_best:
                    prompt_best = sc
                if sc >= self.min_pair_score:
                    res.pairs.append(
                        ContradictionPair(
                            {"claim_id": "PROMPT", "turn_id": -1, "speaker": "user", "text": prompt,
                             "provenance": "PROMPT", "verification": "N/A"},
                            _claim_brief(c),
                            sc,
                            f"prompt-vs-history: {why}",
                        )
                    )
                    res.reasons.append(f"prompt conflicts with turn {c.turn_id}: {why}")
        res.prompt_conflict_score = prompt_best
        res.score = float(min(1.0, max(hist_best, prompt_best)))
        res.pairs.extend(pairs[:8])
        res.pairs.sort(key=lambda p: -p.score)
        res.pairs = res.pairs[:8]
        return res
