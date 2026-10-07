"""Unified feature engine (spec section 12).

Produces a named, ordered feature vector grouped as:

    prompt_*      history_*     provenance_*     consistency_*
    evidence_*    model_* (white-box only)

Ablations select subsets of these groups; the ordering is stable so that
trained models can be reloaded safely.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..ambiguity import AmbiguityResult
from ..contradiction import ContradictionResult
from ..entity import EntityDriftResult
from ..memory import ConversationMemory, ProvenanceType, VerificationStatus
from ..presupposition import PresuppositionResult
from ..temporal import TemporalResult
from ..utils.text import content_tokens, similarity, tokenize
from .evidence import EvidenceResult

FEATURE_GROUPS: Dict[str, List[str]] = {
    "prompt": [
        "prompt_char_len", "prompt_token_len", "prompt_is_question", "prompt_num_entities",
        "prompt_has_number", "prompt_has_negation", "prompt_uncertainty_markers",
        "prompt_presupposition_count", "prompt_unsupported_presupposition",
        "prompt_reference_count", "prompt_freshness_cue", "prompt_q_why", "prompt_q_how",
        "prompt_q_current", "prompt_q_count", "prompt_q_compare",
    ],
    "history": [
        "history_turn_count", "history_claim_count", "history_relevant_turn_count",
        "history_mean_relevance", "history_semantic_distance", "history_char_len",
        "history_user_claims", "history_assistant_claims", "history_reference_density",
        "history_recency_relevance", "history_max_depth",
    ],
    "provenance": [
        "prov_user_asserted", "prov_verified", "prov_retrieved", "prov_model_generated",
        "prov_model_inferred", "prov_unverified", "prov_unknown", "prov_mean_trust",
        "prov_unverified_model_ratio", "prov_contamination",
    ],
    "consistency": [
        "cons_contradiction_score", "cons_history_contradiction", "cons_prompt_contradiction",
        "cons_entity_drift", "cons_temporal_inconsistency", "cons_reference_ambiguity",
        "cons_conflict_pairs", "cons_temporal_uncertainty",
    ],
    "evidence": [
        "ev_coverage", "ev_retrieval_similarity", "ev_supporting", "ev_contradictory",
        "ev_stale", "ev_model_only", "ev_available", "ev_reliability", "ev_risk",
    ],
    "model": [
        "wb_entropy", "wb_logit_margin", "wb_rep_norm", "wb_max_prob",
        "wb_entropy_first_token", "wb_hidden_norm_mean",
    ],
}

ABLATION_GROUPS = {
    "A_prompt_only": ["prompt"],
    "B_prompt_history": ["prompt", "history"],
    "C_history_provenance": ["prompt", "history", "provenance"],
    "D_history_provenance_contradiction": ["prompt", "history", "provenance", "consistency"],
    "E_history_provenance_evidence": ["prompt", "history", "provenance", "evidence"],
    "F_full": ["prompt", "history", "provenance", "consistency", "evidence"],
    "G_full_whitebox": ["prompt", "history", "provenance", "consistency", "evidence", "model"],
}

_Q_UNC = re.compile(r"\b(may|might|could|possibly|perhaps|probably|maybe|likely|unclear|unsure|approx)\b", re.I)
_Q_FRESH = re.compile(r"\b(current|latest|today|now|recent|currently|up-to-date|nowadays|version)\b", re.I)


@dataclass
class FeatureContext:
    prompt: str
    memory: ConversationMemory
    contamination_score: float = 0.0
    contamination: Optional[Any] = None
    contradiction: Optional[ContradictionResult] = None
    entity: Optional[EntityDriftResult] = None
    temporal: Optional[TemporalResult] = None
    presupposition: Optional[PresuppositionResult] = None
    ambiguity: Optional[AmbiguityResult] = None
    evidence: Optional[EvidenceResult] = None
    whitebox: Dict[str, float] = field(default_factory=dict)
    relevant_top_k: int = 8


class FeatureEngine:
    def __init__(self, groups: Optional[Sequence[str]] = None):
        self.groups: List[str] = list(groups or ["prompt", "history", "provenance", "consistency", "evidence"])

    @property
    def feature_names(self) -> List[str]:
        names: List[str] = []
        for g in self.groups:
            names.extend(FEATURE_GROUPS.get(g, []))
        return names

    def extract(self, ctx: FeatureContext) -> Dict[str, float]:
        f: Dict[str, float] = {}
        f.update(self._prompt(ctx))
        f.update(self._history(ctx))
        f.update(self._provenance(ctx))
        f.update(self._consistency(ctx))
        f.update(self._evidence(ctx))
        if "model" in self.groups:
            f.update(self._whitebox(ctx))
        return {k: float(v) for k, v in f.items() if k in set(self.feature_names)}

    # ------------------------------------------------------------- prompt
    def _prompt(self, ctx: FeatureContext) -> Dict[str, float]:
        p = ctx.prompt or ""
        toks = tokenize(p)
        low = p.lower()
        ents = set(re.findall(r"\b([A-Z][A-Za-z0-9]+(?:[\s\-][A-Z][A-Za-z0-9]*)*)\b", p))
        pre = ctx.presupposition
        amb = ctx.ambiguity
        return {
            "prompt_char_len": len(p),
            "prompt_token_len": len(toks),
            "prompt_is_question": 1.0 if "?" in p or re.match(r"^(what|why|how|when|where|who|which|is|are|did|do|does)\b", low) else 0.0,
            "prompt_num_entities": len(ents),
            "prompt_has_number": 1.0 if re.search(r"\d", p) else 0.0,
            "prompt_has_negation": 1.0 if re.search(r"\b(not|no|never|none|without|n't)\b", low) else 0.0,
            "prompt_uncertainty_markers": len(_Q_UNC.findall(p)),
            "prompt_presupposition_count": float(pre.count if pre else 0.0),
            "prompt_unsupported_presupposition": float(pre.unsupported_count if pre else 0.0),
            "prompt_reference_count": float(len(amb.references) if amb else 0.0),
            "prompt_freshness_cue": 1.0 if _Q_FRESH.search(p) else 0.0,
            "prompt_q_why": 1.0 if re.match(r"^why\b", low) or " why " in f" {low} " else 0.0,
            "prompt_q_how": 1.0 if re.match(r"^how\b", low) or " how " in f" {low} " else 0.0,
            "prompt_q_current": 1.0 if _Q_FRESH.search(p) else 0.0,
            "prompt_q_count": 1.0 if re.search(r"\bhow many|how much|number of|count\b", low) else 0.0,
            "prompt_q_compare": 1.0 if re.search(r"\bbetter|worse|compare|versus|vs\.?|difference\b", low) else 0.0,
        }

    # ------------------------------------------------------------ history
    def _history(self, ctx: FeatureContext) -> Dict[str, float]:
        mem = ctx.memory
        p = ctx.prompt or ""
        claims = mem.all_claims
        rel = mem.relevant_claims(p, top_k=ctx.relevant_top_k) if p else []
        rel_turns = {c.turn_id for c, _ in rel}
        rel_sims = [s for _, s in rel]
        chars = sum(len(t.text) for t in mem.turns)
        ref_tokens = ("it", "they", "them", "this", "that", "these", "those", "previous", "earlier", "our")
        ref_count = sum(
            1 for t in mem.turns for tok in tokenize(t.text) if tok in ref_tokens
        )
        # recency-weighted relevance: recent relevant turns matter more
        rec = 0.0
        if rel and mem.turns:
            max_id = max(t.turn_id for t in mem.turns)
            for c, s in rel:
                rec += s * (c.turn_id / max(1, max_id))
        return {
            "history_turn_count": len(mem.turns),
            "history_claim_count": len(claims),
            "history_relevant_turn_count": len(rel_turns),
            "history_mean_relevance": sum(rel_sims) / len(rel_sims) if rel_sims else 0.0,
            # No history means no measurable semantic drift, not maximum drift.
            "history_semantic_distance": (
                1.0 - max(rel_sims) if mem.turns and rel_sims else 0.0
            ),
            "history_char_len": chars,
            "history_user_claims": len(mem.claims_by_speaker("user")),
            "history_assistant_claims": len(mem.claims_by_speaker("assistant")),
            "history_reference_density": ref_count / max(1, len(mem.turns)),
            "history_recency_relevance": rec,
            "history_max_depth": len(mem.turns),
        }

    # ---------------------------------------------------------- provenance
    def _provenance(self, ctx: FeatureContext) -> Dict[str, float]:
        claims = ctx.memory.all_claims
        n = max(1, len(claims))
        cnt = {p: 0 for p in ProvenanceType}
        for c in claims:
            cnt[c.provenance_type] += 1
        unverified = sum(1 for c in claims if c.is_unverified)
        unverified_model = sum(1 for c in claims if c.is_model_claim and c.is_unverified)
        return {
            "prov_user_asserted": cnt[ProvenanceType.USER_ASSERTED] / n,
            "prov_verified": cnt[ProvenanceType.SOURCE_VERIFIED] / n,
            "prov_retrieved": cnt[ProvenanceType.RETRIEVED] / n,
            "prov_model_generated": cnt[ProvenanceType.MODEL_GENERATED] / n,
            "prov_model_inferred": cnt[ProvenanceType.MODEL_INFERRED] / n,
            "prov_unverified": unverified / n,
            "prov_unknown": cnt[ProvenanceType.UNKNOWN] / n,
            "prov_mean_trust": sum(c.trust for c in claims) / n if claims else 0.0,
            "prov_unverified_model_ratio": unverified_model / n,
            "prov_contamination": ctx.contamination_score,
        }

    # --------------------------------------------------------- consistency
    def _consistency(self, ctx: FeatureContext) -> Dict[str, float]:
        con, ent, tmp, amb, pre = (
            ctx.contradiction, ctx.entity, ctx.temporal, ctx.ambiguity, ctx.presupposition
        )
        return {
            "cons_contradiction_score": con.score if con else 0.0,
            "cons_history_contradiction": con.history_conflict_score if con else 0.0,
            "cons_prompt_contradiction": con.prompt_conflict_score if con else 0.0,
            "cons_entity_drift": ent.score if ent else 0.0,
            "cons_temporal_inconsistency": tmp.score if tmp else 0.0,
            "cons_reference_ambiguity": amb.score if amb else 0.0,
            "cons_conflict_pairs": float(len(con.pairs)) if con else 0.0,
            "cons_temporal_uncertainty": 1.0 if (tmp and tmp.temporal_uncertainty) else 0.0,
        }

    # ----------------------------------------------------------- evidence
    def _evidence(self, ctx: FeatureContext) -> Dict[str, float]:
        ev = ctx.evidence
        if ev is None:
            return {k: 0.0 for k in FEATURE_GROUPS["evidence"]}
        return {
            "ev_coverage": ev.coverage_score,
            "ev_retrieval_similarity": ev.retrieval_similarity,
            "ev_supporting": float(ev.supporting_passages),
            "ev_contradictory": float(ev.contradictory_passages),
            "ev_stale": 1.0 if ev.stale else 0.0,
            "ev_model_only": 1.0 if ev.model_only else 0.0,
            "ev_available": 1.0 if ev.available else 0.0,
            "ev_reliability": ev.source_reliability,
            "ev_risk": ev.score,
        }

    # ------------------------------------------------------------ whitebox
    def _whitebox(self, ctx: FeatureContext) -> Dict[str, float]:
        wb = ctx.whitebox or {}
        return {k: float(wb.get(k, 0.0)) for k in FEATURE_GROUPS["model"]}
