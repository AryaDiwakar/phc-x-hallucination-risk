"""PHC-X end-to-end risk pipeline.

    history + prompt
        -> ConversationMemory (provenance-aware)
        -> engines (contamination, contradiction, entity, temporal,
                    presupposition, ambiguity, retrieval, evidence)
        -> FeatureEngine
        -> RiskModel (learned) or transparent heuristic aggregator
        -> calibration
        -> policy decision + explanation
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ambiguity import AmbiguityEngine, AmbiguityResult
from ..contradiction import ContradictionEngine, ContradictionResult
from ..entity import EntityDriftResult, EntityTracker
from ..explain import Explainer, Explanation
from ..features import EvidenceEngine, FeatureContext, FeatureEngine, EvidenceResult
from ..intervention import Decision, RiskPolicy
from ..memory import ConversationMemory, Turn
from ..models import ModelAdapter, get_adapter
from ..presupposition import PresuppositionEngine, PresuppositionResult
from ..retrieval import LocalRetriever, RetrievedPassage
from ..temporal import TemporalEngine, TemporalResult

# transparent zero-shot aggregator (also used as an explicit baseline)
_HEURISTIC_WEIGHTS: Dict[str, float] = {
    "prompt_unsupported_presupposition": 0.55,
    "prov_contamination": 0.50,
    "cons_contradiction_score": 0.45,
    "ev_risk": 0.40,
    "cons_reference_ambiguity": 0.35,
    "cons_entity_drift": 0.30,
    "cons_temporal_inconsistency": 0.30,
    "prov_unverified_model_ratio": 0.12,
    "ev_model_only": 0.25,
    "prompt_uncertainty_markers": 0.12,
    "history_semantic_distance": 0.06,
}


def heuristic_risk(features: Dict[str, float]) -> float:
    """Weighted noisy-OR over explainable risk signals (always available)."""
    p = 1.0
    for k, w in _HEURISTIC_WEIGHTS.items():
        v = float(features.get(k, 0.0))
        if k in ("prompt_uncertainty_markers",):
            v = min(1.0, v / 2.0)
        if k == "history_semantic_distance":
            # only matters when there IS history
            v = v if features.get("history_turn_count", 0) > 0 else 0.0
        v = max(0.0, min(1.0, v))
        if v > 0:
            p *= 1.0 - w * v
    risk = 1.0 - p
    # strong, non-contradictory, non-model-generated evidence actively
    # reduces risk (spec section 11: verified evidence -> risk down)
    ev_cov = float(features.get("ev_coverage", 0.0))
    trust = float(features.get("prov_mean_trust", 0.0))
    ev_clean = (
        float(features.get("ev_contradictory", 0.0)) == 0.0
        and float(features.get("ev_model_only", 0.0)) == 0.0
    )
    if ev_clean and ev_cov > 0.30:
        risk *= 1.0 - 0.60 * (ev_cov - 0.30)
    risk *= 1.0 - 0.25 * ev_cov * trust
    return float(max(0.0, min(1.0, risk)))


@dataclass
class RiskPrediction:
    risk_score: float
    risk_level: str
    recommended_action: str
    factors: List[Dict[str, Any]] = field(default_factory=list)
    engine_reasons: List[str] = field(default_factory=list)
    provenance_warnings: List[str] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    ambiguities: List[Dict[str, Any]] = field(default_factory=list)
    contamination: Dict[str, Any] = field(default_factory=dict)
    temporal: Dict[str, Any] = field(default_factory=dict)
    presuppositions: List[Dict[str, Any]] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    decision: Dict[str, Any] = field(default_factory=dict)
    explanation: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    mode: str = "black_box"
    calibrated: bool = False
    latency_ms: float = 0.0
    memory_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level,
            "recommended_action": self.recommended_action,
            "factors": self.factors,
            "engine_reasons": self.engine_reasons,
            "provenance_warnings": self.provenance_warnings,
            "contradictions": self.contradictions,
            "ambiguities": self.ambiguities,
            "contamination": self.contamination,
            "temporal": self.temporal,
            "presuppositions": self.presuppositions,
            "evidence": self.evidence,
            "decision": self.decision,
            "explanation": self.explanation,
            "mode": self.mode,
            "calibrated": self.calibrated,
            "latency_ms": round(self.latency_ms, 3),
            "memory_summary": self.memory_summary,
            "features": {k: round(v, 4) for k, v in self.features.items()},
        }


class PHCPipeline:
    def __init__(
        self,
        model: Any = None,
        policy: Optional[RiskPolicy] = None,
        retriever: Optional[LocalRetriever] = None,
        groups: Optional[Sequence[str]] = None,
        mode: str = "black_box",
        adapter: Optional[ModelAdapter] = None,
        history_turns: int = 10,
    ):
        self.model = model
        self.policy = policy or RiskPolicy()
        self.retriever = retriever or LocalRetriever()
        self.mode = mode
        self.adapter = adapter
        self.history_turns = history_turns
        self.features = FeatureEngine(groups=groups)
        self.contradiction_engine = ContradictionEngine()
        self.entity_engine = EntityTracker()
        self.temporal_engine = TemporalEngine()
        self.presupposition_engine = PresuppositionEngine()
        self.ambiguity_engine = AmbiguityEngine()
        self.evidence_engine = EvidenceEngine()
        self.explainer = Explainer(
            model=model,
            feature_names=list(getattr(model, "feature_names", None) or self.features.feature_names),
        )

    # ------------------------------------------------------------------ main
    def analyze(
        self,
        prompt: str,
        conversation_history: Optional[Sequence[Dict[str, Any]]] = None,
        evidence: Optional[Sequence[Any]] = None,
        whitebox_signals: Optional[Dict[str, float]] = None,
        conversation_id: str = "conv",
        use_retriever: bool = True,
        prefer_clarify: bool = False,
    ) -> RiskPrediction:
        t0 = time.perf_counter()

        # 1) conversation capture -> provenance-aware memory
        mem = ConversationMemory.from_history(list(conversation_history or []), conversation_id=conversation_id)
        if len(mem.turns) > self.history_turns and self.history_turns > 0:
            # keep the most recent N turns for feature computation but retain
            # full memory for provenance/contamination queries.
            pass

        # 2) contamination (uses FULL memory, never a truncated view)
        contam = mem.contamination(prompt)

        # 3) consistency engines
        contradiction = self.contradiction_engine.analyze(mem, prompt)
        entity = self.entity_engine.track(mem, prompt)
        temporal = self.temporal_engine.analyze(
            mem, prompt, fresh_evidence=[p.text for p in _coerce_passages(evidence)]
        )
        presupposition = self.presupposition_engine.analyze(prompt, mem, evidence_texts=_evidence_texts(evidence))
        ambiguity = self.ambiguity_engine.analyze(prompt, mem)

        # 4) retrieval + evidence coverage
        passages: List[RetrievedPassage] = list(_coerce_passages(evidence))
        if use_retriever and not passages:
            passages = self.retriever.retrieve(prompt, top_k=5)
        ev = self.evidence_engine.analyze(prompt, mem, passages)

        # 5) white-box signals only when mode says so AND adapter provides them
        wb: Dict[str, float] = dict(whitebox_signals or {})
        if self.mode == "white_box" and not wb and self.adapter is not None and self.adapter.whitebox:
            try:
                wb = self.adapter.whitebox_signals(prompt, conversation_history)
            except Exception:
                wb = {}

        # 6) features
        ctx = FeatureContext(
            prompt=prompt,
            memory=mem,
            contamination_score=contam.score,
            contamination=contam,
            contradiction=contradiction,
            entity=entity,
            temporal=temporal,
            presupposition=presupposition,
            ambiguity=ambiguity,
            evidence=ev,
            whitebox=wb,
        )
        feats = self.features.extract(ctx)
        x = np.array([feats.get(n, 0.0) for n in self.features.feature_names], dtype=float)

        # 7) risk prediction + calibration
        calibrated = False
        if self.model is not None and len(x):
            risk = float(self.model.predict_proba(x.reshape(1, -1), calibrated=True)[0])
            calibrated = getattr(self.model, "calibrator", None) is not None
        else:
            risk = heuristic_risk(feats)

        # 8) explanation
        engine_reasons = self._engine_reasons(contam, contradiction, entity, temporal,
                                              presupposition, ambiguity, ev)
        explanation = self.explainer.explain(
            x, {**feats, "_risk": risk}, engine_reasons=engine_reasons
        )
        explanation.risk = risk

        # 9) policy decision
        dec = self.policy.decide(
            risk,
            evidence_available=ev.available,
            prefer_clarify=prefer_clarify or ambiguity.should_clarify,
            conversation_id=conversation_id,
        )

        # 10) provenance warnings (never blind-trust model memory)
        warnings = self._provenance_warnings(mem, contam)

        latency = (time.perf_counter() - t0) * 1000.0
        return RiskPrediction(
            risk_score=risk,
            risk_level=dec.level,
            recommended_action=dec.action,
            factors=[f.to_dict() for f in explanation.top_factors],
            engine_reasons=engine_reasons,
            provenance_warnings=warnings,
            contradictions=[p.to_dict() for p in contradiction.pairs],
            ambiguities=ambiguity.references,
            contamination=contam.to_dict(),
            temporal=temporal.to_dict(),
            presuppositions=[p.to_dict() for p in presupposition.presuppositions],
            evidence=ev.to_dict(),
            decision=dec.to_dict(),
            explanation=explanation.to_dict(),
            features=feats,
            mode=self.mode,
            calibrated=calibrated,
            latency_ms=latency,
            memory_summary=mem.summary(),
        )

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _engine_reasons(contam, contradiction, entity, temporal, presupposition, ambiguity, ev) -> List[str]:
        reasons: List[str] = []
        if presupposition is not None and presupposition.unsupported_count:
            reasons.append(
                f"unsupported presupposition detected ({presupposition.unsupported_count})"
            )
        if contam.score >= 0.35:
            reasons.append(
                f"model-memory contamination (score={contam.score:.2f}) — "
                "a previous unverified MODEL_GENERATED claim is being reused"
            )
        if contradiction.score >= 0.4:
            reasons.append(f"cross-turn contradiction (score={contradiction.score:.2f})")
        if ambiguity.score >= 0.35:
            reasons.append(f"referential ambiguity (score={ambiguity.score:.2f})")
        if entity.score >= 0.4:
            reasons.append(f"entity drift (score={entity.score:.2f})")
        if temporal.score >= 0.4:
            reasons.append(
                f"temporal inconsistency/uncertainty (score={temporal.score:.2f})"
            )
        if ev.score >= 0.5:
            reasons.append(
                f"evidence coverage: {ev.coverage_score:.0%}"
                + (" (none found)" if not ev.available else "")
            )
        return reasons

    @staticmethod
    def _provenance_warnings(mem: ConversationMemory, contam) -> List[str]:
        warns: List[str] = []
        n_um = len(mem.unverified_model_claims)
        if n_um:
            warns.append(
                f"{n_um} unverified MODEL_GENERATED claim(s) in conversation memory "
                "— not treated as ground truth"
            )
        for c in contam.dependent_claims[:3]:
            warns.append(
                f"prompt depends on turn {c.turn_id} claim [{c.provenance_type.value}/"
                f"{c.verification_status.value}]: \"{c.text[:90]}\""
            )
        contradicted = [c for c in mem.all_claims if c.contradicting_claims]
        if contradicted:
            warns.append(f"{len(contradicted)} claim(s) participate in contradictions")
        return warns


# ---------------------------------------------------------------- internal
def _evidence_texts(evidence: Optional[Sequence[Any]]) -> List[str]:
    out: List[str] = []
    for e in evidence or []:
        if isinstance(e, str):
            out.append(e)
        elif isinstance(e, RetrievedPassage):
            out.append(e.text)
        elif isinstance(e, dict):
            out.append(str(e.get("text") or e.get("content") or ""))
        else:
            out.append(str(getattr(e, "text", "")))
    return [t for t in out if t]


def _coerce_passages(evidence: Optional[Sequence[Any]]) -> List[RetrievedPassage]:
    out: List[RetrievedPassage] = []
    for e in evidence or []:
        if isinstance(e, RetrievedPassage):
            out.append(e)
        elif isinstance(e, dict):
            out.append(
                RetrievedPassage(
                    doc_id=str(e.get("doc_id", e.get("id", "ev"))),
                    text=str(e.get("text") or e.get("content") or ""),
                    score=float(e.get("score", 0.0)),
                    title=str(e.get("title", "")),
                    source=str(e.get("source", "")),
                    timestamp=e.get("timestamp"),
                    reliability=float(e.get("reliability", 0.7)),
                )
            )
        elif isinstance(e, str):
            out.append(RetrievedPassage(doc_id="ev", text=e, score=0.0))
    return [p for p in out if p.text]
