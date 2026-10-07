"""Explainability (spec section 28).

Never outputs only "Risk = 0.82": always produces top contributing factors
with signed contributions, combining

* model-based attributions (SHAP for tree models, |coef| for linear models,
  gradient x input for the neural model), and
* engine-level human-readable reasons (presupposition, contamination,
  contradiction, ambiguity, evidence, temporal).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_HUMAN = {
    "prompt_unsupported_presupposition": "unsupported presupposition",
    "prompt_presupposition_count": "presupposed claims in prompt",
    "prov_contamination": "model-memory contamination",
    "prov_unverified_model_ratio": "unverified model claims in memory",
    "prov_model_generated": "model-generated claims in memory",
    "prov_mean_trust": "low provenance trust",
    "cons_contradiction_score": "cross-turn contradiction",
    "cons_history_contradiction": "conflicting memory",
    "cons_prompt_contradiction": "prompt conflicts with memory",
    "cons_entity_drift": "entity drift / replacement",
    "cons_reference_ambiguity": "referential ambiguity",
    "cons_temporal_inconsistency": "temporal inconsistency",
    "cons_temporal_uncertainty": "temporal uncertainty",
    "ev_risk": "weak evidence coverage",
    "ev_coverage": "evidence coverage",
    "ev_model_only": "evidence is model-generated only",
    "ev_contradictory": "contradictory evidence",
    "ev_stale": "stale evidence",
    "history_semantic_distance": "history far from prompt",
    "prompt_uncertainty_markers": "hedging / uncertainty in prompt",
    "prompt_freshness_cue": "freshness demanded by prompt",
    "prompt_q_current": "prompt asks for current information",
    "prompt_reference_count": "anaphoric references in prompt",
    "prompt_num_entities": "entity density in prompt",
    "history_relevant_turn_count": "relevant history depth",
    "ev_available": "evidence availability",
}


@dataclass
class Factor:
    feature: str
    label: str
    contribution: float
    value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "label": self.label,
            "contribution": round(self.contribution, 4),
            "value": round(self.value, 4),
        }


@dataclass
class Explanation:
    risk: float
    top_factors: List[Factor] = field(default_factory=list)
    engine_reasons: List[str] = field(default_factory=list)
    method: str = "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk": round(self.risk, 4),
            "method": self.method,
            "top_factors": [f.to_dict() for f in self.top_factors],
            "engine_reasons": self.engine_reasons,
        }


class Explainer:
    """Signed feature contributions for a single instance."""

    def __init__(self, model: Any = None, feature_names: Optional[Sequence[str]] = None):
        self.model = model
        self.feature_names = list(feature_names or [])
        self._shap_values: Optional[np.ndarray] = None
        self.method = "heuristic"
        if model is not None and len(self.feature_names):
            self.method = self._init_method()

    def _init_method(self) -> str:
        name = getattr(self.model, "name", None)
        base = getattr(self.model, "_model", None)
        try:
            if name in ("lr",) and hasattr(base, "coef_"):
                return "linear"
            if name in ("rf", "xgb", "lgbm") and hasattr(base, "feature_importances_"):
                return "shap"
            if name in ("neural", "mlp"):
                return "gradient"
        except Exception:
            pass
        return "heuristic"

    # --------------------------------------------------------------- shap
    def _shap_contributions(self, x: np.ndarray, background: Optional[np.ndarray]) -> Optional[np.ndarray]:
        try:
            import shap

            base = getattr(self.model, "_model", None)
            if base is None:
                return None
            if background is None or len(background) == 0:
                background = np.zeros((1, x.shape[0]))
            explainer = shap.TreeExplainer(base)
            sv = explainer.shap_values(np.asarray(x).reshape(1, -1))
            if isinstance(sv, list):
                sv = sv[-1]
            arr = np.asarray(sv, dtype=float)
            if arr.ndim == 3:            # (sample, feature, class) output
                arr = arr[..., -1]       # contributions to the positive class
            if arr.ndim > 1 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.size != x.size:
                return None
            return arr.reshape(-1)
        except Exception:
            return None

    def explain(
        self,
        x: np.ndarray,
        features: Dict[str, float],
        engine_reasons: Optional[List[str]] = None,
        background: Optional[np.ndarray] = None,
        top_k: int = 6,
    ) -> Explanation:
        x = np.asarray(x, dtype=float).reshape(-1)
        names = self.feature_names or [f"f{i}" for i in range(len(x))]
        contrib = np.zeros(len(x), dtype=float)
        method = self.method

        if self.model is not None:
            base = getattr(self.model, "_model", None)
            try:
                if method == "linear" and hasattr(base, "coef_"):
                    coef = np.asarray(base.coef_, dtype=float).reshape(-1)
                    contrib = coef * (x - float(np.mean(x)))
                elif method == "shap":
                    sv = self._shap_contributions(x, background)
                    if sv is not None and len(sv) == len(x):
                        contrib = sv
                    else:
                        method = "feature_importance"
                if method == "feature_importance":
                    imp_map = getattr(self.model, "feature_importance")()
                    imp = np.array([imp_map.get(n, 0.0) for n in names])
                    # unsigned importance turned into a signed push relative to baseline 0
                    contrib = imp * (x - 0.0)
                elif method == "gradient":
                    imp_map = getattr(self.model, "feature_importance")()
                    contrib = np.array([imp_map.get(n, 0.0) for n in names]) * x
            except Exception:
                method = "heuristic"

        if method == "heuristic":
            # transparent linear surrogate over normalized signals
            contrib = np.array([self._heuristic_weight(n) * v for n, v in zip(names, x)])

        order = np.argsort(-np.abs(contrib))[:top_k]
        factors = [
            Factor(
                feature=names[i],
                label=_HUMAN.get(names[i], names[i]),
                contribution=float(contrib[i]),
                value=float(features.get(names[i], x[i])),
            )
            for i in order
            if abs(contrib[i]) > 1e-6
        ]
        factors.sort(key=lambda f: -abs(f.contribution))
        return Explanation(
            risk=float(features.get("_risk", 0.0)),
            top_factors=factors,
            engine_reasons=list(engine_reasons or []),
            method=method,
        )

    @staticmethod
    def _heuristic_weight(name: str) -> float:
        table = {
            "prompt_unsupported_presupposition": 0.30,
            "prov_contamination": 0.25,
            "cons_contradiction_score": 0.18,
            "ev_risk": 0.16,
            "cons_reference_ambiguity": 0.14,
            "cons_entity_drift": 0.10,
            "cons_temporal_inconsistency": 0.10,
            "cons_temporal_uncertainty": 0.06,
            "prov_unverified_model_ratio": 0.12,
            "ev_model_only": 0.12,
            "ev_contradictory": 0.10,
            "prompt_presupposition_count": 0.08,
            "prompt_uncertainty_markers": 0.05,
            "prompt_freshness_cue": 0.05,
            "history_semantic_distance": 0.05,
            "prompt_reference_count": 0.06,
            "ev_coverage": -0.18,
            "prov_mean_trust": -0.15,
            "ev_available": -0.10,
            "history_relevant_turn_count": -0.04,
        }
        return table.get(name, 0.0)
