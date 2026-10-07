"""Adaptive intervention policy (spec sections 15-16).

risk + latency budget + evidence availability + user preference -> action.
Thresholds come from configs/policy.yaml and may be re-optimized on the
validation split (never on test data).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.io import load_yaml

ACTIONS = ("GENERATE", "GENERATE_WITH_EVIDENCE", "RETRIEVE", "CLARIFY", "ABSTAIN")

DEFAULT_THRESHOLDS = {
    "low": 0.25,
    "moderate": 0.50,
    "high": 0.75,
    "clarify": 0.75,
    "retrieve": 0.50,
}


def risk_level(score: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if score < t.get("low", 0.25):
        return "LOW"
    if score < t.get("moderate", 0.50):
        return "MODERATE"
    if score < t.get("high", 0.75):
        return "HIGH"
    return "CRITICAL"


@dataclass
class Decision:
    action: str
    risk: float
    level: str
    reason: str = ""
    latency_budget_ms: float = 0.0
    evidence_available: bool = False
    alternatives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "risk": round(self.risk, 4),
            "level": self.level,
            "reason": self.reason,
            "latency_budget_ms": self.latency_budget_ms,
            "evidence_available": self.evidence_available,
            "alternatives": self.alternatives,
        }


class RiskPolicy:
    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        latency_budget_ms: float = 800.0,
        user_preference: str = "balanced",
        min_risk_for_abstain: float = 0.75,
        history: Optional[List[Dict[str, Any]]] = None,
    ):
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.latency_budget_ms = latency_budget_ms
        self.user_preference = user_preference
        self.min_risk_for_abstain = min_risk_for_abstain
        self.history: List[Dict[str, Any]] = history or []

    @classmethod
    def from_yaml(cls, path: str) -> "RiskPolicy":
        cfg = load_yaml(path)
        th = dict(cfg.get("thresholds") or {})
        overrides = cfg.get("overrides") or {}
        if "critical" in th and "high" not in th:
            th["high"] = th["critical"]
        return cls(
            thresholds=th,
            latency_budget_ms=float(overrides.get("latency_budget_ms", 800)),
            user_preference=str(overrides.get("user_preference", "balanced")),
            min_risk_for_abstain=float(overrides.get("min_risk_for_abstain", 0.75)),
        )

    def apply_thresholds(self, optimized: Dict[str, float]) -> None:
        """Merge validation-optimized cut-points into the policy.

        ``optimize_thresholds`` returns a coherent ordered set
        (low < moderate < high < clarify), so keys are merged as-is; the only
        invariant enforced is that the level cut-points stay monotonic.
        """
        optimized = dict(optimized or {})
        for k, v in optimized.items():
            if k in self.thresholds or k in ("decision", "retrieve", "clarify"):
                self.thresholds[k] = float(v)
        thr = self.thresholds
        if "low" in optimized:
            thr["low"] = min(float(thr.get("low", 0.25)), float(thr.get("moderate", 0.5)))
        if "high" in optimized:
            thr["high"] = max(float(thr.get("high", 0.75)), float(thr.get("moderate", 0.5)))
        if "clarify" in optimized:
            self.min_risk_for_abstain = max(self.min_risk_for_abstain,
                                             float(optimized["clarify"]))

    # -------------------------------------------------------------- decide
    def decide(
        self,
        risk: float,
        evidence_available: bool = False,
        latency_ms: Optional[float] = None,
        prefer_clarify: bool = False,
        conversation_id: str = "",
    ) -> Decision:
        level = risk_level(risk, self.thresholds)
        t = self.thresholds
        alts: List[str] = []

        if self.user_preference == "aggressive":
            # act earlier: escalate one level
            level = {"LOW": "MODERATE", "MODERATE": "HIGH", "HIGH": "CRITICAL", "CRITICAL": "CRITICAL"}[level]
        elif self.user_preference == "permissive":
            level = {"LOW": "LOW", "MODERATE": "LOW", "HIGH": "MODERATE", "CRITICAL": "HIGH"}[level]

        # latency-aware downgrade: if we already spent the budget, do the
        # cheapest useful action rather than blocking the user.
        spent = float(latency_ms or 0.0)
        if level in ("HIGH", "CRITICAL") and spent > self.latency_budget_ms:
            level = "HIGH" if level == "CRITICAL" else "MODERATE"

        if level == "LOW":
            action, reason = "GENERATE", "risk below retrieval threshold; generate normally"
        elif level == "MODERATE":
            action = "GENERATE_WITH_EVIDENCE" if evidence_available else "RETRIEVE"
            reason = (
                "moderate risk: generate with evidence recommendation"
                if evidence_available
                else "moderate risk: retrieve supporting evidence first"
            )
            alts = ["GENERATE", "RETRIEVE"]
        elif level == "HIGH":
            action, reason = "RETRIEVE", "high risk: retrieve evidence and constrain generation"
            alts = ["GENERATE_WITH_EVIDENCE", "CLARIFY"]
        else:  # CRITICAL
            # abstention is reserved for the very top band: below
            # min_risk_for_abstain we clarify rather than refuse.
            abstain_at = max(self.min_risk_for_abstain, float(t.get("clarify", 0.75)))
            if prefer_clarify:
                action = "CLARIFY"
                reason = "critical risk: ask clarification before generating"
            elif risk >= abstain_at:
                action = "CLARIFY_OR_ABSTAIN"
                reason = "critical risk: ask clarification or abstain"
            else:
                action = "CLARIFY"
                reason = "critical risk: clarify before proceeding"
            alts = ["RETRIEVE", "GENERATE"]

        dec = Decision(
            action=action,
            risk=risk,
            level=level,
            reason=reason,
            latency_budget_ms=self.latency_budget_ms,
            evidence_available=evidence_available,
            alternatives=alts,
        )
        self.history.append(
            {
                "ts": time.time(),
                "conversation_id": conversation_id,
                "risk": round(risk, 4),
                "level": level,
                "action": action,
                "evidence_available": evidence_available,
                "latency_ms": spent,
            }
        )
        return dec

    def log(self) -> List[Dict[str, Any]]:
        return list(self.history)
