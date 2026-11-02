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

import ast
import math
import operator
import re
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
from ..utils.text import content_tokens

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
    model_risk_score: float
    risk_level: str
    recommended_action: str
    base_risk_score: float = 0.0
    risk_method: str = "heuristic"
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
    context_support: Dict[str, Any] = field(default_factory=dict)
    risk_adjustment: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 4),
            "model_risk_score": round(self.model_risk_score, 4),
            "base_risk_score": round(self.base_risk_score, 4),
            "risk_method": self.risk_method,
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
            "context_support": self.context_support,
            "risk_adjustment": self.risk_adjustment,
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
        model_blend: float = 0.0,
    ):
        self.model = model
        self.policy = policy or RiskPolicy()
        self.retriever = retriever or LocalRetriever()
        self.mode = mode
        self.adapter = adapter
        self.history_turns = history_turns
        self.model_blend = float(max(0.0, min(1.0, model_blend)))
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
        # The history feature block analyses a recent-turn window; the full
        # conversation stays visible to the consistency engines (cross-turn
        # contradiction / drift is exactly what they are meant to find) and to
        # the size statistics.  ``history_turns <= 0`` disables the window.
        mem_window = mem.recent(self.history_turns)

        # 2) contamination (uses FULL memory, never a truncated view)
        contam = mem.contamination(prompt)

        # 3) consistency engines (always over the full conversation)
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
            window=mem_window,
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
        risk_method = "heuristic"
        if self.model is not None and len(x):
            model_risk = float(self.model.predict_proba(x.reshape(1, -1), calibrated=True)[0])
            calibrated = getattr(self.model, "calibrator", None) is not None
            risk = model_risk
            risk_method = "trained_model"
            if self.model_blend > 0.0:
                transparent_risk = heuristic_risk(feats)
                risk = (self.model_blend * model_risk
                        + (1.0 - self.model_blend) * transparent_risk)
                risk_method = f"heuristic_rf_ensemble_{self.model_blend:.2f}"
        else:
            model_risk = heuristic_risk(feats)
            risk = model_risk

        # A synthetic-data classifier can over-penalize ordinary follow-ups.
        # When a question is directly answerable from an earlier user turn,
        # and no consistency engine found a conflict, don't let that alone
        # trigger a blocking action. The source remains USER_ASSERTED—not
        # externally verified—and the raw model score stays visible.
        context_support = self._direct_user_context_support(prompt, mem)
        model_risk = float(model_risk)
        base_risk = risk
        risk_adjustment = ""
        context_only = self._is_context_setting_turn(prompt)
        benign_social = self._is_benign_social_turn(prompt)
        arithmetic = self._is_self_contained_arithmetic(prompt)
        context_conflict = self._context_setting_conflict(prompt, mem) if context_only else None
        if context_conflict:
            context_support = {**context_support, "context_conflict": context_conflict}
        if arithmetic:
            risk = 0.0
            risk_adjustment = "self-contained arithmetic expression"
        elif (
            context_support.get("supported")
            and contradiction.score < 0.35
            and temporal.score < 0.35
            and not presupposition.unsupported_count
        ):
            risk = min(risk, 0.24)
            risk_adjustment = "directly answerable from user-provided context"
        elif context_only and not context_conflict:
            risk = min(risk, 0.24)
            risk_adjustment = "context-setting turn does not request a factual answer"
        elif context_conflict:
            # A context note that changes an already-stated number/date is not
            # benign setup: surface the disagreement instead of auto-passing it.
            risk = max(risk, 0.75)
            risk_adjustment = "conflicting numeric context; clarify which note is authoritative"
        elif benign_social:
            # Greetings and acknowledgements do not request factual content;
            # empty retrieval evidence must not turn them into high-risk tasks.
            risk = 0.0
            risk_adjustment = "benign greeting or acknowledgement; no factual answer requested"

        # 8) explanation
        engine_reasons = self._engine_reasons(contam, contradiction, entity, temporal,
                                              presupposition, ambiguity, ev)
        explanation = self.explainer.explain(
            x, {**feats, "_risk": risk}, engine_reasons=engine_reasons
        )
        explanation.risk = risk

        # 9) policy decision
        # The latency budget is enforced against the time already spent on
        # this analysis: if we are already over budget, the policy downgrades
        # to a cheaper action instead of blocking the user.
        spent_ms = (time.perf_counter() - t0) * 1000.0
        dec = self.policy.decide(
            risk,
            evidence_available=ev.available,
            latency_ms=spent_ms,
            prefer_clarify=(prefer_clarify or ambiguity.should_clarify or bool(context_conflict)) and not (
                (context_only and not context_conflict) or benign_social or arithmetic
            ),
            conversation_id=conversation_id,
        )

        # 10) provenance warnings (never blind-trust model memory)
        warnings = self._provenance_warnings(mem, contam)

        latency = (time.perf_counter() - t0) * 1000.0
        return RiskPrediction(
            risk_score=risk,
            model_risk_score=model_risk,
            base_risk_score=base_risk,
            risk_method=risk_method,
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
            context_support=context_support,
            risk_adjustment=risk_adjustment,
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

    @staticmethod
    def _direct_user_context_support(prompt: str, memory: ConversationMemory) -> Dict[str, Any]:
        """Find a high-overlap user turn that directly covers a factual query.

        This is intentionally lexical and auditable: it never promotes the
        user assertion to verified fact, and only suppresses a hard block when
        the requested answer type is visibly present in the matching turn.
        """
        q = (prompt or "").lower()
        # The prompt enhancer preserves the original request in a labeled
        # question block. Score that request, not its added safety instructions.
        question_block = re.search(r"\bquestion\s*:\s*([^\n]+)", q, re.I)
        if question_block:
            q = question_block.group(1)
        if "?" not in q:
            return {"supported": False, "coverage": 0.0}
        prompt_tokens = set(content_tokens(q))
        ask_words = {
            "many", "much", "number", "amount", "exact", "please", "tell",
            "remind", "took", "part", "give", "can", "you", "what's", "what’s",
            "year", "date", "month",
        }
        participant_aliases = {
            "participant": "participants", "participants": "participants",
            "people": "participants", "person": "participants",
            "individuals": "participants", "enrollees": "participants",
            "enrollment": "participants", "count": "participants",
            "sample": "participants", "size": "participants",
            "run": "conducted", "runs": "conducted", "ran": "conducted",
            "held": "conducted", "occurred": "conducted",
        }
        canonical = lambda words: {participant_aliases.get(word, word) for word in words}
        anchors = canonical(prompt_tokens - ask_words)
        if len(anchors) < 2:
            return {"supported": False, "coverage": 0.0}

        numeric_question = bool(re.search(
            r"\b(how many|how much|what number|exact number|sample size|"
            r"participant count|enrollment count|how big|what(?:'s|’s| is| was) "
            r"(?:the )?(?:exact )?number)\b", q
        ))
        year_question = bool(re.search(r"\b(which year|what year)\b", q))
        time_question = year_question or bool(re.search(r"\b(when|what date|what month)\b", q))
        if not (numeric_question or time_question):
            return {"supported": False, "coverage": 0.0}
        best: Tuple[float, Optional[Turn]] = (0.0, None)
        matched_numeric_turns: List[Tuple[float, Turn, set]] = []
        for turn in memory.turns:
            if turn.speaker != "user":
                continue
            words = canonical(set(content_tokens(turn.text)))
            overlap = len(anchors & words)
            coverage = overlap / max(1, len(anchors))
            has_number = bool(re.search(r"\b\d[\d,]*(?:\.\d+)?%?\b", turn.text))
            years = set(re.findall(r"\b(?:19|20)\d{2}\b", turn.text))
            months = set(re.findall(
                r"\b(january|february|march|april|may|june|july|august|september|"
                r"october|november|december|today|yesterday|tomorrow)\b", turn.text, re.I
            ))
            has_date = bool(years if year_question else (years or months))
            if numeric_question and not has_number:
                continue
            if time_question and not has_date:
                continue
            numbers = set(re.findall(r"\b\d[\d,]*(?:\.\d+)?%?\b", turn.text))
            if numeric_question and coverage >= 0.67 and overlap >= 2:
                matched_numeric_turns.append((coverage, turn, numbers))
            if time_question and coverage >= 0.67 and overlap >= 2:
                matched_numeric_turns.append((coverage, turn, years if year_question else years | months))
            if coverage > best[0]:
                best = (coverage, turn)
        coverage, turn = best
        supported = bool(
            turn and coverage >= 0.67
            and len(anchors & canonical(set(content_tokens(turn.text)))) >= 2
        )
        if (numeric_question or time_question) and len(matched_numeric_turns) > 1:
            # Conflicting user-provided numeric answers are not a safe basis
            # for the grounded-history override.
            number_sets = {tuple(sorted(item[2])) for item in matched_numeric_turns}
            if len(number_sets) > 1:
                supported = False
        if not supported or turn is None:
            return {"supported": False, "coverage": round(coverage, 4)}
        return {
            "supported": True,
            "coverage": round(coverage, 4),
            "source_role": "user",
            "source_turn_id": turn.turn_id,
            "source_excerpt": turn.text[:500],
            "independently_verified": False,
        }

    @staticmethod
    def _is_context_setting_turn(prompt: str) -> bool:
        """Recognize narrow context-only messages that ask for no answer yet."""
        text = (prompt or "").strip().lower()
        if not text or "?" in text:
            return False
        if not re.match(
            r"^(?:for this chat|for this conversation|context\s*:|the notes say|"
            r"the notes state|remember that|keep in mind|fyi\b)", text
        ):
            return False
        return not bool(re.search(
            r"\b(answer|explain|summari[sz]e|compare|calculate|recommend|tell me|"
            r"find|write|generate|analy[sz]e)\b", text
        ))

    @staticmethod
    def _context_setting_conflict(prompt: str, memory: ConversationMemory) -> Optional[Dict[str, Any]]:
        """Detect changed numeric facts for the same named entity in a new note."""
        def named_anchors(text: str) -> set:
            names = set(re.findall(r"\b[A-Z][a-z0-9-]{2,}\b", text or ""))
            return {name.lower() for name in names} - {
                "for", "the", "no", "may", "january", "february", "march",
                "april", "june", "july", "august", "september", "october",
                "november", "december", "outcome", "statistical", "tables",
            }

        def values(text: str) -> set:
            all_values = set(re.findall(r"\b\d[\d,]*(?:\.\d+)?%?\b", text or ""))
            return {value for value in all_values
                    if not re.fullmatch(r"(?:19|20)\d{2}", value.replace(",", ""))}

        current_entities = named_anchors(prompt)
        current_values = values(prompt)
        if not current_entities or not current_values:
            return None
        for turn in memory.turns:
            # Compare newly supplied context only against prior user notes;
            # an assistant paraphrase may omit fields without contradicting them.
            if turn.speaker != "user":
                continue
            shared_entities = current_entities & named_anchors(turn.text)
            previous_values = values(turn.text)
            if shared_entities and previous_values and current_values != previous_values:
                return {
                    "entity": sorted(shared_entities)[0],
                    "previous_values": sorted(previous_values),
                    "new_values": sorted(current_values),
                    "turn_id": turn.turn_id,
                }
        return None

    @staticmethod
    def _is_benign_social_turn(prompt: str) -> bool:
        """Recognize exact greetings/acknowledgements, not mixed factual asks."""
        text = re.sub(r"[^a-z0-9' ]+", " ", (prompt or "").lower())
        text = re.sub(r"\s+", " ", text).strip()
        patterns = (
            r"h+i+", r"he+y+", r"hello+", r"yo+", r"sup(?: there)?",
            r"good (?:morning|afternoon|evening)",
            r"how are you(?: doing)?", r"thanks?", r"thank you(?: so much)?",
            r"thx", r"ty", r"ok(?:ay)?", r"got it", r"cool",
            r"sounds good", r"you are welcome", r"you're welcome", r"no problem",
        )
        return any(re.fullmatch(pattern, text) for pattern in patterns)

    @staticmethod
    def _is_self_contained_arithmetic(prompt: str) -> bool:
        """Recognize simple numeric arithmetic without executing arbitrary code."""
        text = (prompt or "").strip().lower()
        if text.startswith("question:"):
            text = text[len("question:"):].splitlines()[0].strip()
        for prefix in ("what is ", "what's ", "calculate ", "compute ", "solve "):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        text = text.rstrip(" ?=.")
        if len(text) > 80 or not re.search(r"\d\s*[+*/%\-]\s*\d", text):
            return False
        try:
            tree = ast.parse(text, mode="eval")
        except (SyntaxError, ValueError):
            return False
        allowed = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
            ast.Pow, ast.UAdd, ast.USub,
        )
        nodes = list(ast.walk(tree))
        if len(nodes) > 24 or any(not isinstance(node, allowed) for node in nodes):
            return False
        if not any(isinstance(node, ast.BinOp) for node in nodes):
            return False

        def calculate(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return calculate(node.body)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                value = float(node.value)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                inner = calculate(node.operand)
                value = inner if isinstance(node.op, ast.UAdd) else -inner
            elif isinstance(node, ast.BinOp):
                left, right = calculate(node.left), calculate(node.right)
                operations = {
                    ast.Add: operator.add, ast.Sub: operator.sub,
                    ast.Mult: operator.mul, ast.Div: operator.truediv,
                    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
                    ast.Pow: operator.pow,
                }
                if isinstance(node.op, ast.Pow) and abs(right) > 8:
                    raise ValueError("exponent limit")
                value = operations[type(node.op)](left, right)
            else:
                raise ValueError("not arithmetic")
            if not math.isfinite(value) or abs(value) > 1e12:
                raise ValueError("result limit")
            return value

        try:
            calculate(tree)
            return True
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
            return False


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
