"""PHC-X FastAPI service.

Exposes the provenance-aware pre-generation risk pipeline as a local,
offline-first HTTP API used by the dashboard and the browser extension.

Run:  phc serve            (or: uvicorn api.main:app --port 8000)
Env:  PHCX_MODEL=auto | heuristic | /path/to/checkpoints/<exp>/model
      PHCX_POLICY=configs/policy.yaml
      PHCX_LOG=data/decision_log.jsonl
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from phcx.intervention import RiskPolicy
from phcx.models import OllamaAdapter
from phcx.pipeline import PHCPipeline
from phcx.risk import RiskModel

VERSION = "0.2.0"
LOG_PATH = Path(os.environ.get("PHCX_LOG", "data/decision_log.jsonl"))


# --------------------------------------------------------------------- models
class Turn(BaseModel):
    role: str = "user"
    content: str = ""
    #: Provenance of this turn.  Without these the memory cannot tell a
    #: source-verified statement from one the model asserted itself, and the
    #: contamination score degenerates.
    source: Optional[str] = None
    verified: bool = False
    retrieved: bool = False


class AnalyzeRequest(BaseModel):
    prompt: str
    conversation_history: List[Turn] = Field(default_factory=list)
    evidence: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)
    conversation_id: str = "conv"
    whitebox_signals: Optional[Dict[str, float]] = None
    prefer_clarify: bool = False


class EnhanceRequest(AnalyzeRequest):
    """Request for a local, evidence-aware prompt rewrite suggestion."""


class BatchRequest(BaseModel):
    items: List[AnalyzeRequest]


class FeedbackRequest(BaseModel):
    conversation_id: str = "conv"
    request_id: Optional[str] = None
    risk: Optional[float] = None
    action: Optional[str] = None
    label: Optional[int] = None  # 1 = actually hallucinated, 0 = safe
    note: str = ""


class PolicyUpdate(BaseModel):
    thresholds: Optional[Dict[str, float]] = None
    user_preference: Optional[str] = None
    min_risk_for_abstain: Optional[float] = None
    latency_budget_ms: Optional[float] = None


# ---------------------------------------------------------------------- state
class State:
    pipeline: Optional[PHCPipeline] = None
    policy: RiskPolicy = RiskPolicy()
    model_name: str = "heuristic"
    model_path: Optional[str] = None
    n_requests: int = 0
    n_errors: int = 0
    started_at: float = time.time()
    action_counts: Dict[str, int] = {}


STATE = State()
_ENHANCER_CHECKED = False
_ENHANCER: Optional[OllamaAdapter] = None


def _local_enhancer() -> Optional[OllamaAdapter]:
    """Cache an optional local Ollama rewrite model; never call a hosted API."""
    global _ENHANCER_CHECKED, _ENHANCER
    if _ENHANCER_CHECKED:
        return _ENHANCER
    _ENHANCER_CHECKED = True
    # Local LLM wording is opt-in: the small available model occasionally
    # emits irrelevant instructions. Deterministic, source-aware rewrites are
    # the dependable default.
    model = os.environ.get("PHCX_ENHANCER_MODEL", "off").strip()
    if not model or model.lower() in {"off", "none", "disabled"}:
        return None
    candidate = OllamaAdapter(model=model, timeout=45.0)
    try:
        if candidate.available():
            _ENHANCER = candidate
    except Exception:
        _ENHANCER = None
    return _ENHANCER


def _validated_local_instruction(diagnosis: Dict[str, Any]) -> Optional[str]:
    """Ask local Ollama for a short response constraint, never a factual rewrite."""
    enhancer = _local_enhancer()
    if enhancer is None:
        return None
    issue = (
        "directly answerable from user-provided context" if diagnosis.get("direct_user_context_support", {}).get("supported")
        else "evidence appears missing" if diagnosis.get("risk_reasons")
        else "general factual request"
    )
    instruction = (
        "Write one short imperative instruction (maximum 25 words) to help an assistant "
        "answer a factual request safely. Do not answer the request. Do not include any "
        "specific facts, entities, dates, numbers, or a rewritten question. Case: "
        f"{issue}. Return only JSON: {{\"instruction\": \"...\"}}."
    )
    try:
        raw = enhancer.generate(instruction, max_tokens=80)
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return None
        text = json.loads(match.group(0)).get("instruction")
        if not isinstance(text, str) or not text.strip() or len(text) > 300:
            return None
        if re.search(r"\d|\b[A-Z][a-z]+\b", text[1:]):
            return None
        if not set(text.lower().split()) & {
            "evidence", "context", "source", "clarification", "uncertain", "guess",
            "support", "supported", "verifiable", "speculation", "assumptions",
        }:
            return None
        return text.strip()
    except Exception:
        # A rewrite failure must not affect analysis or prevent normal sending.
        return None


def _discover_model() -> Optional[str]:
    # Use a small heuristic/RF ensemble by default. `PHCX_MODEL=heuristic`
    # selects the transparent baseline; `auto` opts into RF-only; an explicit
    # model path selects one checkpoint.
    env = os.environ.get("PHCX_MODEL", "ensemble")
    if env.lower() in {"heuristic", "none", "off"}:
        return None
    if env and Path(env).with_suffix(".json").exists():
        return env
    if env.lower() not in {"auto", "discover", "ensemble"}:
        return None
    root = Path("checkpoints")
    if root.exists():
        preferred = ["phcx_train_baseline_seed42", "phcx_train_baseline",
                     "phcx_rf_seed42"]
        for name in preferred:
            cand = root / name / "model"
            if cand.with_suffix(".json").exists():
                return str(cand)
        for cand in sorted(root.glob("*/model.json")):
            return str(cand.with_suffix(""))
    return None


def build_pipeline() -> None:
    model_path = _discover_model()
    model = RiskModel.load(model_path) if model_path else None
    model_blend = 0.15 if os.environ.get("PHCX_MODEL", "ensemble").lower() == "ensemble" else 0.0
    policy_path = os.environ.get("PHCX_POLICY", "configs/policy.yaml")
    policy = (RiskPolicy.from_yaml(policy_path)
              if Path(policy_path).exists() else RiskPolicy())
    groups = None if model else ["prompt", "history", "provenance",
                                 "consistency", "evidence"]
    STATE.pipeline = PHCPipeline(
        model=model, policy=policy, groups=groups, model_blend=model_blend
    )
    STATE.policy = policy
    STATE.model_name = (
        f"heuristic+{model.name}({model_blend:.2f}):{Path(model_path).parent.name}"
        if model and model_blend
        else f"{model.name}:{Path(model_path).parent.name}" if model
        else "heuristic"
    )
    STATE.model_path = model_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_pipeline()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # pre-warm: first request pays RF importance / SHAP tree setup otherwise
    if STATE.pipeline is not None:
        try:
            STATE.pipeline.analyze("warmup prompt: what is the current status?",
                                   conversation_history=[], evidence=[],
                                   use_retriever=False)
        except Exception:
            STATE.n_errors += 1
    yield


app = FastAPI(
    title="PHC-X",
    description="Provenance-aware pre-generation hallucination risk prediction "
                "and intervention",
    version=VERSION,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # local research tool
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _log(entry: Dict[str, Any]) -> None:
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        STATE.n_errors += 1


# ------------------------------------------------------------------- endpoints
if Path("dashboard").exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/dashboard",
              StaticFiles(directory="dashboard", html=True),
              name="dashboard")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": VERSION,
        "model": STATE.model_name,
        "model_path": STATE.model_path,
        "uptime_s": round(time.time() - STATE.started_at, 1),
        "n_requests": STATE.n_requests,
        "n_errors": STATE.n_errors,
        "decision_log": str(LOG_PATH),
        "log_entries": _log_entries() and len(_log_entries()) or 0,
    }


@app.get("/v1/version")
def version() -> Dict[str, Any]:
    return {"name": "phcx", "version": VERSION, "model": STATE.model_name}


@app.post("/v1/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    if STATE.pipeline is None:
        raise HTTPException(503, "pipeline not initialised")
    t0 = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    try:
        pred = STATE.pipeline.analyze(
            prompt=req.prompt,
            conversation_history=[t.model_dump() for t in req.conversation_history],
            evidence=req.evidence,
            whitebox_signals=req.whitebox_signals,
            conversation_id=req.conversation_id,
            use_retriever=not req.evidence,
            prefer_clarify=req.prefer_clarify,
        )
    except Exception as exc:  # pragma: no cover - defensive
        STATE.n_errors += 1
        raise HTTPException(500, f"analysis failed: {exc}") from exc

    payload = pred.to_dict()
    payload["request_id"] = request_id
    payload["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    payload["model"] = STATE.model_name

    action = payload.get("decision", {}).get("action", "UNKNOWN")
    STATE.n_requests += 1
    STATE.action_counts[action] = STATE.action_counts.get(action, 0) + 1
    _log({
        "ts": time.time(), "request_id": request_id,
        "conversation_id": req.conversation_id, "prompt": req.prompt[:500],
        "risk": payload.get("risk_score"), "level": payload.get("risk_level"),
        "action": action, "model": STATE.model_name,
        "latency_ms": payload["latency_ms"],
        "n_history": len(req.conversation_history),
        "n_evidence": len(req.evidence),
    })
    return payload


@app.post("/v1/enhance")
def enhance_prompt(req: EnhanceRequest) -> Dict[str, Any]:
    """Build a grounded rewrite; optionally use a local Ollama model as editor."""
    if STATE.pipeline is None:
        raise HTTPException(503, "pipeline not initialised")
    history = [turn.model_dump() for turn in req.conversation_history]
    common = dict(
        conversation_history=history,
        evidence=req.evidence,
        conversation_id=req.conversation_id,
        use_retriever=not req.evidence,
        prefer_clarify=req.prefer_clarify,
    )
    original = STATE.pipeline.analyze(req.prompt, **common)
    source = original.context_support.get("source_excerpt")
    arithmetic = STATE.pipeline._is_self_contained_arithmetic(req.prompt)
    conflicts = []
    for pair in original.contradictions[:3]:
        left, right = pair.get("a", {}).get("text"), pair.get("b", {}).get("text")
        for statement in (left, right):
            if statement and statement not in conflicts:
                conflicts.append(statement)
    context_conflict = original.context_support.get("context_conflict")
    if context_conflict:
        turn_index = int(context_conflict.get("turn_id", 0)) - 1
        prior_text = history[turn_index].get("content", "") if 0 <= turn_index < len(history) else ""
        for statement in (
            f"Earlier user note: {prior_text}" if prior_text else "",
            f"New user note: {req.prompt}",
        ):
            if statement and statement not in conflicts:
                conflicts.append(statement)
    ambiguity_refs = original.ambiguities[:3]
    diagnosis = {
        "direct_user_context_support": original.context_support,
        "conflicting_statements": conflicts,
        "ambiguous_references": ambiguity_refs,
        "risk_reasons": original.engine_reasons,
    }
    if arithmetic:
        deterministic = (
            f"Question: {req.prompt}\n\n"
            "This is a self-contained arithmetic calculation. Compute it directly, "
            "check the arithmetic, and give the concise result. Do not add unrelated facts."
        )
        method = "verified self-contained arithmetic rewrite"
    elif source:
        deterministic = (
            "The following is user-provided context, not independently verified:\n"
            f'"{source}"\n\n'
            f"Question: {req.prompt}\n\n"
            "Answer only if that context supports the answer. Cite the exact detail. "
            "If it is missing or conflicting, say so and ask one focused question. "
            "Do not add or infer facts. Format as Answer / Supporting context / Limits."
        )
        method = "source-attributed deterministic rewrite"
    elif conflicts:
        quoted = "\n".join(f'- "{text}"' for text in conflicts)
        deterministic = (
            f"Question: {req.prompt}\n\n"
            "The conversation contains conflicting statements:\n"
            f"{quoted}\n"
            "Do not choose one as true. Ask which statement or source to use, then "
            "answer only from the clarification and available evidence. Do not invent "
            "a resolution or a value."
        )
        method = "conflict-aware deterministic rewrite"
    elif ambiguity_refs:
        refs = []
        for ref in ambiguity_refs:
            name = ref.get("reference", "")
            candidates = ref.get("candidates", [])
            descriptions = [c.get("text") or c.get("entity") for c in candidates]
            refs.append(f"{name}: " + " | ".join(str(d) for d in descriptions if d))
        deterministic = (
            f"Question: {req.prompt}\n\n"
            "This request may refer to more than one item in the conversation: "
            + "; ".join(refs)
            + ". Ask which item is meant before giving a factual answer; do not guess."
        )
        method = "ambiguity-aware deterministic rewrite"
    else:
        deterministic = (
            f"Question: {req.prompt}\n\n"
            "Use only information explicitly present in the conversation or supplied "
            "evidence. Cite the specific support. If the answer, source, or referent is "
            "missing or conflicting, state what is unclear and ask one focused "
            "clarification instead of guessing. Format as Answer / Supporting context / Limits."
        )
        method = "evidence-constrained deterministic rewrite"

    candidates = [(deterministic, method)]
    # Do not delegate ambiguous or contradictory cases to a small editor model:
    # its best-looking paraphrase could silently choose one interpretation.
    local_instruction = None
    if not conflicts and not ambiguity_refs and not arithmetic:
        local_instruction = _validated_local_instruction(diagnosis)
    enhancer = _local_enhancer() if local_instruction else None
    if local_instruction and enhancer:
        candidates.append((
            deterministic + "\n\nAdditional local response instruction: " + local_instruction,
            f"{method} + local {enhancer.model} instruction",
        ))

    scored = []
    for candidate, candidate_method in candidates:
        result = STATE.pipeline.analyze(candidate, **common)
        scored.append((result.risk_score, candidate, result, candidate_method))
    after_risk, rewritten, after, selected_method = min(
        scored,
        key=lambda item: (item[0], 0 if " + local " in item[3] else 1),
    )
    return {
        "enhanced_prompt": rewritten,
        "original_risk": round(original.risk_score, 4),
        "estimated_risk": round(after_risk, 4),
        "risk_reduced": after_risk < original.risk_score,
        "original_action": original.recommended_action,
        "estimated_action": after.recommended_action,
        "context_support": original.context_support,
        "enhancement_method": selected_method,
        "local_rewrite_available": bool(enhancer),
        "note": (
            "Review the draft before sending. The original request is preserved verbatim; "
            "local-model instructions are checked for added facts/numbers but can still be wrong. "
            "A lower PHC-X estimate does not prove the answer is correct."
        ),
    }


@app.post("/v1/batch")
def analyze_batch(req: BatchRequest) -> Dict[str, Any]:
    out = [analyze(item) for item in req.items]
    return {"count": len(out), "results": out}


@app.get("/v1/policy")
def get_policy() -> Dict[str, Any]:
    p = STATE.policy
    return {
        "thresholds": p.thresholds,
        "user_preference": p.user_preference,
        "min_risk_for_abstain": p.min_risk_for_abstain,
        "latency_budget_ms": p.latency_budget_ms,
        "model": STATE.model_name,
    }


@app.put("/v1/policy")
def update_policy(update: PolicyUpdate) -> Dict[str, Any]:
    p = STATE.policy
    if update.thresholds:
        p.apply_thresholds(update.thresholds)
    if update.user_preference:
        if update.user_preference not in ("balanced", "aggressive", "permissive"):
            raise HTTPException(422, "user_preference must be balanced|aggressive|permissive")
        p.user_preference = update.user_preference
    if update.min_risk_for_abstain is not None:
        p.min_risk_for_abstain = float(update.min_risk_for_abstain)
    if update.latency_budget_ms is not None:
        p.latency_budget_ms = float(update.latency_budget_ms)
    if STATE.pipeline is not None:
        STATE.pipeline.policy = p
    return get_policy()


@app.post("/v1/feedback")
def feedback(req: FeedbackRequest) -> Dict[str, Any]:
    entry = {"ts": time.time(), **req.model_dump()}
    _log(entry)
    return {"logged": True, "entry": entry}


@app.get("/v1/metrics")
def metrics() -> Dict[str, Any]:
    entries = _log_entries()
    actions: Dict[str, int] = {}
    labeled = [e for e in entries if e.get("label") is not None]
    risks: List[float] = []
    for e in entries:
        if "action" in e and e.get("risk") is not None:
            actions[e["action"]] = actions.get(e["action"], 0) + 1
            try:
                risks.append(float(e["risk"]))
            except (TypeError, ValueError):
                pass
    out: Dict[str, Any] = {
        "n_logged": len(entries),
        "n_requests": STATE.n_requests,
        "action_distribution": actions or STATE.action_counts,
        "mean_risk": round(sum(risks) / len(risks), 4) if risks else None,
        "labeled": len(labeled),
    }
    if labeled:
        import numpy as np

        y = np.array([int(e["label"]) for e in labeled])
        out["label_rate"] = round(float(y.mean()), 4)
        flagged = [e for e in labeled if (e.get("risk") or 0) >= STATE.policy.thresholds.get("moderate", 0.5)]
        if flagged:
            out["pos_rate_flagged"] = round(
                float(np.mean([int(e["label"]) for e in flagged])), 4)
    # surface the latest benchmark gates for the dashboard
    if RESULTS_DIR.exists():
        bench = sorted(RESULTS_DIR.glob("benchmark_*.json"))
        if bench:
            try:
                out["benchmark_gates"] = json.loads(bench[-1].read_text()).get("gates")
            except (OSError, json.JSONDecodeError):
                pass
    return out


RESULTS_DIR = Path(os.environ.get("PHCX_RESULTS", "results/raw"))


@app.get("/v1/results")
def list_results() -> Dict[str, Any]:
    if not RESULTS_DIR.exists():
        return {"dir": str(RESULTS_DIR), "files": []}
    files = sorted(p.name for p in RESULTS_DIR.glob("*.json"))
    return {"dir": str(RESULTS_DIR), "files": files}


@app.get("/v1/results/{name}")
def get_result(name: str) -> Dict[str, Any]:
    safe = Path(name).name
    if not safe.endswith(".json"):
        safe += ".json"
    path = (RESULTS_DIR / safe).resolve()
    if not path.exists() or RESULTS_DIR.resolve() != path.parent:
        raise HTTPException(404, f"result '{safe}' not found")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"corrupt result file: {exc}") from exc


@app.post("/v1/reset")
def reset_log() -> Dict[str, Any]:
    """Clear the decision log (research convenience; local tool only)."""
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    STATE.n_requests = 0
    STATE.n_errors = 0
    STATE.action_counts = {}
    return {"reset": True}


def _log_entries() -> List[Dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return []
    return out
