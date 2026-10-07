"""PHC-X FastAPI service.

Exposes the provenance-aware pre-generation risk pipeline as a local,
offline-first HTTP API used by the dashboard and the browser extension.

Run:  phc serve            (or: uvicorn api.main:app --port 8000)
Env:  PHCX_MODEL=/path/to/checkpoints/<exp>/model   (optional; heuristic fallback)
      PHCX_POLICY=configs/policy.yaml
      PHCX_LOG=data/decision_log.jsonl
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from phcx.intervention import RiskPolicy
from phcx.pipeline import PHCPipeline
from phcx.risk import RiskModel

VERSION = "0.1.0"
LOG_PATH = Path(os.environ.get("PHCX_LOG", "data/decision_log.jsonl"))


# --------------------------------------------------------------------- models
class Turn(BaseModel):
    role: str = "user"
    content: str = ""


class AnalyzeRequest(BaseModel):
    prompt: str
    conversation_history: List[Turn] = Field(default_factory=list)
    evidence: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)
    conversation_id: str = "conv"
    whitebox_signals: Optional[Dict[str, float]] = None
    prefer_clarify: bool = False


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


def _discover_model() -> Optional[str]:
    env = os.environ.get("PHCX_MODEL")
    if env and Path(env).with_suffix(".json").exists():
        return env
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
    policy_path = os.environ.get("PHCX_POLICY", "configs/policy.yaml")
    policy = (RiskPolicy.from_yaml(policy_path)
              if Path(policy_path).exists() else RiskPolicy())
    groups = None if model else ["prompt", "history", "provenance",
                                 "consistency", "evidence"]
    STATE.pipeline = PHCPipeline(model=model, policy=policy, groups=groups)
    STATE.policy = policy
    STATE.model_name = (f"rf:{Path(model_path).parent.name}" if model
                        else "heuristic")
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
