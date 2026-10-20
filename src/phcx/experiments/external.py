"""External benchmark adapters and cross-dataset evaluation.

The adapters normalize public datasets into the PHC-X row schema without
changing their labels.  Dataset-specific limitations are carried in the
output manifest so an unlabeled prompt set is never reported as supervised
evidence.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..intervention import BLOCKING_ACTIONS, RiskPolicy
from ..utils.text import tokenize
from .matrix import DatasetMatrix, featurize_rows
from .metrics import auprc, auroc, bootstrap_ci, classification_metrics, per_category_metrics
from .train import load_checkpoint

EXTERNAL_SOURCES = {
    "diahalu": {
        "url": "https://raw.githubusercontent.com/141forever/DiaHalu/main/DiaHalu_Bench.jsonl",
        "license_note": "ACL Anthology materials are CC BY 4.0; repository has no visible dataset license.",
        "label_note": "label=1 means hallucination",
    },
    "halueval_dialogue": {
        "url": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/dialogue_data.json",
        "license_note": "HaluEval repository/data terms must be reviewed before redistribution.",
        "label_note": "paired right_response=0 and hallucinated_response=1",
    },
    "halluhard_seed": {
        "url": "https://huggingface.co/datasets/dyfan/halluhard-seed-questions",
        "license_note": "Hugging Face dataset card declares MIT; source material may have additional terms.",
        "label_note": "no gold hallucination labels in the public seed files",
    },
}


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _parse_diahalu_text(text: str) -> List[Dict[str, str]]:
    pat = re.compile(r"([AB])(\d+)\s*:\s*(.*?)(?=\s+[AB]\d+\s*:|$)", re.S)
    return [{"speaker": m.group(1), "index": m.group(2), "content": " ".join(m.group(3).split())}
            for m in pat.finditer(text)]


def _parse_bracket_history(text: str) -> List[Dict[str, str]]:
    pat = re.compile(r"\[(Human|Assistant)\]\s*:\s*(.*?)(?=\s+\[(?:Human|Assistant)\]\s*:|$)", re.S)
    return [
        {"role": "user" if m.group(1) == "Human" else "assistant",
         "content": " ".join(m.group(2).split())}
        for m in pat.finditer(text)
    ]


def _canonical(
    dataset: str,
    sample_id: str,
    prompt: str,
    history: Sequence[Dict[str, Any]],
    evidence: Sequence[Any],
    answer: str,
    label: Optional[int],
    category: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "family_id": f"{dataset}:{sample_id}",
        "category": category,
        "prompt": prompt,
        "conversation_history": list(history),
        "evidence": list(evidence),
        "answer": answer,
        "label": label,
        "metadata": metadata or {},
    }


def load_diahalu(path: str | Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Normalize DiaHalu's A1/B1/... JSONL records."""
    out: List[Dict[str, Any]] = []
    for raw in _read_jsonl(path)[:limit]:
        turns = _parse_diahalu_text(str(raw.get("text", "")))
        a_turns = [t for t in turns if t["speaker"] == "A"]
        b_turns = [t for t in turns if t["speaker"] == "B"]
        if not a_turns or not b_turns:
            continue
        prompt = a_turns[-1]["content"]
        last_a_index = int(a_turns[-1]["index"])
        history = []
        for t in turns:
            if t["speaker"] == "A" and int(t["index"]) >= last_a_index:
                break
            history.append({"role": "user" if t["speaker"] == "A" else "assistant",
                            "content": t["content"]})
        answer = next((t["content"] for t in b_turns if int(t["index"]) == last_a_index),
                      b_turns[-1]["content"])
        out.append(_canonical(
            "diahalu", str(raw.get("ID", len(out))), prompt, history, [], answer,
            int(raw.get("label", 0)), str(raw.get("domain", "unknown")),
            {"source": raw.get("source"), "index": raw.get("index"),
             "explanation": raw.get("explaination"), "llm": raw.get("Which LLM")},
        ))
    return out


def load_halueval_dialogue(path: str | Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Normalize HaluEval dialogue into paired correct/hallucinated rows."""
    raw_rows = _read_jsonl(path)
    out: List[Dict[str, Any]] = []
    for i, raw in enumerate(raw_rows[:limit]):
        turns = _parse_bracket_history(str(raw.get("dialogue_history", "")))
        if not turns:
            continue
        prompt_index = max((j for j, t in enumerate(turns) if t["role"] == "user"), default=-1)
        if prompt_index < 0:
            continue
        prompt = turns[prompt_index]["content"]
        history = turns[:prompt_index]
        evidence = [{"doc_id": f"halueval-{i}", "text": str(raw.get("knowledge", "")),
                     "source": "HaluEval", "reliability": 0.8}]
        common = {"source": "HaluEval", "pair_id": i}
        out.append(_canonical("halueval_dialogue", f"{i}:right", prompt, history, evidence,
                              str(raw.get("right_response", "")), 0, "supported_response", common))
        out.append(_canonical("halueval_dialogue", f"{i}:hallucinated", prompt, history, evidence,
                              str(raw.get("hallucinated_response", "")), 1,
                              "hallucinated_response", common))
    return out


def load_halluhard_seed(path: str | Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load HalluHard seed prompts as unlabeled stress rows."""
    p = Path(path)
    if p.suffix == ".jsonl":
        raw_rows = _read_jsonl(p)
    else:
        raw = json.loads(p.read_text(encoding="utf-8"))
        raw_rows = raw if isinstance(raw, list) else raw.get("data", [])
    out = []
    for i, raw in enumerate(raw_rows[:limit]):
        prompt = raw.get("prompt") or raw.get("question") or raw.get("research_question") or ""
        if not prompt:
            continue
        evidence_text = raw.get("guideline_text") or raw.get("abstract") or ""
        evidence = ([{"doc_id": f"halluhard-{i}", "text": evidence_text,
                      "source": raw.get("source", "HalluHard")}]
                    if evidence_text else [])
        out.append(_canonical("halluhard_seed", str(i), str(prompt), [], evidence, "", None,
                              str(raw.get("task", raw.get("question_category", "unknown"))),
                              {k: v for k, v in raw.items() if k not in {"prompt", "question", "research_question"}}))
    return out


def load_external_rows(name: str, path: str | Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if name == "diahalu":
        return load_diahalu(path, limit)
    if name == "halueval_dialogue":
        return load_halueval_dialogue(path, limit)
    if name == "halluhard_seed":
        return load_halluhard_seed(path, limit)
    raise ValueError(f"unknown external dataset: {name}")


def _simple_baselines(rows: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    lengths = np.asarray([len(tokenize(str(r.get("prompt", "")))) for r in rows], dtype=float)
    # Transparent uncertainty baselines; these are not claimed to be SOTA.
    length = np.clip((lengths - 6.0) / 14.0, 0.0, 1.0)
    no_evidence = np.asarray([1.0 if not r.get("evidence") else 0.0 for r in rows])
    return {"prompt_length": length, "evidence_absence": no_evidence}


def _metrics(y: np.ndarray, scores: np.ndarray, threshold: float, n_boot: int,
             seed: int) -> Dict[str, Any]:
    return {
        "metrics": classification_metrics(y, scores, threshold),
        "auroc_ci": bootstrap_ci(y, scores, auroc, n_boot=n_boot, seed=seed),
        "auprc_ci": bootstrap_ci(y, scores, auprc, n_boot=n_boot, seed=seed + 1),
    }


def intervention_outcomes(
    rows: Sequence[Dict[str, Any]], scores: np.ndarray, policy: RiskPolicy,
    threshold: float,
) -> Dict[str, Any]:
    decisions = [policy.decide(float(s), bool(r.get("evidence")), conversation_id=str(r.get("sample_id")))
                 for r, s in zip(rows, scores)]
    labels = np.asarray([r.get("label") for r in rows])
    labeled = np.asarray([x is not None for x in labels])
    y = labels[labeled].astype(int)
    actions = [d.action for d in decisions]
    blocking = np.asarray([a in BLOCKING_ACTIONS for a in actions])
    nonblocking = ~blocking
    result: Dict[str, Any] = {
        "n": len(rows),
        "action_distribution": {a: actions.count(a) for a in sorted(set(actions))},
        "blocking_share": round(float(blocking.mean()), 5) if len(actions) else 0.0,
        "coverage": round(float(nonblocking.mean()), 5) if len(actions) else 0.0,
        "abstention_burden": round(float(blocking.mean()), 5) if len(actions) else 0.0,
        "latency_budget_ms": policy.latency_budget_ms,
        "answer_quality": {"status": "not_measured", "reason": "no external generation/evaluator run"},
        "generation_cost": {"status": "not_measured", "reason": "token pricing is provider-specific"},
        "evaluation_type": "oracle-label replay; no generated answer quality measured",
    }
    if labeled.any():
        generated_y = y[nonblocking[labeled]]
        result.update({
            "base_hallucination_rate": round(float(y.mean()), 5),
            "nonblocking_hallucination_rate": round(float(generated_y.mean()), 5)
            if generated_y.size else None,
            "hallucination_reduction": round(float(y.mean() - generated_y.mean()), 5)
            if generated_y.size else None,
            "positives_caught": round(float(y[blocking[labeled]].sum() / max(1, y.sum())), 5),
        })
    return result


def evaluate_external(
    checkpoint: str | Path,
    datasets: Dict[str, str | Path],
    out_path: str | Path = "results/raw/external_eval.json",
    limit: Optional[int] = None,
    n_boot: int = 1000,
    seed: int = 42,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Evaluate a synthetic-trained checkpoint without external retraining."""
    model, meta = load_checkpoint(checkpoint)
    groups = meta.get("groups")
    threshold = float(meta.get("threshold", 0.5))
    policy = RiskPolicy()
    result: Dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "training_domain": "PHC-Conversation synthetic",
        "threshold": threshold,
        "datasets": {},
        "sources": EXTERNAL_SOURCES,
        "limitations": [
            "External models are not retrained or calibrated on external labels.",
            "HalluHard seed prompts are coverage-only because public seeds have no gold labels.",
            "Answer quality and monetary token cost require an LLM generation run and are not inferred from labels.",
        ],
    }
    for name, path in datasets.items():
        rows = load_external_rows(name, path, limit=limit)
        if not rows:
            result["datasets"][name] = {"status": "empty", "path": str(path)}
            continue
        if show_progress:
            print(f"[external] {name}: normalizing {len(rows)} rows", flush=True)
        t0 = time.perf_counter()
        matrix: DatasetMatrix = featurize_rows(rows, groups=groups, show_progress=show_progress)
        p = model.predict_proba(matrix.X, calibrated=True)
        raw = model.predict_proba(matrix.X, calibrated=False)
        unlabeled = any(r.get("label") is None for r in rows)
        item: Dict[str, Any] = {
            "status": "coverage_only" if unlabeled else "scored",
            "path": str(path),
            "n_rows": len(rows),
            "n_labeled": int(sum(r.get("label") is not None for r in rows)),
            "label_rate": (round(float(np.mean([r["label"] for r in rows if r.get("label") is not None])), 5)
                           if not unlabeled else None),
            "feature_seconds": round(time.perf_counter() - t0, 3),
            "history_turns": {
                "min": min(len(r.get("conversation_history") or []) for r in rows),
                "median": float(np.median([len(r.get("conversation_history") or []) for r in rows])),
                "max": max(len(r.get("conversation_history") or []) for r in rows),
            },
            "action_replay": intervention_outcomes(rows, p, policy, threshold),
        }
        if not unlabeled:
            y = np.asarray([int(r["label"]) for r in rows])
            item["model"] = _metrics(y, p, threshold, n_boot, seed)
            item["model_uncalibrated"] = _metrics(y, raw, threshold, n_boot, seed + 2)
            item["heuristic"] = _metrics(y, matrix.heuristic, threshold, n_boot, seed + 4)
            for baseline, scores in _simple_baselines(rows).items():
                item[baseline] = _metrics(y, scores, threshold, n_boot, seed + 6)
            item["per_category"] = per_category_metrics(y, p, matrix.categories, threshold)
            item["length_buckets"] = {}
            lengths = np.asarray([len(r.get("conversation_history") or []) for r in rows])
            for name_bucket, mask in (("0", lengths == 0), ("1-3", (lengths >= 1) & (lengths <= 3)),
                                      ("4+", lengths >= 4)):
                if mask.any() and len(set(y[mask].tolist())) > 1:
                    item["length_buckets"][name_bucket] = classification_metrics(y[mask], p[mask], threshold)
        else:
            item["unlabeled_note"] = EXTERNAL_SOURCES.get(name, {}).get("label_note")
        result["datasets"][name] = item

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    if show_progress:
        print(f"[external] -> {out}")
    return result
