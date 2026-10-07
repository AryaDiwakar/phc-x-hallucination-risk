"""Optional local-LLM baselines.

These are deliberately named lexical approximations.  Official SelfCheckGPT
and semantic-entropy implementations use stronger similarity/NLI components;
this module never labels a lexical fallback as the official method.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import requests

from ..utils.text import similarity, tokenize
from .external import _metrics


def ollama_generate(
    prompt: str,
    model: str = "llama3.2:3b",
    temperature: float = 0.8,
    seed: int = 42,
    timeout: int = 120,
    endpoint: str = "http://127.0.0.1:11434/api/generate",
) -> str:
    response = requests.post(
        endpoint,
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": temperature, "seed": seed}},
        timeout=timeout,
    )
    response.raise_for_status()
    return str(response.json().get("response", ""))


def _context(row: Dict[str, Any]) -> str:
    history = "\n".join(
        f"{t.get('role', 'user')}: {t.get('content', '')}"
        for t in row.get("conversation_history", [])
    )
    evidence = "\n".join(
        str(e.get("text", e)) if isinstance(e, dict) else str(e)
        for e in row.get("evidence", [])
    )
    return (f"Conversation:\n{history}\nEvidence:\n{evidence}\n"
            f"User: {row.get('prompt', '')}\nAssistant:")


def selfcheck_lexical_score(candidate: str, samples: Sequence[str]) -> float:
    """Black-box consistency risk using lexical similarity as a fallback."""
    if not samples:
        return 0.5
    return float(np.clip(1.0 - np.mean([similarity(candidate, s) for s in samples]), 0.0, 1.0))


def semantic_entropy_lexical_score(samples: Sequence[str], threshold: float = 0.55) -> float:
    """Normalized entropy over greedy lexical-semantic response clusters."""
    if not samples:
        return 0.5
    clusters: List[List[str]] = []
    for sample in samples:
        for cluster in clusters:
            if similarity(sample, cluster[0]) >= threshold:
                cluster.append(sample)
                break
        else:
            clusters.append([sample])
    probs = np.asarray([len(c) / len(samples) for c in clusters], dtype=float)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return float(np.clip(entropy / max(math.log(len(samples)), 1.0), 0.0, 1.0))


def evaluate_local_baselines(
    rows: Sequence[Dict[str, Any]],
    model_names: Sequence[str] = ("llama3.2:3b", "qwen2.5:3b"),
    sample_count: int = 3,
    limit: int = 50,
    seed: int = 42,
    threshold: float = 0.5,
    out_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Run lexical SelfCheck/entropy approximations on a small local sample."""
    selected = list(rows[:limit])
    result: Dict[str, Any] = {
        "status": "started",
        "evaluation_type": "local Ollama generation with lexical approximations",
        "official_method_note": (
            "SelfCheckGPT and semantic entropy are represented by lexical fallbacks; "
            "these numbers must not be cited as official implementations."
        ),
        "sample_count": sample_count,
        "rows": len(selected),
        "models": {},
        "post_generation_baselines": {
            "cove": {"status": "not_comparable", "reason": "verification after drafting, not pre-generation risk"},
            "rarr": {"status": "not_comparable", "reason": "retrieval/revision after generation, not pre-generation risk"},
        },
    }
    for model_name in model_names:
        started = time.perf_counter()
        scores_self, scores_entropy, labels = [], [], []
        failures = 0
        for i, row in enumerate(selected):
            if row.get("label") is None:
                continue
            try:
                samples = [ollama_generate(_context(row), model=model_name,
                                            temperature=0.8, seed=seed + i * 31 + k)
                           for k in range(sample_count)]
                candidate = str(row.get("answer", ""))
                scores_self.append(selfcheck_lexical_score(candidate, samples))
                scores_entropy.append(semantic_entropy_lexical_score(samples))
                labels.append(int(row["label"]))
            except Exception:
                failures += 1
        item: Dict[str, Any] = {
            "n_scored": len(labels),
            "failures": failures,
            "seconds": round(time.perf_counter() - started, 3),
        }
        if labels and len(set(labels)) > 1:
            y = np.asarray(labels, dtype=int)
            item["selfcheck_lexical"] = _metrics(y, np.asarray(scores_self), threshold, 500, seed)
            item["semantic_entropy_lexical"] = _metrics(y, np.asarray(scores_entropy), threshold, 500, seed + 1)
        else:
            item["status"] = "insufficient_labeled_class_variation"
        result["models"][model_name] = item
    result["status"] = "complete"
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, default=str))
    return result
