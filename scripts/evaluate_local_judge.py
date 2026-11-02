#!/usr/bin/env python3
"""Zero-shot evaluate an already-installed local Ollama model; no training."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "data/stress/live_context_10k.jsonl"

SYSTEM = """You are an independent pre-answerability auditor. Evaluate only whether the next answer can be supported by the supplied conversation and evidence.
Rules:
- A user statement is valid as user-provided conversational context, but is not independently verified.
- A direct answer in context is safe even if no external search result exists.
- Missing requested facts, conflicting values, or an unresolved reference are risky.
- A context-setting message that asks no question is safe to receive.
- Do not label a prompt risky merely because an external retriever has no result when the conversation itself answers it.
Return ONLY JSON with this schema: {"risky": true or false, "action": "answer" or "retrieve" or "clarify", "confidence": number from 0 to 1}."""


def ask_ollama(case: Dict[str, Any], model: str, endpoint: str, timeout: int) -> Dict[str, Any]:
    user = json.dumps({
        "conversation": case["conversation_history"],
        "evidence": case.get("evidence", []),
        "current_request": case["prompt"],
    }, ensure_ascii=False)
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0, "num_predict": 80},
    }).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/api/chat",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.load(response)
    text = str(body.get("message", {}).get("content", ""))
    result = json.loads(text)
    if not isinstance(result.get("risky"), bool):
        raise ValueError(f"invalid risky value in model response: {text[:200]}")
    if result.get("action") not in {"answer", "retrieve", "clarify"}:
        raise ValueError(f"invalid action in model response: {text[:200]}")
    confidence = float(result.get("confidence", 0.5))
    return {
        "risky": result["risky"],
        "action": result["action"],
        "confidence": max(0.0, min(1.0, confidence)),
        "latency_ms": (time.perf_counter() - started) * 1000,
        "reason": str(result.get("reason", ""))[:300],
    }


def sample_cases(path: Path, per_category: int, seed: int) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        case = json.loads(line)
        buckets[case["category"]].append(case)
    rng = random.Random(seed)
    selected = []
    for category in sorted(buckets):
        bucket = buckets[category]
        rng.shuffle(bucket)
        selected.extend(bucket[:per_category])
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--per-category", type=int, default=100,
                        help="stratified sample count (default: 100; 500 total)")
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "reports" / "local_judge_qwen_500.json")
    args = parser.parse_args()
    cases = sample_cases(args.cases, args.per_category, args.seed)
    rows = []
    failures = []
    started = time.perf_counter()
    for ix, case in enumerate(cases, 1):
        try:
            result = ask_ollama(case, args.model, args.endpoint, args.timeout)
            rows.append({
                "case_id": case["case_id"],
                "category": case["category"],
                "label_risky": bool(case["label_risky"]),
                "prompt": case["prompt"],
                **result,
            })
        except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
            failures.append({"case_id": case["case_id"], "error": str(exc)})
        if ix % 50 == 0:
            print(f"judged {ix:,}/{len(cases):,}", flush=True)

    correct = sum(row["risky"] == row["label_risky"] for row in rows)
    risky = [row for row in rows if row["label_risky"]]
    safe = [row for row in rows if not row["label_risky"]]
    per_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_category[row["category"]].append(row)
    category_stats = {}
    for category, items in sorted(per_category.items()):
        positive = sum(row["risky"] for row in items)
        category_stats[category] = {
            "n": len(items),
            "constructed_label": "risky" if items[0]["label_risky"] else "safe",
            "accuracy": round(sum(row["risky"] == row["label_risky"] for row in items) / len(items), 4),
            "predicted_risky_rate": round(positive / len(items), 4),
            "mean_latency_ms": round(statistics.mean(row["latency_ms"] for row in items), 2),
        }
    report = {
        "evaluation": "zero-shot local pretrained model; no fine-tuning or retraining",
        "synthetic": True,
        "human_labeled": False,
        "model": args.model,
        "seed": args.seed,
        "n_requested": len(cases),
        "n_scored": len(rows),
        "n_failures": len(failures),
        "accuracy_vs_constructed_labels": round(correct / len(rows), 4) if rows else None,
        "safe_false_risk_rate": round(sum(row["risky"] for row in safe) / len(safe), 4) if safe else None,
        "risky_miss_rate": round(sum(not row["risky"] for row in risky) / len(risky), 4) if risky else None,
        "risky_recall": round(sum(row["risky"] for row in risky) / len(risky), 4) if risky else None,
        "mean_latency_ms": round(statistics.mean(row["latency_ms"] for row in rows), 2) if rows else None,
        "p95_latency_ms": round(sorted(row["latency_ms"] for row in rows)[int(.95 * (len(rows)-1))], 2) if rows else None,
        "wall_seconds": round(time.perf_counter() - started, 2),
        "categories": category_stats,
        "limitations": [
            "The model's self-reported confidence is not calibrated and is not used as a probability metric.",
            "Labels are template-construction labels, not human judgments.",
            "A stratified 500-case result does not establish real-world accuracy or safety.",
        ],
        "failures": failures,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "failures", "limitations")}, indent=2))
    print(f"full report -> {args.out}")


if __name__ == "__main__":
    main()
