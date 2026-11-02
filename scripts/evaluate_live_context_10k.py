#!/usr/bin/env python3
"""Generate and evaluate a 10k synthetic, human-style ChatGPT context suite.

Labels describe the construction (answerable/context-setting vs.
unsupported/conflicting/ambiguous), not human judgments.  The script uses the
same locally discovered model and policy as ``phc serve``; only SHAP rendering
is skipped because explanations do not affect model scores or decisions.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from api.main import STATE, build_pipeline  # noqa: E402
from phcx.explain.core import Explanation  # noqa: E402
from phcx.pipeline.risk_pipeline import PHCPipeline  # noqa: E402


SUBJECTS = [
    "Northstar pilot", "Cedar cohort", "Harbor study", "Juniper trial",
    "Orion rollout", "Maple program", "Lumen evaluation", "Brookfield survey",
    "Meridian experiment", "Redwood initiative", "Summit field test",
    "Willow research project", "Atlas onboarding study", "Vantage pilot",
    "Aurora user trial", "Pioneer cohort", "Cobalt assessment",
    "Silverline study", "Bluebird experiment", "Stonebridge pilot",
]
MONTHS = ["January", "March", "May", "July", "September", "November"]
CONTEXT_STYLES = [
    "For this chat, the {subject} notes say {n} people took part in {month} {year}; no outcome data was supplied.",
    "For this conversation: {subject} — {n} participants, run in {month} {year}. The notes don't include results.",
    "Context: the {subject} included {n} participants and ran in {month} {year}; outcomes aren't listed.",
    "The notes say the {subject} had {n} participants in {month} {year}. No results were provided.",
    "The notes state that {n} people joined the {subject}, held in {month} {year}; outcome measures are missing.",
    "Remember that the {subject} involved {n} participants in {month} {year}. No outcome metrics are in the notes.",
    "Keep in mind: {subject} had {n} participants, and took place in {month} {year}. The outcome data isn't available.",
    "FYI, the {subject} ran in {month} {year} with {n} participants; the notes don't report an outcome.",
]
FOLLOWUP_PROMPTS = [
    "How many participants were in the {subject}?",
    "What's the number of participants in the {subject}?",
    "What was the exact number of people in the {subject}?",
    "Can you remind me how many people took part in the {subject}?",
]
UNSUPPORTED_HISTORY = [
    "We're reviewing the {subject}. The available notes describe the protocol but don't state the enrollment count.",
    "The {subject} report covers the method and schedule; it doesn't give a participant total.",
    "I found notes about the {subject}, but no sample-size figure is included.",
    "The {subject} materials explain the procedure. The number of participants isn't recorded here.",
]
UNSUPPORTED_PROMPTS = [
    "How many participants were in the {subject}?",
    "What's the exact sample size for the {subject}?",
    "Can you give me the participant count for the {subject}?",
    "What number of people took part in the {subject}?",
]
AMBIGUOUS_PROMPTS = [
    "How many participants did it have?",
    "What was its sample size?",
    "Can you remind me how many people were in it?",
    "How many participants were there?",
]


def make_cases(per_category: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    cases: List[Dict[str, Any]] = []
    categories = (
        "supported_followup", "context_setting", "missing_fact",
        "conflicting_numbers", "ambiguous_reference",
    )
    for category in categories:
        for i in range(per_category):
            subject = rng.choice(SUBJECTS)
            other = rng.choice([s for s in SUBJECTS if s != subject])
            n = rng.randint(24, 2400)
            n2 = n + rng.choice([-31, -13, 17, 29, 43])
            month, year = rng.choice(MONTHS), rng.randint(2021, 2026)
            if category == "supported_followup":
                history = [
                    {"role": "user", "content": (
                        f"The {subject} had {n} participants and ran in {month} {year}."
                    )},
                    {"role": "assistant", "content": (
                        f"Got it. The {subject} notes show:\nParticipants: {n}\n"
                        f"Period: {month} {year}"
                    )},
                ]
                prompt = rng.choice(FOLLOWUP_PROMPTS).format(subject=subject)
                label = 0
            elif category == "context_setting":
                history = []
                prompt = rng.choice(CONTEXT_STYLES).format(
                    subject=subject, n=n, month=month, year=year
                )
                label = 0
            elif category == "missing_fact":
                history = [{"role": "user", "content": rng.choice(UNSUPPORTED_HISTORY).format(subject=subject)}]
                prompt = rng.choice(UNSUPPORTED_PROMPTS).format(subject=subject)
                label = 1
            elif category == "conflicting_numbers":
                history = [
                    {"role": "user", "content": f"The {subject} included {n} participants."},
                    {"role": "assistant", "content": f"I have {n} participants noted for the {subject}."},
                    {"role": "user", "content": (
                        f"Correction: the {subject} had {n2} participants, not {n}."
                    )},
                ]
                prompt = rng.choice(FOLLOWUP_PROMPTS).format(subject=subject)
                label = 1
            else:
                history = [
                    {"role": "user", "content": f"The {subject} involved {n} participants."},
                    {"role": "assistant", "content": f"The {subject} had {n} people."},
                    {"role": "user", "content": f"The {other} included {n2} participants."},
                    {"role": "assistant", "content": f"The {other} had {n2} people."},
                ]
                prompt = rng.choice(AMBIGUOUS_PROMPTS)
                label = 1
            cases.append({
                "case_id": f"{category}-{i:05d}",
                "category": category,
                "label_risky": label,
                "prompt": prompt,
                "conversation_history": history,
                "evidence": [],
                "conversation_id": f"stress-{category}-{i}",
            })
    rng.shuffle(cases)
    return cases


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def evaluate(cases: Iterable[Dict[str, Any]], seed: int, model_mode: str) -> Dict[str, Any]:
    if model_mode == "auto":
        os.environ["PHCX_MODEL"] = "auto"
    elif model_mode == "ensemble":
        os.environ["PHCX_MODEL"] = "ensemble"
    else:
        os.environ["PHCX_MODEL"] = "heuristic"
    build_pipeline()
    pipeline = STATE.pipeline
    if pipeline is None:
        raise RuntimeError("PHC-X pipeline did not initialize")
    if model_mode == "heuristic":
        pipeline = PHCPipeline(
            model=None,
            policy=pipeline.policy,
            groups=["prompt", "history", "provenance", "consistency", "evidence"],
        )
        STATE.pipeline = pipeline
        STATE.model_name = "heuristic"
    model_name = STATE.model_name
    # Explanation/SHAP is presentation-only. Avoid recomputing SHAP for every
    # row while exercising the exact same feature, model, calibration, and
    # policy paths used for the final decisions.
    pipeline.explainer.explain = lambda _x, features, engine_reasons=None: Explanation(
        risk=float(features.get("_risk", 0.0)),
        engine_reasons=list(engine_reasons or []),
        method="stress-test-no-shap",
    )

    nonblocking_actions = {"GENERATE", "GENERATE_WITH_EVIDENCE"}
    blocking_actions = {"CLARIFY", "CLARIFY_OR_ABSTAIN", "ABSTAIN"}
    rows: List[Dict[str, Any]] = []
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    started = time.perf_counter()
    for ix, case in enumerate(cases, 1):
        pred = pipeline.analyze(
            prompt=case["prompt"],
            conversation_history=case["conversation_history"],
            evidence=case["evidence"],
            conversation_id=case["conversation_id"],
            use_retriever=True,
        )
        row = {
            "case_id": case["case_id"],
            "category": case["category"],
            "label_risky": case["label_risky"],
            "prompt": case["prompt"],
            "risk_score": pred.risk_score,
            "base_risk_score": pred.base_risk_score,
            "model_risk_score": pred.model_risk_score,
            "risk_method": pred.risk_method,
            "action": pred.recommended_action,
            "risk_adjustment": pred.risk_adjustment,
            "context_supported": bool(pred.context_support.get("supported")),
            "contamination_score": pred.contamination.get("score", 0.0),
            "ambiguity_score": pred.features.get("cons_reference_ambiguity", 0.0),
            "contradiction_score": pred.features.get("cons_contradiction_score", 0.0),
            "latency_ms": pred.latency_ms,
        }
        rows.append(row)
        by_category[case["category"]].append(row)
        if ix % 1000 == 0:
            print(f"scored {ix:,}/{len(cases):,}", flush=True)

    labels = [int(r["label_risky"]) for r in rows]
    scores = [float(r["risk_score"]) for r in rows]
    base_scores = [float(r["base_risk_score"]) for r in rows]
    model_scores = [float(r["model_risk_score"]) for r in rows]
    actions = Counter(r["action"] for r in rows)
    safe_rows = [r for r in rows if not r["label_risky"]]
    risky_rows = [r for r in rows if r["label_risky"]]
    category_report = {}
    for category, items in sorted(by_category.items()):
        risky = bool(items[0]["label_risky"])
        category_report[category] = {
            "n": len(items),
            "label": "constructed_risky" if risky else "constructed_safe",
            "mean_effective_risk": round(sum(r["risk_score"] for r in items) / len(items), 4),
            "mean_raw_model_risk": round(sum(r["model_risk_score"] for r in items) / len(items), 4),
            "nonblocking_rate": _rate(sum(r["action"] in nonblocking_actions for r in items), len(items)),
            "hard_block_rate": _rate(sum(r["action"] in blocking_actions for r in items), len(items)),
            "intervention_rate": _rate(sum(r["action"] != "GENERATE" for r in items), len(items)),
            "grounded_override_rate": _rate(sum(bool(r["risk_adjustment"]) for r in items), len(items)),
            "actions": dict(Counter(r["action"] for r in items)),
        }
    elapsed = time.perf_counter() - started
    try:
        effective_auc = float(roc_auc_score(labels, scores))
        base_auc = float(roc_auc_score(labels, base_scores))
        raw_auc = float(roc_auc_score(labels, model_scores))
    except ValueError:
        effective_auc = base_auc = raw_auc = float("nan")
    return {
        "synthetic": True,
        "human_labeled": False,
        "suite": "PHC-X generated live-context stress suite",
        "seed": seed,
        "n": len(rows),
        "model": model_name,
        "model_mode": model_mode,
        "total_seconds": round(elapsed, 2),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / len(rows), 3),
        "p95_latency_ms": round(sorted(r["latency_ms"] for r in rows)[int(0.95 * (len(rows) - 1))], 3),
        "effective_risk_auroc_vs_constructed_labels": round(effective_auc, 4),
        "base_pipeline_auroc_vs_constructed_labels": round(base_auc, 4),
        "raw_model_auroc_vs_constructed_labels": round(raw_auc, 4),
        "safe_case_nonblocking_rate": _rate(sum(r["action"] in nonblocking_actions for r in safe_rows), len(safe_rows)),
        "safe_case_hard_block_rate": _rate(sum(r["action"] in blocking_actions for r in safe_rows), len(safe_rows)),
        "risky_case_intervention_rate": _rate(sum(r["action"] != "GENERATE" for r in risky_rows), len(risky_rows)),
        "risky_case_no_intervention_rate": _rate(sum(r["action"] == "GENERATE" for r in risky_rows), len(risky_rows)),
        "risky_case_hard_block_rate": _rate(sum(r["action"] in blocking_actions for r in risky_rows), len(risky_rows)),
        "action_distribution": dict(actions),
        "categories": category_report,
        "limitations": [
            "Generated from templates; not human-authored or human-labeled.",
            "Measures these five constructed scenarios, not real-world hallucination prevention.",
            "AUROC is against construction labels and must not be compared with human evaluation.",
        ],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=2000,
                        help="cases per scenario (default: 2,000; total 10,000)")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--model-mode", choices=("auto", "heuristic", "ensemble"), default="auto",
                        help="trained RF, transparent heuristic, or 85/15 heuristic-RF blend")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "live_context_10k.json")
    parser.add_argument("--cases-out", type=Path,
                        default=ROOT / "data" / "stress" / "live_context_10k.jsonl")
    args = parser.parse_args()
    if args.per_category < 1:
        parser.error("--per-category must be positive")

    cases = make_cases(args.per_category, args.seed)
    args.cases_out.parent.mkdir(parents=True, exist_ok=True)
    with args.cases_out.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"generated {len(cases):,} cases -> {args.cases_out}", flush=True)

    report = evaluate(cases, args.seed, args.model_mode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    summary = {k: v for k, v in report.items() if k not in ("rows", "categories", "limitations")}
    summary["categories"] = report["categories"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"full report -> {args.out}")


if __name__ == "__main__":
    main()
