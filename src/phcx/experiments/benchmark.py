"""Action-level benchmark: run the trained risk model through the intervention
policy on the test split and report safety / coverage style gates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..intervention import BLOCKING_ACTIONS, INTERVENTION_ACTIONS, RiskPolicy
from .matrix import featurize_rows, load_splits
from .metrics import auroc
from .train import groups_from_config, load_config, train_model

ACTION_ORDER = ["GENERATE", "GENERATE_WITH_EVIDENCE", "RETRIEVE", "CLARIFY",
                "CLARIFY_OR_ABSTAIN", "ABSTAIN"]
MIN_COVERAGE = 0.50  # at least half of test turns must still be answered
MIN_POS_CAUGHT = 0.50  # at least half of true positives must be intercepted


def run_benchmark(
    cfg: Optional[Dict[str, Any]] = None,
    data_dir: str | Path = "data/phc_conversation",
    out_dir: str | Path = "results",
    seed: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    cfg = cfg or load_config("configs/train.yaml")
    seed = int(seed if seed is not None else cfg.get("seed", 42))
    np.random.seed(seed)

    splits = load_splits(data_dir)
    groups = groups_from_config(cfg)
    model_cfg = cfg.get("model", {}) or {}
    hyper = model_cfg.get("hyperparameters", {}) or {}
    cal_method = (cfg.get("calibration", {}) or {}).get("method", "isotonic")

    train = featurize_rows(splits["train"], groups=groups)
    val = featurize_rows(splits.get("val", []), groups=groups) if splits.get("val") else None
    test = featurize_rows(splits["test"], groups=groups, show_progress=show_progress)
    model, _, _ = train_model(train, val, model_name=model_cfg.get("name", "rf"),
                              seed=seed, hyper=hyper, calibrate_method=cal_method)

    # thresholds: cost-sensitive search on val (falls back to default policy)
    policy_path = Path("configs/policy.yaml")
    policy = RiskPolicy.from_yaml(str(policy_path)) if policy_path.exists() else RiskPolicy()
    if val is not None and val.n_rows:
        from ..calibration.core import enforce_min_coverage, optimize_thresholds

        pcfg = cfg.get("policy", {}) or {}
        p_val = model.predict_proba(val.X, calibrated=True)
        search = optimize_thresholds(
            p_val, val.y,
            cost_fp=float(pcfg.get("cost_fp", 1.0)),
            cost_fn=float(pcfg.get("cost_fn", 3.0)),
            abstention_cost=float(pcfg.get("abstention_cost", 2.0)))
        thresholds = dict(search.thresholds)
        min_cov = pcfg.get("min_coverage", None)
        if min_cov is not None:
            # keep at least `min_coverage` of turns answerable by raising the
            # blocking cut-point; the cost objective alone would over-block
            thresholds = enforce_min_coverage(thresholds, p_val, float(min_cov))["thresholds"]
        policy.apply_thresholds(thresholds)
    thr = dict(policy.thresholds)

    p = model.predict_proba(test.X, calibrated=True)
    y = test.y

    # decide() per row — the same policy the pipeline/API uses
    rows_test = splits["test"]
    actions: List[str] = []
    for pi, row in zip(p, rows_test):
        has_evidence = bool(row.get("evidence"))
        dec = policy.decide(float(pi), evidence_available=has_evidence,
                            conversation_id=row.get("family_id", ""))
        actions.append(dec.action)

    buckets: Dict[str, Any] = {}
    for a in ACTION_ORDER:
        mask = np.array([aa == a for aa in actions])
        n = int(mask.sum())
        buckets[a] = {
            "n": n,
            "share": round(n / max(len(actions), 1), 4),
            "hallucination_rate": round(float(y[mask].mean()), 4) if n else None,
        }

    gen_mask = np.array([a == "GENERATE" for a in actions])
    # blocking actions: the turn is not answered (clarify / abstain)
    block_mask = np.array([a in BLOCKING_ACTIONS for a in actions])
    # interception: anything that forces evidence retrieval or clarification
    intervene_mask = np.array([a in INTERVENTION_ACTIONS for a in actions])

    pos_rate_generated = float(y[gen_mask].mean()) if gen_mask.any() else 0.0
    pos_rate_overall = float(y.mean())
    # coverage = fraction of turns still answered (directly or after retrieval)
    coverage = 1.0 - float(block_mask.mean())
    positives_caught = (float(y[intervene_mask].sum() / max(int(y.sum()), 1))
                        if intervene_mask.any() else 0.0)

    gates = {
        # generated cases must be safer than the unfiltered base rate
        "safety": bool(gen_mask.any() and pos_rate_generated < pos_rate_overall),
        "safety_margin": round(pos_rate_overall - pos_rate_generated, 4),
        "coverage": round(coverage, 4),
        "coverage_min": MIN_COVERAGE,
        "coverage_ok": bool(coverage >= MIN_COVERAGE),
        "positives_caught": round(positives_caught, 4),
        "positives_caught_min": MIN_POS_CAUGHT,
        "positives_caught_ok": bool(positives_caught >= MIN_POS_CAUGHT),
        "blocked_share": round(float(block_mask.mean()), 4),
        "risk_generated": round(pos_rate_generated, 4),
        "risk_overall": round(pos_rate_overall, 4),
    }

    report = {
        "experiment": f"benchmark_{cfg.get('experiment', 'phcx')}",
        "model": model_cfg.get("name", "rf"),
        "seed": seed,
        "n_test": int(test.n_rows),
        "label_rate": round(pos_rate_overall, 4),
        "thresholds": {k: round(float(v), 3) for k, v in thr.items()},
        "user_preference": policy.user_preference,
        "action_distribution": buckets,
        "gates": gates,
        "auroc": round(float(auroc(y, p)), 5),
        "gate_status": ("PASS" if (gates["safety"] and gates["coverage_ok"]
                                   and gates["positives_caught_ok"]) else "FAIL"),
    }
    out = Path(out_dir) / "raw"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"benchmark_{cfg.get('experiment', 'phcx')}.json"
    path.write_text(json.dumps(report, indent=2))
    if show_progress:
        print(f"[benchmark] test={report['n_test']} AUROC={report['auroc']:.3f} "
              f"gates={report['gate_status']} -> {path}")
        for a in ACTION_ORDER:
            r = buckets[a]
            if r["n"]:
                print(f"  {a:22s} n={r['n']:4d} share={r['share']:.2%} "
                      f"pos_rate={r['hallucination_rate']}")
        print(f"  coverage={gates['coverage']:.1%} (min {MIN_COVERAGE:.0%}) "
              f"safety_margin={gates['safety_margin']:.3f} "
              f"positives_caught={gates['positives_caught']:.1%}")
    return report
