"""Validation-only model sweep for the leak-controlled benchmark.

This script never reads the test split. It compares model families and RF/XGB
hyperparameters on train/validation, averages over fixed seeds, and records
the selected candidate and all validation metrics. The chosen model is then
evaluated exactly once on the held-out test split by the normal training and
report pipeline.

Run from the repository root:

    python3 scripts/sweep_leakcontrol_models.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from phcx.experiments.matrix import featurize_rows, load_splits
from phcx.experiments.metrics import auprc, auroc, brier_score
from phcx.experiments.train import groups_from_config, train_model


CANDIDATES: List[Tuple[str, str, Dict[str, Any]]] = [
    ("rf_current", "rf", {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5}),
    ("rf_depth_none_leaf2", "rf", {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2}),
    ("rf_depth_none_leaf5", "rf", {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 5}),
    ("rf_depth12_leaf2", "rf", {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 2}),
    ("rf_depth6_leaf2", "rf", {"n_estimators": 500, "max_depth": 6, "min_samples_leaf": 2}),
    ("rf_depth8_leaf2", "rf", {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 2}),
    ("lr_C1", "lr", {"C": 1.0}),
    ("lr_C01", "lr", {"C": 0.1}),
    ("lr_C10", "lr", {"C": 10.0}),
    ("xgb_d3", "xgb", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03}),
    ("xgb_d4", "xgb", {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03}),
    ("lgbm_d3", "lgbm", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03}),
]
SEEDS = (42, 123, 2024)


def main() -> int:
    config_path = Path("configs/train_leakcontrol.yaml")
    cfg = yaml.safe_load(config_path.read_text())
    data_dir = Path((cfg.get("dataset") or {}).get("path", "data/phc_conversation_leakcontrol"))
    splits = load_splits(data_dir)
    groups = groups_from_config(cfg)

    # Featurize once; candidates differ only in the classifier.
    train = featurize_rows(splits["train"], groups=groups, show_progress=True)
    val = featurize_rows(splits["val"], groups=groups, show_progress=True)
    cal_method = (cfg.get("calibration") or {}).get("method", "platt")

    results: List[Dict[str, Any]] = []
    for label, model_name, hyper in CANDIDATES:
        runs = []
        for seed in SEEDS:
            try:
                model, _, _ = train_model(
                    train, val, model_name=model_name, seed=seed,
                    hyper=hyper, calibrate_method=cal_method,
                )
                # Selection uses the raw validation ranking. Calibration and
                # threshold tuning happen later in the normal train pipeline.
                p = model.predict_proba(val.X, calibrated=False)
                runs.append({
                    "seed": seed,
                    "auroc": float(auroc(val.y, p)),
                    "auprc": float(auprc(val.y, p)),
                    "brier_raw": float(brier_score(p, val.y)),
                })
            except Exception as exc:
                print(f"[sweep] {label} seed={seed}: skipped ({exc})")
                runs = []
                break
        if not runs:
            continue
        results.append({
            "name": label,
            "model": model_name,
            "hyperparameters": hyper,
            "validation": {
                key: {
                    "mean": float(np.mean([r[key] for r in runs])),
                    "std": float(np.std([r[key] for r in runs])),
                    "values": [r[key] for r in runs],
                }
                for key in ("auroc", "auprc", "brier_raw")
            },
            "runs": runs,
        })

    results.sort(key=lambda r: r["validation"]["auroc"]["mean"], reverse=True)
    out = {
        "experiment": "leakcontrol_validation_model_sweep",
        "data_dir": str(data_dir),
        "validation_only": True,
        "seeds": list(SEEDS),
        "selection_metric": "mean raw validation AUROC across seeds",
        "test_split_used_for_selection": False,
        "selected": results[0] if results else None,
        "candidates": results,
        "note": (
            "The test split is not read by this script. The selected candidate "
            "is evaluated once by the normal training/report pipeline."
        ),
    }
    path = Path("reports/model_selection_leakcontrol.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("\nvalidation ranking:")
    for r in results:
        m = r["validation"]["auroc"]
        print(f"  {r['name']:24} AUROC {m['mean']:.4f} +/- {m['std']:.4f}")
    print(f"selected: {out['selected']['name'] if out['selected'] else 'none'}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
