"""Family-grouped, training-only model sweep for the leak-controlled set.

Candidate selection uses five-fold GroupKFold by ``family_id`` within the
training split, repeated across fixed seeds. Neither validation nor test rows
are used to rank classifiers. The ordinary training pipeline subsequently
fits calibration/thresholds on validation and evaluates the selected model
once on the held-out test split.

Run from the repository root:

    python3 scripts/sweep_leakcontrol_models.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from sklearn.model_selection import GroupKFold

from phcx.experiments.matrix import featurize_rows, load_splits
from phcx.experiments.metrics import auprc, auroc
from phcx.risk import RiskModel
from phcx.experiments.train import groups_from_config

CANDIDATES: List[Tuple[str, str, Dict[str, Any]]] = [
    ("rf_current", "rf", {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5}),
    ("rf_depth8_leaf2", "rf", {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 2}),
    ("rf_unlimited_leaf2", "rf", {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2}),
    ("xgb_d2", "xgb", {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.03}),
    ("xgb_d3", "xgb", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03}),
    ("xgb_d4", "xgb", {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03}),
    ("xgb_d3_slow", "xgb", {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.01}),
    ("lgbm_d2", "lgbm", {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.03}),
    ("lgbm_d3", "lgbm", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03}),
    ("lgbm_d4", "lgbm", {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03}),
]
SEEDS = (42, 123, 2024)
N_FOLDS = 5


def main() -> int:
    cfg = yaml.safe_load(Path("configs/train_leakcontrol.yaml").read_text())
    data_dir = Path((cfg.get("dataset") or {}).get("path", "data/phc_conversation_leakcontrol"))
    splits = load_splits(data_dir)
    groups = groups_from_config(cfg)

    # Only the training split enters model selection. Each validation fold is
    # grouped by family so sibling examples cannot leak across train/valid.
    train = featurize_rows(splits["train"], groups=groups, show_progress=True)
    family_ids = np.asarray(train.family_ids)
    if len(family_ids) != train.n_rows:
        raise ValueError("training matrix has no aligned family_ids")
    cv = GroupKFold(n_splits=N_FOLDS)

    results: List[Dict[str, Any]] = []
    for label, model_name, hyper in CANDIDATES:
        runs: List[Dict[str, Any]] = []
        for seed in SEEDS:
            for fold, (train_idx, valid_idx) in enumerate(
                cv.split(train.X, train.y, groups=family_ids)
            ):
                try:
                    model = RiskModel(name=model_name, seed=seed, **hyper)
                    model.fit(train.X[train_idx], train.y[train_idx], train.feature_names)
                    scores = model.predict_proba(train.X[valid_idx], calibrated=False)
                    runs.append({
                        "seed": seed,
                        "fold": fold,
                        "n_valid": int(len(valid_idx)),
                        "n_families_valid": int(len(set(family_ids[valid_idx]))),
                        "auroc": float(auroc(train.y[valid_idx], scores)),
                        "auprc": float(auprc(train.y[valid_idx], scores)),
                    })
                except Exception as exc:
                    print(f"[sweep] {label} seed={seed} fold={fold}: skipped ({exc})")
                    runs = []
                    break
            if not runs:
                break
        if not runs:
            continue
        aurocs = [r["auroc"] for r in runs]
        auprcs = [r["auprc"] for r in runs]
        results.append({
            "name": label,
            "model": model_name,
            "hyperparameters": hyper,
            "cv": {
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_std": float(np.std(aurocs)),
                "auprc_mean": float(np.mean(auprcs)),
                "auprc_std": float(np.std(auprcs)),
            },
            "runs": runs,
        })

    results.sort(key=lambda r: r["cv"]["auroc_mean"], reverse=True)
    report = {
        "experiment": "leakcontrol_grouped_cv_model_sweep",
        "data_dir": str(data_dir),
        "selection_split": "train only",
        "cv": f"{N_FOLDS}-fold GroupKFold by family_id",
        "seeds": list(SEEDS),
        "selection_metric": "mean raw AUROC over all grouped validation folds and seeds",
        "validation_split_used_for_selection": False,
        "test_split_used_for_selection": False,
        "selected": results[0] if results else None,
        "candidates": results,
        "note": (
            "Neither validation nor test split participates in model selection. "
            "The selected candidate is calibrated on validation and evaluated "
            "once on test by the standard pipeline."
        ),
    }
    out = Path("reports/model_selection_leakcontrol.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("\nfamily-grouped training-CV ranking:")
    for candidate in results:
        cv = candidate["cv"]
        print(f"  {candidate['name']:24} AUROC {cv['auroc_mean']:.4f} "
              f"+/- {cv['auroc_std']:.4f}  AUPRC {cv['auprc_mean']:.4f}")
    print(f"selected: {report['selected']['name'] if report['selected'] else 'none'}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
