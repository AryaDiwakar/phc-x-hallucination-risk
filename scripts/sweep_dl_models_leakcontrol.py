#!/usr/bin/env python3
"""Training-family-only CV comparison of MLP/neural and tree baselines.

Selection uses only the leak-controlled training split with GroupKFold by
family_id. Validation is reserved for calibration, and test is not accessed by
this script. This produces a reproducible model-selection artifact; it does
not make synthetic labels evidence of real-world performance.
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
from phcx.experiments.train import groups_from_config
from phcx.risk import RiskModel

SEEDS = (42, 123, 2024)
FOLDS = 5
CANDIDATES: List[Tuple[str, str, Dict[str, Any]]] = [
    ("rf_current", "rf", {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5}),
    ("xgb_depth4", "xgb", {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03}),
    ("sklearn_mlp_64_32", "mlp", {
        "hidden_layer_sizes": (64, 32), "alpha": 1e-3,
        "lr": 1e-3, "max_iter": 400,
    }),
    ("torch_mlp_64x3", "neural", {
        "hidden": 64, "layers": 3, "dropout": 0.2,
        "lr": 1e-3, "epochs": 120, "batch_size": 64,
    }),
]


def main() -> int:
    config_path = Path("configs/train_leakcontrol.yaml")
    cfg = yaml.safe_load(config_path.read_text())
    data_dir = Path((cfg.get("dataset") or {}).get("path", "data/phc_conversation_leakcontrol"))
    splits = load_splits(data_dir)
    groups = groups_from_config(cfg)
    train = featurize_rows(splits["train"], groups=groups, show_progress=True)
    family_ids = np.asarray(train.family_ids)
    if len(family_ids) != train.n_rows:
        raise ValueError("training matrix has no aligned family_ids")
    cv = GroupKFold(n_splits=FOLDS)

    results: List[Dict[str, Any]] = []
    for label, model_name, hyper in CANDIDATES:
        runs = []
        for seed in SEEDS:
            for fold, (fit_idx, valid_idx) in enumerate(
                cv.split(train.X, train.y, groups=family_ids)
            ):
                model = RiskModel(name=model_name, seed=seed, **hyper)
                model.fit(train.X[fit_idx], train.y[fit_idx], train.feature_names)
                scores = model.predict_proba(train.X[valid_idx], calibrated=False)
                runs.append({
                    "seed": seed,
                    "fold": fold,
                    "n_valid": int(len(valid_idx)),
                    "n_families_valid": len(set(family_ids[valid_idx])),
                    "auroc": float(auroc(train.y[valid_idx], scores)),
                    "auprc": float(auprc(train.y[valid_idx], scores)),
                })
        results.append({
            "name": label,
            "model": model_name,
            "hyperparameters": hyper,
            "cv": {
                "auroc_mean": float(np.mean([r["auroc"] for r in runs])),
                "auroc_std": float(np.std([r["auroc"] for r in runs])),
                "auprc_mean": float(np.mean([r["auprc"] for r in runs])),
                "auprc_std": float(np.std([r["auprc"] for r in runs])),
            },
            "runs": runs,
        })
    results.sort(key=lambda r: r["cv"]["auroc_mean"], reverse=True)
    report = {
        "experiment": "leakcontrol_training_only_dl_model_sweep",
        "synthetic": True,
        "data_dir": str(data_dir),
        "selection_split": "training split only",
        "cv": f"{FOLDS}-fold GroupKFold by family_id",
        "seeds": list(SEEDS),
        "validation_used_for_selection": False,
        "test_used_for_selection": False,
        "selected": results[0],
        "candidates": results,
        "note": (
            "Model selection is limited to a synthetic leak-controlled benchmark. "
            "The result is not a claim of human or real-world generalization."
        ),
    }
    output = Path("reports/model_selection_dl_leakcontrol.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("training-family CV (AUROC):")
    for row in results:
        cv_result = row["cv"]
        print(f"  {row['name']:20} {cv_result['auroc_mean']:.4f} +/- {cv_result['auroc_std']:.4f}")
    print(f"selected: {report['selected']['name']}; wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
