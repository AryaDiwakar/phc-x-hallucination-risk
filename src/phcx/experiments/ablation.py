"""Feature-group ablations (spec: ablating provenance, consistency, evidence...)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..features import ABLATION_GROUPS
from .metrics import classification_metrics, significance_summary
from .train import (
    evaluate,
    groups_from_config,
    load_config,
    train_model,
)
from .matrix import featurize_splits, load_splits


def run_ablations(
    cfg: Optional[Dict[str, Any]] = None,
    data_dir: str | Path = "data/phc_conversation",
    out_dir: str | Path = "results",
    seed: Optional[int] = None,
    ablations: Optional[Dict[str, Sequence[str]]] = None,
    model_name: Optional[str] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    cfg = cfg or load_config("configs/train.yaml")
    seed = int(seed if seed is not None else cfg.get("seed", 42))
    np.random.seed(seed)
    model_cfg = cfg.get("model", {}) or {}
    model_name = model_name or model_cfg.get("name", "rf")
    hyper = model_cfg.get("hyperparameters", {}) or {}
    cal_method = (cfg.get("calibration", {}) or {}).get("method", "isotonic")
    ablations = dict(ablations or ABLATION_GROUPS)

    if show_progress:
        print(f"[ablation] data={data_dir} model={model_name} seed={seed} "
              f"runs={len(ablations)}")
    splits = load_splits(data_dir)
    # featurize once with the union of all groups; ablations are column slices
    full_groups = list(dict.fromkeys(
        g for gs in ablations.values() for g in gs))
    matrices = featurize_splits(splits, groups=full_groups,
                                show_progress=show_progress)
    train, val, test = matrices["train"], matrices.get("val"), matrices["test"]

    runs: Dict[str, Dict[str, Any]] = {}
    preds_test: Dict[str, np.ndarray] = {}
    t_start = time.perf_counter()
    for name, groups in ablations.items():
        sub_tr = train.groups(groups)
        sub_va = val.groups(groups) if val is not None else None
        sub_te = test.groups(groups)
        if sub_tr.X.shape[1] == 0:
            runs[name] = {"error": "no features"}
            continue
        model, _, _ = train_model(sub_tr, sub_va, model_name=model_name, seed=seed,
                                  hyper=hyper, calibrate_method=cal_method)
        thr = 0.5
        p_test = model.predict_proba(sub_te.X, calibrated=True)
        preds_test[name] = p_test
        ev = classification_metrics(sub_te.y, p_test, thr)
        runs[name] = {
            "groups": list(groups),
            "n_features": int(sub_tr.X.shape[1]),
            "test": ev,
        }
        if show_progress:
            print(f"  {name:42s} feats={sub_tr.X.shape[1]:3d} "
                  f"AUROC={ev['auroc']:.3f} AUPRC={ev['auprc']:.3f} "
                  f"F1={ev['f1']:.3f} ECE={ev['ece']:.3f}", flush=True)

    # paired significance of each ablation against the full feature set (F)
    reference = "F_full" if "F_full" in preds_test else next(iter(preds_test), None)
    if reference is not None:
        for name, p in preds_test.items():
            if name == reference:
                continue
            runs[name]["vs_reference"] = significance_summary(
                test.y, p, preds_test[reference],
                name_a=name, name_b=reference, n_boot=500, seed=seed)

    summary = {
        "experiment": f"ablation_{cfg.get('experiment', 'phcx')}",
        "model": model_name,
        "seed": seed,
        "n_rows": {k: m.n_rows for k, m in matrices.items()},
        "reference": reference,
        "elapsed_seconds": round(time.perf_counter() - t_start, 2),
        "runs": runs,
    }
    out_dir = Path(out_dir) / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ablation_{cfg.get('experiment', 'phcx')}.json"
    path.write_text(json.dumps(summary, indent=2))

    if show_progress:
        print(f"[ablation] -> {path}")
        order = sorted(runs.items(),
                       key=lambda kv: -(kv[1].get("test", {}).get("auroc", 0.0)))
        print("  rank  ablation                                  AUROC  AUPRC    F1   ECE")
        for i, (name, r) in enumerate(order, 1):
            t = r.get("test")
            if t:
                print(f"  {i:4d}  {name:40s} {t['auroc']:.3f} {t['auprc']:.3f} "
                      f"{t['f1']:.3f} {t['ece']:.3f}")
    return summary
