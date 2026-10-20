"""Stress tests for the PHC-X risk model.

Two questions the headline test split cannot answer on its own:

**Does the model detect hallucination, or does it detect our own label
source?**  PHC-Conversation carries two independent labels: ``label_gold``
(by construction) and ``label_verified`` (from the post-hoc verifier).  They
agree on only ~0.69 of the rows, so they behave like two different
annotators.  Training on one and evaluating against the other measures
whether the model learned a property of the *conversation* or the quirks of
the label source it was fitted on.

**Does the model detect risk types it has never seen?**  Leave-one-category-
out trains on ten risk categories and tests on the eleventh, which is the
only way to claim the framework generalises across risk types rather than
memorising ten of them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..calibration.core import make_calibrator
from .matrix import DatasetMatrix, featurize_splits, load_splits
from .metrics import auroc, classification_metrics
from .train import groups_from_config, load_config


def _labels(rows: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([int(r.get(key, 0) or 0) for r in rows], dtype=int)


def run_cross_label_generalization(
    cfg: Dict[str, Any],
    data_dir: str | Path = "data/phc_conversation",
    out_path: Optional[str | Path] = None,
    seed: int = 42,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Train on one label source, evaluate against the other.

    The features are identical in both directions -- only the labels change,
    so this isolates the label source.
    """
    from .train import train_model

    groups = groups_from_config(cfg)
    cal_method = (cfg.get("calibration", {}) or {}).get("method", "platt")
    splits = load_splits(data_dir)
    matrices = featurize_splits(splits, groups=groups, show_progress=show_progress)
    train, val, test = matrices.get("train"), matrices.get("val"), matrices.get("test")
    if train is None or test is None:
        raise ValueError("train and test splits are required")

    sources = [("label_gold", "label_verified"), ("label_verified", "label_gold")]
    results: Dict[str, Any] = {}
    for train_key, test_key in sources:
        y_train = _labels(splits["train"], train_key)
        y_val = _labels(splits["val"], train_key)
        y_test = _labels(splits["test"], test_key)

        model = train_model(
            train.with_labels(y_train),
            val.with_labels(y_val) if val is not None else None,
            (cfg.get("model", {}) or {}).get("name", "rf"),
            seed,
            (cfg.get("model", {}) or {}).get("hyperparameters", {}) or {},
            cal_method,
        )[0]
        p_test = model.predict_proba(test.X, calibrated=True)
        m = classification_metrics(y_test, p_test, 0.5)
        # the same predictions scored against the *training* label source,
        # for reference
        m_train_label = classification_metrics(
            _labels(splits["test"], train_key), p_test, 0.5)
        results[f"train_{train_key}__test_{test_key}"] = {
            "n": int(len(y_test)),
            "train_label_rate": round(float(y_train.mean()), 5),
            "test_label_rate": round(float(y_test.mean()), 5),
            "auroc_vs_other_source": round(float(m["auroc"]), 5),
            "auroc_vs_own_source": round(float(m_train_label["auroc"]), 5),
            "auprc_vs_other_source": round(float(m["auprc"]), 5),
            "f1_vs_other_source": round(float(m["f1"]), 5),
        }
        if show_progress:
            print(f"[cross-label] train={train_key} -> test={test_key}: "
                  f"AUROC vs other source {m['auroc']:.4f} "
                  f"(vs own source {m_train_label['auroc']:.4f})")

    out: Dict[str, Any] = {
        "experiment": "cross_label_generalization",
        "data_dir": str(data_dir),
        "seed": seed,
        "results": results,
        "note": (
            "training on one label source and scoring against the other: a "
            "model that learned the conversation rather than the label source "
            "keeps most of its AUROC"
        ),
    }
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
    return out


def run_leave_one_category_out(
    cfg: Dict[str, Any],
    data_dir: str | Path = "data/phc_conversation",
    out_path: Optional[str | Path] = None,
    seed: int = 42,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Train on every category but one; test on the held-out one.

    This is the strongest available generalisation claim for the framework:
    the model has to detect a risk type it has never seen during training.
    """
    from .train import train_model

    groups = groups_from_config(cfg)
    cal_method = (cfg.get("calibration", {}) or {}).get("method", "platt")
    model_cfg = cfg.get("model", {}) or {}
    splits = load_splits(data_dir)
    matrices = featurize_splits(splits, groups=groups, show_progress=show_progress)
    train, val, test = matrices.get("train"), matrices.get("val"), matrices.get("test")
    if train is None or test is None:
        raise ValueError("train and test splits are required")

    categories = sorted({c for c in train.categories if c})
    per_category: List[Dict[str, Any]] = []
    for held_out in categories:
        tr_mask = np.asarray([c != held_out for c in train.categories])
        te_mask = np.asarray([c == held_out for c in test.categories])
        if not te_mask.any() or len(set(train.y[tr_mask].tolist())) < 2:
            continue
        y_te = test.y[te_mask]
        if len(set(y_te.tolist())) < 2:
            # a single-class held-out split has no AUROC to report
            per_category.append({
                "category": held_out, "n": int(te_mask.sum()),
                "n_pos": int(y_te.sum()), "auroc": None,
                "note": "the held-out split is single-class, so AUROC is undefined",
            })
            continue
        model = train_model(
            train.rows(tr_mask),
            val,
            model_cfg.get("name", "rf"),
            seed,
            model_cfg.get("hyperparameters", {}) or {},
            cal_method,
        )[0]
        p = model.predict_proba(test.rows(te_mask).X, calibrated=True)
        a = float(auroc(y_te, p))
        per_category.append({
            "category": held_out,
            "n": int(te_mask.sum()),
            "n_pos": int(y_te.sum()),
            "auroc": round(a, 5),
        })
        if show_progress:
            print(f"[leave-one-category-out] {held_out:32} n={int(te_mask.sum()):4} AUROC={a:.4f}")

    scored = [c["auroc"] for c in per_category if c.get("auroc") is not None]
    out: Dict[str, Any] = {
        "experiment": "leave_one_category_out",
        "data_dir": str(data_dir),
        "seed": seed,
        "per_category": per_category,
        "mean_auroc": round(float(np.mean(scored)), 5) if scored else None,
        "std_auroc": round(float(np.std(scored)), 5) if len(scored) > 1 else None,
        "n_categories": len(scored),
        "note": (
            "each risk type is held out in turn; the model has to detect a "
            "risk type it has never been trained on"
        ),
    }
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
    return out
