"""Training / calibration / evaluation harness (spec sections 14-15)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

from ..calibration.core import (
    calibration_report,
    cross_fitted_calibration,
    enforce_min_coverage,
    make_calibrator,
    optimize_thresholds,
    reliability_bins,
)
from ..features import FEATURE_GROUPS
from ..features.engine import USE_FLAG_GROUPS
from ..risk import RiskModel
from .matrix import DatasetMatrix, featurize_splits, load_splits
from .metrics import (
    classification_metrics,
    per_category_metrics,
    template_lookup_scores,
)

MODEL_NAMES = ("lr", "rf", "xgb", "lgbm", "mlp", "neural")
NON_MODEL_GROUPS = [g for g in FEATURE_GROUPS if g != "model"]


def load_config(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def groups_from_config(cfg: Dict[str, Any]) -> List[str]:
    """Resolve the active feature groups from the ``features:`` config block.

    Every ``use_*`` switch is honoured: a group is kept only when all of the
    switches that map to it are enabled.  Previously these switches were
    ignored, so setting e.g. ``use_prompt: false`` had no effect at all.
    """
    feats = cfg.get("features", {}) or {}
    mode = feats.get("mode", "black_box")
    white = bool(feats.get("use_whitebox", False)) or mode == "white_box"

    groups: List[str] = []
    for g in NON_MODEL_GROUPS:
        flags = [
            f for f, gs in USE_FLAG_GROUPS.items()
            if g in gs and f != "use_whitebox"
        ]
        if flags and not all(bool(feats.get(f, True)) for f in flags):
            continue
        groups.append(g)
    if white:
        groups.append("model")
    return groups


@dataclass
class ExperimentResult:
    report: Dict[str, Any]
    model: Optional[RiskModel] = None
    matrices: Optional[Dict[str, DatasetMatrix]] = None
    p_test_raw: Optional[np.ndarray] = None
    p_test_cal: Optional[np.ndarray] = None


def _base_report(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    return {
        "experiment": cfg.get("experiment", "phcx_experiment"),
        "seed": seed,
        "model": (cfg.get("model") or {}).get("name", "rf"),
        "features_mode": (cfg.get("features") or {}).get("mode", "black_box"),
        "groups": None,
        "n_features": None,
        "label_rates": {},
    }


def train_model(
    train: DatasetMatrix,
    val: Optional[DatasetMatrix],
    model_name: str = "rf",
    seed: int = 42,
    hyper: Optional[Dict[str, Any]] = None,
    calibrate_method: str = "isotonic",
) -> tuple[RiskModel, np.ndarray, np.ndarray]:
    """Fit on train, calibrate on val. Returns (model, p_val_raw, p_val_cal)."""
    model = RiskModel(name=model_name, seed=seed, **(hyper or {}))
    t0 = time.perf_counter()
    if model_name == "neural":
        model.fit(train.X, train.y, train.feature_names,
                  X_val=None if val is None else val.X,
                  y_val=None if val is None else val.y)
    else:
        model.fit(train.X, train.y, train.feature_names)
    fit_s = time.perf_counter() - t0

    p_val_raw = p_val_cal = np.zeros(0)
    if val is not None and val.n_rows:
        p_val_raw = model.predict_proba(val.X, calibrated=False)
        cal = make_calibrator(calibrate_method)
        cal.fit(p_val_raw, val.y)
        model.calibrator = cal
        p_val_cal = model.predict_proba(val.X, calibrated=True)
    model.fit_seconds = fit_s  # type: ignore[attr-defined]
    return model, p_val_raw, p_val_cal


def evaluate(
    model: RiskModel,
    matrix: DatasetMatrix,
    threshold: float = 0.5,
    with_categories: bool = True,
    extra_scores: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    p_raw = model.predict_proba(matrix.X, calibrated=False)
    p_cal = model.predict_proba(matrix.X, calibrated=True)
    out: Dict[str, Any] = {
        "n": matrix.n_rows,
        "label_rate": round(matrix.label_rate, 5),
        "threshold": threshold,
        "raw": classification_metrics(matrix.y, p_raw, threshold),
        "calibrated": classification_metrics(matrix.y, p_cal, threshold),
    }
    if with_categories:
        out["per_category"] = per_category_metrics(matrix.y, p_cal, matrix.categories,
                                                    threshold)
    if matrix.heuristic is not None:
        out["heuristic_baseline"] = classification_metrics(matrix.y, matrix.heuristic,
                                                            threshold)
    for name, scores in (extra_scores or {}).items():
        if scores is not None and len(scores) == matrix.n_rows:
            out[name] = classification_metrics(matrix.y, scores, threshold)
    out["reliability"] = reliability_bins(p_cal, matrix.y)
    return out


def run_training(
    cfg: Dict[str, Any],
    data_dir: str | Path = "data/phc_conversation",
    out_dir: str | Path = "results",
    checkpoint_dir: str | Path = "checkpoints",
    seed: Optional[int] = None,
    groups: Optional[Sequence[str]] = None,
    save_model: bool = True,
    show_progress: bool = True,
) -> ExperimentResult:
    """Full train -> calibrate -> evaluate run driven by configs/train.yaml."""
    seed = int(seed if seed is not None else cfg.get("seed", 42))
    np.random.seed(seed)

    model_cfg = cfg.get("model", {}) or {}
    model_name = model_cfg.get("name", "rf")
    hyper = model_cfg.get("hyperparameters", {}) or {}
    cal_method = (cfg.get("calibration", {}) or {}).get("method", "isotonic")
    policy_cfg = cfg.get("policy", {}) or {}
    eval_cfg = cfg.get("evaluation", {}) or {}
    if show_progress:
        print(f"[train] data={data_dir} model={model_name} seed={seed}")

    splits = load_splits(data_dir)
    used_groups = list(groups) if groups else groups_from_config(cfg)
    matrices = featurize_splits(splits, groups=used_groups, show_progress=show_progress)

    train, val, test = matrices.get("train"), matrices.get("val"), matrices.get("test")
    if train is None or test is None:
        raise ValueError("train and test splits are required")

    # The leakage probe: a lookup table over the *prompt template*, fit on
    # the training split only.  It learns nothing about hallucination -- only
    # "prompts that look like this are positive X% of the time" -- so a
    # trained model has to clearly beat it to be doing anything.
    template_lookup = template_lookup_scores(splits.get("train", []), splits.get("test", []))

    model, p_val_raw, p_val_cal = train_model(
        train, val, model_name=model_name, seed=seed, hyper=hyper,
        calibrate_method=cal_method,
    )

    # threshold search on VALIDATION probabilities only
    threshold_search = None
    threshold = 0.5
    min_coverage = policy_cfg.get("min_coverage", None)
    thresholds = None
    if val is not None and val.n_rows:
        threshold_search = optimize_thresholds(
            p_val_cal, val.y,
            cost_fp=float(policy_cfg.get("cost_fp", 1.0)),
            cost_fn=float(policy_cfg.get("cost_fn", 3.0)),
            abstention_cost=float(policy_cfg.get("abstention_cost", 2.0)),
        )
        threshold = float(threshold_search.thresholds.get("decision", 0.5))
        thresholds = dict(threshold_search.thresholds)
        if min_coverage is not None:
            # the deployment gate constrains the *blocking* cut-point so that
            # enough turns are still answered
            cov = enforce_min_coverage(thresholds, p_val_cal, float(min_coverage))
            thresholds = cov["thresholds"]

    report = _base_report(cfg, seed)
    report.update({
        "groups": used_groups,
        "n_features": int(train.X.shape[1]),
        "n_rows": {k: m.n_rows for k, m in matrices.items()},
        "label_rates": {k: round(m.label_rate, 5) for k, m in matrices.items()},
        "fit_seconds": round(getattr(model, "fit_seconds", 0.0), 3),
        "threshold": threshold,
        "thresholds": thresholds if thresholds is not None else None,
        "threshold_objective": threshold_search.objective if threshold_search else None,
        "test": evaluate(model, test, threshold,
                         extra_scores={"template_lookup_baseline": template_lookup}),
        "calibration": None,
        "feature_importance": dict(
            sorted(model.feature_importance().items(),
                   key=lambda kv: -kv[1])[:25]),
    })
    if val is not None and val.n_rows:
        report["val"] = evaluate(model, val, threshold, with_categories=False)
        report["calibration"] = calibration_report(p_val_raw, p_val_cal, val.y)
        # Isotonic interpolation fits its training split exactly, so the
        # in-sample val ECE above is a memorisation artefact.  The
        # out-of-fold figure is the honest estimate of how the calibrator
        # generalises.
        report["calibration_out_of_fold"] = calibration_report(
            p_val_raw,
            cross_fitted_calibration(p_val_raw, val.y, method=cal_method, seed=seed),
            val.y,
        )
        if np.size(p_val_cal):
            from .metrics import auroc

            report["val_auroc"] = round(float(auroc(val.y, p_val_cal)), 5)

    result = ExperimentResult(report=report, model=model, matrices=matrices,
                              p_test_raw=model.predict_proba(test.X, calibrated=False),
                              p_test_cal=model.predict_proba(test.X, calibrated=True))

    # ---- persist ----------------------------------------------------------
    out_dir, checkpoint_dir = Path(out_dir), Path(checkpoint_dir)
    exp = str(report["experiment"])
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    import json

    (raw_dir / f"{exp}.json").write_text(json.dumps(report, indent=2))
    if save_model:
        ckpt = checkpoint_dir / exp / "model"
        model.save(ckpt)
        np.save(checkpoint_dir / exp / "test_predictions.npy", result.p_test_cal)
        (checkpoint_dir / exp / "test_meta.json").write_text(json.dumps({
            "sample_ids": test.sample_ids,
            "y": test.y.tolist(),
            "categories": test.categories,
            "threshold": threshold,
            "feature_names": test.feature_names,
            "groups": used_groups,
            "model": {"name": model_name, "hyper": hyper, "seed": seed},
            "calibration": cal_method,
            "label_rate": round(test.label_rate, 5),
        }, indent=2))
    if show_progress:
        t = report["test"]["calibrated"]
        print(f"[train] test AUROC={t['auroc']:.3f} AUPRC={t['auprc']:.3f} "
              f"F1={t['f1']:.3f} ECE={t['ece']:.3f} thr={threshold:.2f} "
              f"-> {raw_dir / (exp + '.json')}")
    return result


def run_from_args(
    config_path: str | Path = "configs/train.yaml",
    data_dir: str | Path = "data/phc_conversation",
    out_dir: str | Path = "results",
    seed: Optional[int] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> ExperimentResult:
    cfg = load_config(config_path)
    if model:
        cfg.setdefault("model", {})["name"] = model
    if kwargs.get("experiment"):
        cfg["experiment"] = kwargs["experiment"]
    return run_training(cfg, data_dir=data_dir, out_dir=out_dir, seed=seed)


# ------------------------------------------------------------ checkpoint reuse
def load_checkpoint(ckpt: str | Path) -> tuple[RiskModel, Dict[str, Any]]:
    """Load a saved model + its test metadata (feature names, groups, thr)."""
    import json

    ckpt = Path(ckpt)
    if ckpt.suffix:
        ckpt = ckpt.with_suffix("")
    model = RiskModel.load(ckpt)
    meta = json.loads((ckpt.parent / "test_meta.json").read_text())
    return model, meta


def evaluate_checkpoint(
    ckpt: str | Path,
    data_dir: str | Path = "data/phc_conversation",
    out_dir: str | Path | None = None,
    recalibrate: bool = False,
) -> Dict[str, Any]:
    """Evaluate (optionally recalibrate) a saved checkpoint on the test split."""
    import json as _json

    model, meta = load_checkpoint(ckpt)
    splits = load_splits(data_dir)
    rows = splits["test"]
    groups = meta.get("groups") or groups_from_config({})
    matrix = featurize_splits({"test": rows}, groups=groups)["test"]

    recalibrated = None
    if recalibrate and "val" in splits:
        val = featurize_splits({"val": splits["val"]}, groups=groups)["val"]
        p_raw = model.predict_proba(val.X, calibrated=False)
        cal = make_calibrator(meta.get("calibration", "isotonic"))
        cal.fit(p_raw, val.y)
        model.calibrator = cal
        recalibrated = {
            "before_ece": None,
            "after": calibration_report(p_raw, model.predict_proba(val.X), val.y),
        }

    thr = float(meta.get("threshold", 0.5))
    report = {
        "checkpoint": str(ckpt),
        "threshold": thr,
        "n_rows": matrix.n_rows,
        "test": evaluate(model, matrix, thr),
        "recalibrated": recalibrated,
    }
    if out_dir:
        out = Path(out_dir) / "raw"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"eval_{Path(str(ckpt)).parent.name}.json"
        path.write_text(_json.dumps(report, indent=2))
        report["saved_to"] = str(path)
    return report
