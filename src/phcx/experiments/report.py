"""Aggregate statistical report + figures for the paper (spec sections 16-17)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..calibration.core import calibration_report
from .figures import make_all
from .matrix import DatasetMatrix, featurize_splits, load_splits
from .metrics import (
    auprc,
    auroc,
    bootstrap_ci,
    classification_metrics,
    delong_test,
    mcnemar_test,
    per_category_metrics,
    prompt_template,
    template_lookup_scores,
)
from .train import groups_from_config, load_config, load_checkpoint


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _leakage_probe(splits: Dict[str, Any]) -> Dict[str, Any]:
    """How much does the prompt identify the risk category?

    Computed on the training split and reported on the test split, exactly
    like every other baseline, so the two benchmarks are measured the same
    way.
    """
    train = splits.get("train", [])
    test = splits.get("test", [])
    p = template_lookup_scores(train, test)
    y = np.asarray([int(r.get("label", 0) or 0) for r in test], dtype=int)
    templates = {prompt_template(str(r.get("prompt", ""))) for r in test}
    cats_per_template: Dict[str, set] = {}
    for r in train:
        cats_per_template.setdefault(
            prompt_template(str(r.get("prompt", ""))), set()
        ).add(str(r.get("category", "unknown")))
    unique = sum(1 for t in templates if len(cats_per_template.get(t, set())) == 1)
    return {
        "template_lookup_auroc": round(float(auroc(y, p)), 5) if len(y) else None,
        "n_distinct_templates": len(templates),
        "frac_templates_uniquely_categorised": round(unique / len(templates), 5) if templates else None,
        "note": (
            "a template-lookup table that never sees the conversation; a "
            "trained model must clearly beat it to be reading the context"
        ),
    }


def run_report(
    cfg: Optional[Dict[str, Any]] = None,
    data_dir: str | Path = "data/phc_conversation",
    checkpoint: str | Path | None = None,
    out_dir: str | Path = "reports",
    results_dir: str | Path = "results/raw",
    seed: int = 42,
    n_boot: int = 1000,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Test-split metrics, significance tests, and paper figures."""
    t0 = time.perf_counter()
    cfg = cfg if cfg is not None else load_config("configs/train.yaml")
    results_dir, out_dir = Path(results_dir), Path(out_dir)

    # checkpoint discovery
    if checkpoint is None:
        for cand in ("checkpoints/phcx_train_baseline_seed42/model",
                     "checkpoints/phcx_train_baseline/model",
                     "checkpoints/phcx_rf_seed42/model"):
            if Path(cand).with_suffix(".json").exists():
                checkpoint = cand
                break
    if checkpoint is None or not Path(str(checkpoint)).with_suffix(".json").exists():
        raise FileNotFoundError("no trained checkpoint found — run `phc train` first")

    model, meta = load_checkpoint(checkpoint)
    groups = meta.get("groups") or groups_from_config(cfg)
    threshold = float(meta.get("threshold", 0.5))
    if show_progress:
        print(f"[report] checkpoint={checkpoint} groups={len(groups)} thr={threshold:.2f}")

    splits = load_splits(data_dir)
    matrices = featurize_splits({"test": splits["test"]}, groups=groups,
                                show_progress=show_progress)
    test: DatasetMatrix = matrices["test"]

    p_cal = model.predict_proba(test.X, calibrated=True)
    p_raw = model.predict_proba(test.X, calibrated=False)
    p_heur = test.heuristic if test.heuristic is not None else None
    y = test.y

    stats: Dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "seed": seed,
        "n_test": int(test.n_rows),
        "label_rate": round(float(y.mean()), 5),
        "threshold": threshold,
        "n_boot": n_boot,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    build_summary = _load_json(Path(data_dir) / "build_summary.json")
    if build_summary:
        stats["dataset"] = {
            "n_rows": build_summary.get("n_rows"),
            "n_families": build_summary.get("n_families"),
            "split_sizes": build_summary.get("split_sizes"),
            "label_rate": build_summary.get("label_rate"),
            "label_source": build_summary.get("label_source"),
            "agreement": build_summary.get("gold_verified_agreement"),
            "leakage_passed": build_summary.get("leakage_passed"),
            "seed": build_summary.get("seed"),
        }

    stats["model"] = classification_metrics(y, p_cal, threshold)
    stats["model_uncalibrated"] = classification_metrics(y, p_raw, threshold)
    if p_heur is not None:
        stats["heuristic_baseline"] = classification_metrics(y, p_heur, threshold)
    # Leakage probe: a lookup table over the prompt template, fit on the
    # training split only.  It knows nothing about hallucination, so it
    # measures how much of the headline is explained by "prompts that look
    # like this tend to be positive".
    p_lookup = template_lookup_scores(splits.get("train", []), splits.get("test", []))
    stats["template_lookup_baseline"] = classification_metrics(y, p_lookup, threshold)
    stats["leakage_probe"] = _leakage_probe(splits)
    stats["per_category"] = per_category_metrics(y, p_cal, test.categories, threshold)
    stats["calibration"] = calibration_report(p_raw, p_cal, y)

    # ---- significance -----------------------------------------------------
    stats["bootstrap"] = {
        "model_auroc": bootstrap_ci(y, p_cal, auroc, n_boot=n_boot, seed=seed),
        "model_auprc": bootstrap_ci(y, p_cal, auprc, n_boot=n_boot, seed=seed),
    }
    if p_heur is not None:
        stats["bootstrap"]["heuristic_auroc"] = bootstrap_ci(
            y, p_heur, auroc, n_boot=n_boot, seed=seed)
        stats["significance_vs_heuristic"] = {
            "delong": delong_test(y, p_cal, p_heur),
            "mcnemar": mcnemar_test(y, p_cal, p_heur, threshold),
        }

    # multi-seed summary if present
    ms = None
    for name in ("multiseed_phcx_train_baseline.json", "multiseed_phcx_rf.json"):
        ms = _load_json(results_dir / name)
        if ms:
            break
    if ms:
        stats["multiseed"] = ms.get("test", {})
        stats["multiseed_seeds"] = ms.get("seeds", [])

    # ---- the leak-controlled variant of the benchmark ---------------------
    # Same generator, same label logic, one shared prompt pool: the prompt
    # cannot identify the risk category any more.
    leak = _load_json(results_dir / "ablation_phcx_leakcontrol.json")
    leak_bench = _load_json(results_dir / "benchmark_phcx_leakcontrol.json")
    leak_summary = _load_json(Path("data/phc_conversation_leakcontrol") / "build_summary.json")
    leak_train = _load_json(results_dir / "phcx_leakcontrol.json")
    model_selection = _load_json(Path("reports/model_selection_leakcontrol.json"))
    leak_splits = load_splits("data/phc_conversation_leakcontrol")
    if leak or leak_summary:
        block: Dict[str, Any] = {}
        if leak_summary:
            block["dataset"] = {
                "n_rows": leak_summary.get("n_rows"),
                "n_families": leak_summary.get("n_families"),
                "categories": leak_summary.get("categories"),
                "label_rate": leak_summary.get("label_rate"),
                "gold_verified_agreement": leak_summary.get("gold_verified_agreement"),
            }
            block["dataset"]["n_test"] = (leak_summary.get("split_sizes") or {}).get("test")
        if model_selection:
            selected = model_selection.get("selected") or {}
            block["model_selection"] = {
                "validation_only": model_selection.get("validation_only"),
                "selection_metric": model_selection.get("selection_metric"),
                "selected": selected,
                "model": selected.get("model"),
                "hyperparameters": selected.get("hyperparameters"),
                "candidates": [
                    {
                        "name": c.get("name"),
                        "model": c.get("model"),
                        "hyperparameters": c.get("hyperparameters"),
                        "validation": c.get("validation"),
                    }
                    for c in (model_selection.get("candidates") or [])
                ],
            }
        if leak_splits:
            block["leakage_probe"] = _leakage_probe(leak_splits)
        if leak_train and isinstance(leak_train.get("test"), dict):
            block["model"] = leak_train["test"].get("calibrated")
            block["template_lookup_baseline"] = leak_train["test"].get("template_lookup_baseline")
        if leak:
            block["ablation_ranking"] = sorted(
                ((k, (v.get("test", {}) or {}).get("auroc"))
                 for k, v in (leak.get("runs") or {}).items()),
                key=lambda kv: -(kv[1] or 0.0),
            )
        if leak_bench:
            block["benchmark"] = {
                "gates": leak_bench.get("gates"),
                "gate_status": leak_bench.get("gate_status"),
            }
        stats["leak_control"] = block

    # ---- generalisation stress tests -------------------------------------
    cross = _load_json(results_dir / "cross_label_generalization.json")
    if cross:
        stats["cross_label_generalization"] = cross.get("results", {})
    loco = _load_json(results_dir / "leave_one_category_out.json")
    if loco:
        stats["leave_one_category_out"] = {
            "per_category": loco.get("per_category"),
            "mean_auroc": loco.get("mean_auroc"),
            "std_auroc": loco.get("std_auroc"),
            "n_categories": loco.get("n_categories"),
        }

    # other trained baselines (e.g. logistic regression)
    lr = _load_json(results_dir / "phcx_lr.json")
    if lr and isinstance(lr.get("test"), dict):
        stats["baselines"] = stats.get("baselines", {})
        stats["baselines"]["logistic_regression"] = {
            "calibrated": lr["test"].get("calibrated"),
            "raw": lr["test"].get("raw"),
        }

    benchmark = _load_json(results_dir / "benchmark_phcx_train_baseline.json")
    if benchmark:
        stats["benchmark"] = {
            "gates": benchmark.get("gates"),
            "gate_status": benchmark.get("gate_status"),
            "action_distribution": benchmark.get("action_distribution"),
        }
    ablation = _load_json(results_dir / "ablation_phcx_train_baseline.json")
    if ablation:
        stats["ablation_ranking"] = sorted(
            ((k, v.get("test", {}).get("auroc"))
             for k, v in (ablation.get("runs") or {}).items()
             if v.get("test")),
            key=lambda kv: -(kv[1] or 0),
        )
    external = _load_json(results_dir / "external_eval.json")
    if external:
        stats["external_eval"] = external

    # ---- latency (warm, measured locally) ---------------------------------
    try:
        import time as _time
        from ..pipeline import PHCPipeline

        hist = [
            {"role": "user", "content": "Tell me about the Apollo program."},
            {"role": "assistant",
             "content": "Apollo was NASA's human spaceflight program, active 1961-1975."},
        ]
        prompts = [
            "Didn't NASA say the program ended in 1979?",
            "So the first Moon landing was in 1969, right?",
            "What about the satellite we launched last year?",
        ]
        pipe = PHCPipeline(model=model, groups=groups)
        pipe.analyze(prompts[0], conversation_history=hist)  # warm-up
        times = []
        for i in range(25):
            t = _time.perf_counter()
            pipe.analyze(prompts[i % len(prompts)], conversation_history=hist)
            times.append((_time.perf_counter() - t) * 1000.0)
        arr = np.asarray(times)
        stats["latency_ms"] = {
            "n": len(times),
            "median": round(float(np.median(arr)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "mean": round(float(arr.mean()), 2),
            "note": "warm PHCPipeline.analyze incl. SHAP, machine-local",
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        stats["latency_ms"] = {"error": str(exc)}

    # ---- figures ----------------------------------------------------------
    figs = make_all(
        y=y, p_model=p_cal, p_heur=p_heur, p_raw=p_raw,
        benchmark=benchmark, ablation=ablation,
        out_dir=out_dir / "figures",
    )
    stats["figures"] = [str(f) for f in figs]

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stats.json"
    path.write_text(json.dumps(stats, indent=2, default=str))
    stats["elapsed_seconds"] = round(time.perf_counter() - t0, 2)

    if show_progress:
        m = stats["model"]
        print(f"[report] AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f} "
              f"F1={m['f1']:.3f} ECE={m['ece']:.3f}")
        if "significance_vs_heuristic" in stats:
            sig = stats["significance_vs_heuristic"]
            print(f"[report] vs heuristic: DeLong p={sig['delong']['p_value']:.4f}, "
                  f"McNemar p={sig['mcnemar']['p_value']:.4f}")
        print(f"[report] {len(figs)} figures -> {out_dir / 'figures'}")
        print(f"[report] -> {path}")
    return stats
