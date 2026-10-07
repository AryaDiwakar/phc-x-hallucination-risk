"""PHC-X command line interface.

Entry point ``phc`` (see pyproject.toml). Every command is deterministic and
seeding-aware; machine-readable output goes to ``results/raw/*.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ------------------------------------------------------------------- commands
def cmd_build_dataset(args: argparse.Namespace) -> int:
    from phcx.dataset.build import build_dataset

    cfg = _load_yaml(args.config)
    summary = build_dataset(
        out_dir=args.out,
        samples_per_category=int(args.samples_per_category or cfg.get(
            "samples_per_category", 100)),
        n_families=int(args.families or 240),
        seed=int(args.seed if args.seed is not None else cfg.get("seed", 42)),
        ratios=tuple(cfg.get("split", {}).get("ratios", [0.7, 0.15, 0.15])),
    )
    _print_json(summary)
    return 0 if summary.get("leakage_passed", True) else 1


def cmd_train(args: argparse.Namespace) -> int:
    from phcx.experiments.train import load_config, run_training

    cfg = load_config(args.config)
    if args.model:
        cfg.setdefault("model", {})["name"] = args.model
    if args.experiment:
        cfg["experiment"] = args.experiment

    seeds: List[int] = []
    if args.seeds:
        seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    elif args.seed is not None:
        seeds = [int(args.seed)]
    else:
        raw = cfg.get("seeds") or [cfg.get("seed", 42)]
        seeds = [int(s) for s in raw]

    groups = args.groups.split(",") if args.groups else None
    base_experiment = str(cfg.get("experiment", "phcx"))
    reports = []
    for seed in seeds:
        if len(seeds) > 1:
            cfg["experiment"] = f"{base_experiment}_seed{seed}"
        res = run_training(
            cfg,
            data_dir=args.data,
            out_dir=args.out,
            checkpoint_dir=args.checkpoint_dir,
            seed=seed,
            groups=groups,
            show_progress=not args.quiet,
        )
        reports.append(res.report)

    if len(seeds) > 1:
        import statistics as _stats

        def agg(key_path: str) -> Dict[str, Any]:
            vals = []
            for r in reports:
                cur: Any = r
                for k in key_path.split("."):
                    cur = (cur or {}).get(k) if isinstance(cur, dict) else None
                if isinstance(cur, (int, float)):
                    vals.append(float(cur))
            if not vals:
                return {}
            return {"mean": round(_stats.mean(vals), 5),
                    "std": round(_stats.pstdev(vals), 5) if len(vals) > 1 else 0.0,
                    "values": [round(v, 5) for v in vals]}

        summary = {
            "experiment": args.experiment or base_experiment,
            "model": cfg.get("model", {}).get("name"),
            "seeds": seeds,
            "test": {k: agg(f"test.calibrated.{k}")
                     for k in ("auroc", "auprc", "f1", "precision", "recall",
                               "ece", "brier", "accuracy")},
            "threshold": agg("threshold"),
            "runs": [{"experiment": r["experiment"], "seed": r["seed"],
                      "test": r["test"]["calibrated"]} for r in reports],
        }
        out = Path(args.out) / "raw"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"multiseed_{(args.experiment or base_experiment)}.json"
        path.write_text(json.dumps(summary, indent=2))
        if args.json:
            _print_json(summary)
        elif not args.quiet:
            t = summary["test"]
            print(f"[train] {len(seeds)} seeds -> {path}")
            for k, v in t.items():
                print(f"  {k:10s} {v['mean']:.3f} +/- {v['std']:.3f}")
        return 0

    if args.json:
        _print_json(reports[0])
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from phcx.experiments.train import evaluate_checkpoint

    report = evaluate_checkpoint(
        args.checkpoint, data_dir=args.data, out_dir=args.out,
        recalibrate=args.recalibrate,
    )
    _print_json(report)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Recalibrate a checkpoint's probability outputs on the validation split."""
    from phcx.experiments.train import evaluate_checkpoint

    report = evaluate_checkpoint(
        args.checkpoint, data_dir=args.data, out_dir=args.out,
        recalibrate=True,
    )
    _print_json(report)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from phcx.experiments.benchmark import run_benchmark
    from phcx.experiments.train import load_config

    cfg = load_config(args.config)
    report = run_benchmark(cfg, data_dir=args.data, out_dir=args.out,
                           seed=args.seed, show_progress=not args.quiet)
    if args.json:
        _print_json(report)
    return 0 if report.get("gate_status") == "PASS" else 1


def cmd_ablation(args: argparse.Namespace) -> int:
    from phcx.experiments.ablation import run_ablations
    from phcx.experiments.train import load_config

    cfg = load_config(args.config)
    summary = run_ablations(
        cfg, data_dir=args.data, out_dir=args.out, seed=args.seed,
        model_name=args.model, show_progress=not args.quiet,
    )
    if args.json:
        _print_json(summary)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Paired significance tests between two saved checkpoints."""
    import numpy as np

    from phcx.experiments.metrics import significance_summary
    from phcx.experiments.train import load_checkpoint

    out: Dict[str, Any] = {}
    preds = []
    for label, path in (("a", args.a), ("b", args.b)):
        _, meta = load_checkpoint(path)
        npy = Path(str(path)).with_suffix("").parent / "test_predictions.npy"
        p = np.load(npy)
        preds.append((p, meta))
    (pa, meta_a), (pb, meta_b) = preds
    y = np.asarray(meta_a["y"], int)
    if pa.shape != pb.shape or len(y) != len(pa):
        print("checkpoint prediction vectors do not align", file=sys.stderr)
        return 2
    out = significance_summary(
        y, pa, pb,
        name_a=args.a, name_b=args.b,
        threshold=float(meta_a.get("threshold", 0.5)),
        n_boot=args.bootstrap, seed=args.seed,
    )
    _print_json(out)
    return 0


def cmd_risk(args: argparse.Namespace) -> int:
    """Run the full PHC-X pipeline on a single prompt (offline, deterministic)."""
    from phcx.intervention import RiskPolicy
    from phcx.pipeline import PHCPipeline
    from phcx.risk import RiskModel

    history: List[Dict[str, Any]] = []
    if args.history:
        history = json.loads(Path(args.history).read_text())
    evidence: List[Any] = []
    if args.evidence:
        raw = json.loads(Path(args.evidence).read_text())
        evidence = raw if isinstance(raw, list) else raw.get("evidence", [])

    model_path = args.model
    if not model_path:
        for candidate in (
            "checkpoints/phcx_train_baseline_seed42/model",
            "checkpoints/phcx_train_baseline/model",
            "checkpoints/phcx_rf_seed42/model",
        ):
            if Path(candidate).with_suffix(".json").exists():
                model_path = candidate
                break
    model = RiskModel.load(model_path) if model_path else None
    policy = RiskPolicy.from_yaml(args.policy) if args.policy else RiskPolicy()
    pipe = PHCPipeline(model=model, policy=policy,
                       groups=None if args.whitebox else
                       ["prompt", "history", "provenance", "consistency", "evidence"])
    pred = pipe.analyze(args.prompt, conversation_history=history,
                        evidence=evidence, use_retriever=not evidence)
    _print_json(pred.to_dict())
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from phcx.experiments.report import run_report
    from phcx.experiments.train import load_config

    cfg = load_config(args.config)
    stats = run_report(
        cfg, data_dir=args.data, checkpoint=args.checkpoint, out_dir=args.out,
        n_boot=args.bootstrap, seed=args.seed, show_progress=not args.quiet,
    )
    if args.json:
        _print_json(stats)
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    from phcx.experiments.paper import run_paper
    result = run_paper(stats_path=args.stats, paper_dir=args.paper_dir,
                       figures_dir=args.figures, compile=not args.no_compile,
                       engine=args.engine, show_progress=not args.quiet)
    if args.json:
        _print_json(result)
    return 0


def cmd_external_eval(args: argparse.Namespace) -> int:
    from phcx.experiments.external import evaluate_external, load_external_rows

    checkpoint = args.checkpoint or "checkpoints/phcx_train_baseline_seed42/model"
    datasets = {
        "diahalu": args.diahalu,
        "halueval_dialogue": args.halueval,
    }
    if args.halluhard:
        datasets["halluhard_seed"] = args.halluhard
    result = evaluate_external(
        checkpoint, datasets, out_path=args.out, limit=args.limit,
        n_boot=args.bootstrap, seed=args.seed, show_progress=not args.quiet,
    )
    if args.llm_baselines:
        from phcx.experiments.llm_baselines import evaluate_local_baselines

        rows = load_external_rows("diahalu", args.diahalu, limit=args.llm_limit)
        result["local_llm_baselines"] = evaluate_local_baselines(
            rows, model_names=tuple(args.llm_models.split(",")),
            sample_count=args.llm_samples, limit=args.llm_limit,
            seed=args.seed, out_path=None,
        )
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    if args.json:
        _print_json(result)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    if not Path("api/main.py").exists():
        print("api/main.py not found (Phase 9 not built yet)", file=sys.stderr)
        return 2
    import os
    import sys

    import uvicorn

    # console scripts do not put the working directory on sys.path
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    from api.main import app

    uvicorn.run(app, host=args.host, port=args.port, reload=False)
    return 0


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="phc",
        description="PHC-X: provenance-aware pre-generation hallucination risk "
                    "prediction and intervention",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def add(p: argparse.ArgumentParser, fn, **kw):
        p.set_defaults(func=fn, **kw)

    # build-dataset
    p = sub.add_parser("build-dataset", help="generate the PHC-Conversation benchmark")
    p.add_argument("--config", default="configs/dataset.yaml")
    p.add_argument("--out", default="data/phc_conversation")
    p.add_argument("--samples-per-category", type=int, default=None)
    p.add_argument("--families", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    add(p, cmd_build_dataset)

    # train
    p = sub.add_parser("train", help="train + calibrate + evaluate a risk model")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--data", default="data/phc_conversation")
    p.add_argument("--out", default="results")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--model", default=None, choices=["lr", "rf", "xgb", "lgbm", "mlp", "neural"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", default=None, help="comma-separated seeds "
                                                 "(default: seeds from config)")
    p.add_argument("--experiment", default=None)
    p.add_argument("--groups", default=None, help="comma-separated feature groups")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    add(p, cmd_train)

    # evaluate
    p = sub.add_parser("evaluate", help="evaluate a saved checkpoint on the test split")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/phc_conversation")
    p.add_argument("--out", default="results")
    p.add_argument("--recalibrate", action="store_true")
    add(p, cmd_evaluate)

    # calibrate
    p = sub.add_parser("calibrate", help="recalibrate a checkpoint on the val split")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/phc_conversation")
    p.add_argument("--out", default="results")
    add(p, cmd_calibrate)

    # benchmark
    p = sub.add_parser("benchmark", help="action-level safety/coverage gates on test")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--data", default="data/phc_conversation")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    add(p, cmd_benchmark)

    # ablation
    p = sub.add_parser("ablation", help="feature-group ablations A-G")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--data", default="data/phc_conversation")
    p.add_argument("--out", default="results")
    p.add_argument("--model", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    add(p, cmd_ablation)

    # stats
    p = sub.add_parser("stats", help="paired significance between two checkpoints")
    p.add_argument("--a", required=True, help="checkpoint A (…/model)")
    p.add_argument("--b", required=True, help="checkpoint B (…/model)")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    add(p, cmd_stats)

    # risk (single analysis)
    p = sub.add_parser("risk", help="analyze a single prompt with the full pipeline")
    p.add_argument("--prompt", required=True)
    p.add_argument("--history", default=None, help="JSON file: [{role, content}...]")
    p.add_argument("--evidence", default=None, help="JSON file: evidence passages")
    p.add_argument("--model", default=None, help="checkpoint path")
    p.add_argument("--policy", default=None, help="policy YAML")
    p.add_argument("--whitebox", action="store_true")
    add(p, cmd_risk)

    # report
    p = sub.add_parser("report", help="test metrics, significance tests, figures")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--data", default="data/phc_conversation")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default="reports")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    add(p, cmd_report)

    # paper
    p = sub.add_parser("paper", help="render macros/tables + compile paper/main.tex")
    p.add_argument("--stats", default="reports/stats.json")
    p.add_argument("--paper-dir", default="paper")
    p.add_argument("--figures", default="reports/figures")
    p.add_argument("--engine", default=None, help="tectonic | pdflatex")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    add(p, cmd_paper)

    # external evaluation
    p = sub.add_parser("external-eval", help="cross-dataset external benchmark evaluation")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--diahalu", default="data/raw/diahalu_bench.jsonl")
    p.add_argument("--halueval", default="data/raw/halueva_dialogue.json")
    p.add_argument("--halluhard", default=None,
                   help="optional unlabeled HalluHard seed JSON/JSONL")
    p.add_argument("--out", default="results/raw/external_eval.json")
    p.add_argument("--limit", type=int, default=None,
                   help="maximum source records per dataset")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--llm-baselines", action="store_true",
                   help="run explicitly labeled lexical Ollama approximations")
    p.add_argument("--llm-models", default="llama3.2:3b,qwen2.5:3b")
    p.add_argument("--llm-samples", type=int, default=3)
    p.add_argument("--llm-limit", type=int, default=50)
    add(p, cmd_external_eval)

    # serve
    p = sub.add_parser("serve", help="run the FastAPI service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    add(p, cmd_serve)

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
