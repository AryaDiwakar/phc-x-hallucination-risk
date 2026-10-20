"""Smoke tests for the ``phc`` command line interface.

The CLI had no coverage at all: a broken argument, a renamed subcommand or a
crash in an entry point would only show up when a user ran it.  These tests
invoke the real entry point in a subprocess against a tiny dataset, so they
also check that the command exits zero and produces its artifacts.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args, expect_zero: bool = True, cwd: Path = ROOT):
    """Invoke the CLI the way a user would: ``python -m phcx.cli.main ...``."""
    proc = subprocess.run(
        [sys.executable, "-m", "phcx.cli.main", *[str(a) for a in args]],
        cwd=str(cwd), capture_output=True, text=True, timeout=900,
    )
    if expect_zero:
        assert proc.returncode == 0, (
            f"`phc {' '.join(str(a) for a in args)}` failed "
            f"(exit {proc.returncode})\nstdout:\n{proc.stdout[-3000:]}\n"
            f"stderr:\n{proc.stderr[-3000:]}"
        )
    return proc


@pytest.fixture(scope="module")
def tiny_workspace(tmp_path_factory):
    """A tiny dataset plus a trained checkpoint, built once for the module."""
    base = tmp_path_factory.mktemp("cli")
    data = base / "data"
    results = base / "results"
    ckpts = base / "checkpoints"

    run_cli("build-dataset", "--config", "configs/dataset.yaml", "--out", data,
            "--samples-per-category", "3", "--families", "12", "--seed", "7")
    assert (data / "train.jsonl").exists()

    run_cli("train", "--config", "configs/train.yaml", "--data", data,
            "--out", results, "--checkpoint-dir", ckpts,
            "--model", "rf", "--seed", "42", "--experiment", "smoke", "--quiet")
    ckpt = ckpts / "smoke" / "model"
    assert (ckpts / "smoke" / "model.json").exists()
    return {"base": base, "data": data, "results": results,
            "ckpts": ckpts, "ckpt": ckpt}


def test_cli_help_lists_every_subcommand():
    proc = run_cli("--help")
    for sub in ("build-dataset", "train", "evaluate", "calibrate", "benchmark",
                "ablation", "stats", "risk", "report", "paper", "external-eval",
                "serve"):
        assert sub in proc.stdout, f"subcommand {sub} missing from --help"


def test_cli_build_dataset_writes_splits(tiny_workspace):
    ws = tiny_workspace
    for name in ("train.jsonl", "val.jsonl", "test.jsonl",
                 "leakage_report.json", "build_summary.json"):
        assert (ws["data"] / name).exists(), f"{name} was not written"
    summary = json.loads((ws["data"] / "build_summary.json").read_text())
    assert summary["n_rows"] > 0
    assert summary["leakage_passed"] is True


def test_cli_train_writes_checkpoint_and_results(tiny_workspace):
    ws = tiny_workspace
    raw = ws["results"] / "raw" / "smoke.json"
    assert raw.exists()
    report = json.loads(raw.read_text())
    assert "test" in report and report["test"]["calibrated"]["auroc"] >= 0.0
    assert (ws["ckpts"] / "smoke" / "model.json").exists()
    assert (ws["ckpts"] / "smoke" / "model_calibrator.joblib").exists()


def test_cli_risk_scores_a_prompt(tiny_workspace):
    ws = tiny_workspace
    proc = run_cli("risk", "--prompt", "What is the current accuracy of Atlas A?",
                   "--model", ws["ckpt"])
    payload = json.loads(proc.stdout)
    assert 0.0 <= payload["risk_score"] <= 1.0
    assert payload["risk_level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")
    assert payload["recommended_action"]


def test_cli_risk_accepts_history_and_evidence_files(tiny_workspace, tmp_path):
    ws = tiny_workspace
    history = tmp_path / "history.json"
    history.write_text(json.dumps([
        {"role": "user", "content": "How big is Atlas A?"},
        {"role": "assistant", "content": "Atlas A contains 50,000 samples."},
    ]))
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(["Atlas A holds 12,000 samples."]))
    proc = run_cli("risk", "--prompt", "Summarize Atlas A", "--model", ws["ckpt"],
                   "--history", history, "--evidence", evidence)
    payload = json.loads(proc.stdout)
    assert payload["recommended_action"]
    assert payload["features"]


def test_cli_evaluate_and_calibrate(tiny_workspace):
    ws = tiny_workspace
    run_cli("evaluate", "--checkpoint", ws["ckpt"], "--data", ws["data"],
            "--out", ws["results"], "--recalibrate")
    assert (ws["results"] / "raw" / "eval_smoke.json").exists()
    run_cli("calibrate", "--checkpoint", ws["ckpt"], "--data", ws["data"],
            "--out", ws["results"])
    assert (ws["results"] / "raw" / "eval_smoke.json").exists()


def test_cli_benchmark_and_ablation(tiny_workspace):
    ws = tiny_workspace
    run_cli("benchmark", "--config", "configs/train.yaml", "--data", ws["data"],
            "--out", ws["results"], "--quiet")
    bench = ws["results"] / "raw" / "benchmark_phcx_train_baseline.json"
    assert bench.exists()
    data = json.loads(bench.read_text())
    assert "gates" in data and "action_distribution" in data

    run_cli("ablation", "--config", "configs/ablation.yaml", "--data", ws["data"],
            "--out", ws["results"], "--quiet")
    assert (ws["results"] / "raw" / "ablation_phcx_train_baseline.json").exists()


def test_cli_stats_compares_two_checkpoints(tiny_workspace):
    ws = tiny_workspace
    run_cli("train", "--config", "configs/train.yaml", "--data", ws["data"],
            "--out", ws["results"], "--checkpoint-dir", ws["ckpts"],
            "--model", "lr", "--seed", "42", "--experiment", "smoke_lr", "--quiet")
    other = ws["ckpts"] / "smoke_lr" / "model"
    proc = run_cli("stats", "--a", ws["ckpt"], "--b", other, "--bootstrap", "50")
    payload = json.loads(proc.stdout)
    assert "mcnemar" in payload and "delong" in payload


def test_cli_report_and_paper(tiny_workspace):
    ws = tiny_workspace
    reports = ws["base"] / "reports"
    run_cli("report", "--config", "configs/train.yaml", "--data", ws["data"],
            "--checkpoint", ws["ckpt"], "--out", reports, "--bootstrap", "50",
            "--quiet")
    stats = reports / "stats.json"
    assert stats.exists()
    payload = json.loads(stats.read_text())
    assert payload["n_test"] > 0
    assert "model" in payload and "heuristic_baseline" in payload

    paper = ws["base"] / "paper"
    run_cli("paper", "--stats", stats, "--paper-dir", paper,
            "--figures", reports / "figures", "--no-compile", "--quiet")
    assert (paper / "macros.tex").exists()
    assert (paper / "tables.tex").exists()


def test_cli_unknown_subcommand_fails():
    proc = run_cli("definitely-not-a-command", expect_zero=False)
    assert proc.returncode != 0


def test_cli_risk_requires_a_prompt(tiny_workspace):
    proc = run_cli("risk", expect_zero=False)
    assert proc.returncode != 0
