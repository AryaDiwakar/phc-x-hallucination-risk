"""Render the reproducible LaTeX paper from experiment JSON artifacts."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


ABLATION_DESCRIPTIONS = {
    "A_prompt_only": "Prompt only",
    "B_prompt_history": "Prompt + history",
    "C_history_provenance": "Prompt + history + provenance",
    "D_history_provenance_contradiction": "Prompt + history + provenance + contradiction",
    "E_history_provenance_evidence": "History + provenance + evidence",
    "F_full": "Full black-box feature set",
    "G_full_whitebox": "Full + white-box model signals",
}


def _f(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def _p(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return "<0.001" if number < 0.001 else f"{number:.3f}"


def _esc(value: Any) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _macro(name: str, value: Any) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def render_macros(stats: Dict[str, Any]) -> str:
    model = stats.get("model", {})
    uncal = stats.get("model_uncalibrated", {})
    heuristic = stats.get("heuristic_baseline", {})
    boot = stats.get("bootstrap", {})
    sig = stats.get("significance_vs_heuristic", {})
    dataset = stats.get("dataset", {})
    rates = dataset.get("label_rate", {}) or {}
    gates = (stats.get("benchmark") or {}).get("gates", {}) or {}
    multiseed = stats.get("multiseed", {}) or {}
    auc_summary = multiseed.get("auroc", {}) if isinstance(multiseed.get("auroc"), dict) else {}
    lr = ((stats.get("baselines") or {}).get("logistic_regression") or {}).get("calibrated", {}) or {}
    latency = stats.get("latency_ms", {}) or {}
    ablation = dict(stats.get("ablation_ranking") or [])
    external = (stats.get("external_eval") or {}).get("datasets", {})
    dia = external.get("diahalu", {}).get("model", {}).get("metrics", {})
    halu = external.get("halueval_dialogue", {}).get("model", {}).get("metrics", {})

    values = {
        "nrows": dataset.get("n_rows", "--"),
        "nfamilies": dataset.get("n_families", "--"),
        "ntrain": (dataset.get("split_sizes") or {}).get("train", "--"),
        "nval": (dataset.get("split_sizes") or {}).get("val", "--"),
        "ntest": stats.get("n_test", "--"),
        "agreement": _f(dataset.get("agreement")),
        "leakage": "true" if dataset.get("leakage_passed") else "false",
        "ncategories": len(stats.get("per_category") or {}),
        "trainrate": _f(rates.get("train")),
        "valrate": _f(rates.get("val")),
        "testrate": _f(rates.get("test")),
        "auroc": _f(model.get("auroc")),
        "auroclo": _f((boot.get("model_auroc") or {}).get("lo")),
        "aurochi": _f((boot.get("model_auroc") or {}).get("hi")),
        "auprc": _f(model.get("auprc")),
        "auprclo": _f((boot.get("model_auprc") or {}).get("lo")),
        "auprchi": _f((boot.get("model_auprc") or {}).get("hi")),
        "fone": _f(model.get("f1")),
        "precis": _f(model.get("precision")),
        "recla": _f(model.get("recall")),
        "ece": _f(model.get("ece")),
        "brier": _f(model.get("brier")),
        "threshold": _f(model.get("threshold"), 2),
        "labelrate": _f(stats.get("label_rate")),
        "uauroc": _f(uncal.get("auroc")),
        "ufone": _f(uncal.get("f1")),
        "uece": _f(uncal.get("ece")),
        "hauroc": _f(heuristic.get("auroc")),
        "hauprc": _f(heuristic.get("auprc")),
        "hfone": _f(heuristic.get("f1")),
        "hece": _f(heuristic.get("ece")),
        "lrauroc": _f(lr.get("auroc")),
        "lrfone": _f(lr.get("f1")),
        "lrece": _f(lr.get("ece")),
        "msmean": _f(auc_summary.get("mean")),
        "msstd": _f(auc_summary.get("std")),
        "msses": ", ".join(map(str, stats.get("multiseed_seeds", []))) or "--",
        "pdelong": _p((sig.get("delong") or {}).get("p_value")),
        "pmcnemar": _p((sig.get("mcnemar") or {}).get("p_value")),
        "ndiscordant": (sig.get("mcnemar") or {}).get("n_discordant", "--"),
        "coverage": _f(gates.get("coverage")),
        "coveragemin": _f(gates.get("coverage_min")),
        "caught": _f(gates.get("positives_caught")),
        "caughtmin": _f(gates.get("positives_caught_min")),
        "safetymargin": _f(gates.get("safety_margin")),
        "blockedshare": _f(gates.get("blocked_share")),
        "riskgenerated": _f(gates.get("risk_generated")),
        "riskoverall": _f(gates.get("risk_overall")),
        "gatestatus": _esc((stats.get("benchmark") or {}).get("gate_status", "--")),
        "latmedian": _f(latency.get("median"), 1),
        "lathigh": _f(latency.get("p95"), 1),
        "latmean": _f(latency.get("mean"), 1),
        "promptauroc": _f(ablation.get("A_prompt_only")),
        "evidenceauroc": _f(ablation.get("E_history_provenance_evidence")),
        "dian": external.get("diahalu", {}).get("n_rows", "--"),
        "diaauroc": _f(dia.get("auroc")),
        "halun": external.get("halueval_dialogue", {}).get("n_rows", "--"),
        "haluauroc": _f(halu.get("auroc")),
    }
    return "".join(_macro(name, value) for name, value in values.items())


def _main_rows(stats: Dict[str, Any]) -> List[str]:
    rows = [("PHC-X (RF, calibrated)", stats.get("model", {})),
            ("PHC-X (RF, uncalibrated)", stats.get("model_uncalibrated", {}))]
    lr = ((stats.get("baselines") or {}).get("logistic_regression") or {}).get("calibrated")
    if lr:
        rows.append(("Logistic regression (calibrated)", lr))
    if stats.get("heuristic_baseline"):
        rows.append(("Heuristic (no training)", stats["heuristic_baseline"]))
    return [
        f"{_esc(name)} & {_f(row.get('auroc'))} & {_f(row.get('auprc'))} & "
        f"{_f(row.get('f1'))} & {_f(row.get('precision'))} & {_f(row.get('recall'))} & "
        f"{_f(row.get('ece'))} & {_f(row.get('brier'))} " + r"\\"
        for name, row in rows
    ]


def render_tables(stats: Dict[str, Any]) -> str:
    out = [
        "\\begin{table}[t]\n\\centering\n\\caption{Test-split performance.}\n"
        "\\label{tab:main}\n\\small\n\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
        "Method & AUROC & AUPRC & F1 & Prec. & Rec. & ECE & Brier \\\\n\\midrule\n"
        + "\n".join(_main_rows(stats)) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    ]
    categories = []
    for name, row in sorted((stats.get("per_category") or {}).items()):
        categories.append(f"{_esc(name)} & {row.get('n', '--')} & {_f(row.get('auroc'))} & "
                         f"{_f(row.get('f1'))} & {_f(row.get('ece'))} " + r"\\")
    out.append("\\begin{table}[t]\n\\centering\n\\caption{Per-category test results.}\n"
               "\\label{tab:category}\n\\small\n\\begin{tabular}{lrrrr}\n\\toprule\n"
               "Category & $n$ & AUROC & F1 & ECE \\\\n\\midrule\n" + "\n".join(categories) +
               "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    ablation = stats.get("ablation_ranking") or []
    reference = dict(ablation).get("F_full")
    ab_rows = []
    for key, score in ablation:
        delta = "" if key == "F_full" or reference is None else f" & {score - reference:+.3f}"
        ab_rows.append(f"{_esc(key.split('_', 1)[0])} & {_esc(ABLATION_DESCRIPTIONS.get(key, key))} & "
                       f"{_f(score)}{delta} " + r"\\")
    if ab_rows:
        out.append("\\begin{table}[t]\n\\centering\n\\caption{Feature ablations.}\n\\label{tab:ablation}\n"
                   "\\small\n\\begin{tabular}{clrr}\n\\toprule\nID & Feature group & AUROC & $\\Delta$ \\\\n\\midrule\n" +
                   "\n".join(ab_rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    benchmark = stats.get("benchmark") or {}
    gates = benchmark.get("gates") or {}
    if gates:
        gate_rows = []
        for label, key, ok_key in (("Coverage", "coverage", "coverage_ok"),
                                   ("Positives caught", "positives_caught", "positives_caught_ok"),
                                   ("Safety margin", "safety_margin", None),
                                   ("Blocked share", "blocked_share", None)):
            status = "PASS" if ok_key and gates.get(ok_key) else "--"
            gate_rows.append(f"{label} & {_f(gates.get(key))} & {status} " + r"\\")
        out.append("\\begin{table}[t]\n\\centering\n\\caption{Action-level safety gates.}\n\\label{tab:gates}\n"
                   "\\small\n\\begin{tabular}{lrr}\n\\toprule\nGate & Value & Status \\\\n\\midrule\n" +
                   "\n".join(gate_rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    ext = (stats.get("external_eval") or {}).get("datasets", {})
    ext_rows = []
    for name, label in (("diahalu", "DiaHalu"), ("halueval_dialogue", "HaluEval dialogue")):
        item = ext.get(name, {})
        metric = item.get("model", {}).get("metrics", {})
        ci = item.get("model", {}).get("auroc_ci", {})
        if metric:
            ext_rows.append(f"{label} & {item.get('n_rows', '--')} & {_f(metric.get('auroc'))} & "
                            f"{_f(ci.get('lo'))}--{_f(ci.get('hi'))} & {_f(metric.get('auprc'))} " + r"\\")
    if ext_rows:
        out.append("\\begin{table}[t]\n\\centering\n"
                   "\\caption{External generalization without retraining.}\n\\label{tab:external}\n"
                   "\\small\n\\begin{tabular}{lrrrr}\n\\toprule\nDataset & $n$ & AUROC & 95\\% CI & AUPRC \\\\n\\midrule\n" +
                   "\n".join(ext_rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    # A few compact table headers use the readable ``\\n`` marker above;
    # convert it to real line breaks before writing TeX.
    return "".join(out).replace(r"\\n", r"\\" + "\n")


def run_paper(stats_path: str | Path = "reports/stats.json", paper_dir: str | Path = "paper",
              figures_dir: str | Path = "reports/figures", compile: bool = True,
              engine: Optional[str] = None, show_progress: bool = True) -> Dict[str, Any]:
    started = time.perf_counter()
    stats_path, paper_dir = Path(stats_path), Path(paper_dir)
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} not found — run `phc report` first")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "macros.tex").write_text(render_macros(stats), encoding="utf-8")
    (paper_dir / "tables.tex").write_text(render_tables(stats), encoding="utf-8")
    copied: List[str] = []
    source_dir = Path(figures_dir)
    if source_dir.is_dir():
        (paper_dir / "figures").mkdir(exist_ok=True)
        for image in sorted(source_dir.glob("*.png")):
            shutil.copy2(image, paper_dir / "figures" / image.name)
            copied.append(image.name)
    result: Dict[str, Any] = {"stats": str(stats_path), "macros": str(paper_dir / "macros.tex"),
                              "tables": str(paper_dir / "tables.tex"), "figures": copied,
                              "pdf": None, "compiled": False}
    if compile:
        executable = engine or ("tectonic" if shutil.which("tectonic") else
                                ("pdflatex" if shutil.which("pdflatex") else None))
        if executable is None:
            result["warning"] = "no LaTeX engine found"
        elif executable == "tectonic":
            proc = subprocess.run([executable, "-X", "compile", "main.tex"], cwd=paper_dir,
                                  capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError((proc.stdout + proc.stderr)[-4000:])
            result["compiled"] = (paper_dir / "main.pdf").exists()
        else:
            for _ in range(2):
                proc = subprocess.run([executable, "-interaction=nonstopmode", "main.tex"],
                                      cwd=paper_dir, capture_output=True, text=True)
                if proc.returncode:
                    raise RuntimeError((proc.stdout + proc.stderr)[-4000:])
            result["compiled"] = (paper_dir / "main.pdf").exists()
        if result["compiled"]:
            result["pdf"] = str(paper_dir / "main.pdf")
    result["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    if show_progress:
        print(f"[paper] macros + tables + {len(copied)} figures -> {paper_dir}")
        if result.get("pdf"):
            print(f"[paper] -> {result['pdf']} ({result['elapsed_seconds']}s)")
    return result
