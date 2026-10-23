"""Render the research paper in the supplied IEEE A4 conference style.

The current ``paper/main.tex`` is kept as the readable research draft.  This
script derives a separate two-column conference source using IEEEtran, strips
the author-year labels from bibliography entries so citations are numbered,
and replaces the abstract with plain text (the supplied Word template forbids
math/symbols/footnotes in the title and abstract).

The author affiliation is intentionally a placeholder: it is not present in
the repository and must be supplied by the authors before submission.

Run from the repository root:

    python3 scripts/build_conference_paper.py
    cd paper && tectonic -X compile conference.tex
"""
from __future__ import annotations

import re
import argparse
import json
from pathlib import Path
from typing import Any, List

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

PREAMBLE = r"""\documentclass[conference,a4paper]{IEEEtran}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{cite}
\usepackage[hidelinks]{hyperref}

\input{macros.tex}

\title{Provenance Aware Conversational Pre Generation Hallucination Risk Prediction and Intervention}
\author{\IEEEauthorblockN{Arya D, Jeswanth G, N Sai Baba}
\IEEEauthorblockA{Capstone Project\\
Affiliation and contact details to be completed}}

\begin{document}
"""

ABSTRACT = r"""\begin{abstract}
We present PHC X, a framework that estimates whether an assistant's next
reply is at risk of hallucination before generation and selects an
intervention. It uses conversation history, claim provenance, consistency
checks, and evidence coverage rather than model internals. On a leak
controlled synthetic benchmark with a shared prompt pool, validation-selected
XGBoost reaches AUROC 0.975 on 107 held out prompts, while a prompt only
model scores 0.434. A three seed run gives mean AUROC 0.9753 with standard
deviation 0.0010. On
the original synthetic benchmark, AUROC is 0.987, but a template lookup
baseline reaches 0.808, so we report the leak controlled result as the
primary measure. On a context coverage counterfactual derived from HaluEval,
the system reaches AUROC 0.705. Human validation remains pending.
\end{abstract}

\begin{IEEEkeywords}
hallucination risk, conversational systems, provenance, calibration,
pre generation intervention
\end{IEEEkeywords}
"""


def _number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def _escape(text: Any) -> str:
    value = str(text)
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&")


def _conference_tables() -> str:
    """Compact tables sized for IEEE's two-column A4 layout."""
    stats = json.loads((ROOT / "reports" / "stats.json").read_text())
    tables: List[str] = []

    def table(caption: str, label: str, header: str, rows: List[str],
              spec: str, wide: bool = True) -> str:
        env = "table*" if wide else "table"
        width = r"\textwidth" if wide else r"\columnwidth"
        return (
            f"\\begin{{{env}}}[t]\n\\centering\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\n"
            "\\scriptsize\n"
            f"\\resizebox{{{width}}}{{!}}{{%\n"
            f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n"
            "\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}}\n"
            f"\\end{{{env}}}\n"
        )

    # Main model and baselines.
    main_rows: List[str] = []
    for name, metric in (
        ("PHC-X, RF calibrated", stats.get("model", {})),
        ("PHC-X, RF uncalibrated", stats.get("model_uncalibrated", {})),
        ("Heuristic", stats.get("heuristic_baseline", {})),
        ("Template lookup", stats.get("template_lookup_baseline", {})),
    ):
        main_rows.append(
            f"{_escape(name)} & {_number(metric.get('auroc'))} & "
            f"{_number(metric.get('auprc'))} & {_number(metric.get('f1'))} & "
            f"{_number(metric.get('precision'))} & {_number(metric.get('recall'))} & "
            f"{_number(metric.get('ece'))} & {_number(metric.get('brier'))} " + r"\\"
        )
    tables.append(table(
        "Test split: PHC-X and baselines.", "tab:main",
        "Method & AUROC & AUPRC & F1 & Precision & Recall & ECE & Brier",
        main_rows, "lrrrrrrr", wide=True,
    ))

    # Per-category table, wide to avoid long category names overflowing a
    # single IEEE column.
    cat_rows = []
    for name, row in sorted((stats.get("per_category") or {}).items()):
        cat_rows.append(
            f"{_escape(name)} & {row.get('n', '--')} & {_number(row.get('auroc'))} & "
            f"{_number(row.get('f1'))} & {_number(row.get('ece'))} " + r"\\"
        )
    tables.append(table("Per-category test results.", "tab:category",
                        "Category & $n$ & AUROC & F1 & ECE", cat_rows,
                        "lrrrr", wide=True))

    # Ablations.
    ab_rows = []
    descriptions = {
        "A_prompt_only": "Prompt only",
        "B_prompt_history": "Prompt and history",
        "C_history_provenance": "History and provenance",
        "D_history_provenance_contradiction": "Plus contradiction",
        "E_history_provenance_evidence": "Plus evidence",
        "F_full": "Full black box",
        "G_full_whitebox": "Full plus white box",
    }
    for key, score in stats.get("ablation_ranking", []):
        ab_rows.append(f"{key.split('_', 1)[0]} & {_escape(descriptions.get(key, key))} & {_number(score)} " + r"\\")
    tables.append(table("Feature ablations on the standard benchmark.", "tab:ablation",
                        "ID & Feature group & AUROC", ab_rows, "clr", wide=True))

    # Policy gates.
    gates = stats.get("benchmark", {}).get("gates", {})
    gate_rows = []
    for name, key, ok_key in (("Coverage", "coverage", "coverage_ok"),
                              ("Positives caught", "positives_caught", "positives_caught_ok"),
                              ("Safety margin", "safety_margin", None),
                              ("Generated risk", "risk_generated", None),
                              ("Base risk", "risk_overall", None)):
        status = "PASS" if ok_key and gates.get(ok_key) else "--"
        gate_rows.append(f"{name} & {_number(gates.get(key))} & {status} " + r"\\")
    tables.append(table("Action-level safety gates.", "tab:gates",
                        "Gate & Value & Status", gate_rows, "lrr", wide=False))

    # Leak-control comparison; includes the different classifier so the
    # benchmark comparison does not imply an identical learner.
    lc = stats.get("leak_control", {})
    main_ab = dict(stats.get("ablation_ranking", []))
    leak_ab = dict(lc.get("ablation_ranking", []))
    leak_rows = [
        f"Classifier & Random Forest & XGBoost, depth 3 " + r"\\",
        f"Templates identifying one category & 84.2\\% & 0.0\\% " + r"\\",
        f"Template lookup AUROC & {_number((stats.get('template_lookup_baseline') or {}).get('auroc'))} & "
        f"{_number((lc.get('leakage_probe') or {}).get('template_lookup_auroc'))} " + r"\\",
    ]
    for key, label in (("A_prompt_only", "A: prompt only"),
                       ("B_prompt_history", "B: prompt and history"),
                       ("C_history_provenance", "C: plus provenance"),
                       ("D_history_provenance_contradiction", "D: plus contradiction"),
                       ("E_history_provenance_evidence", "E: plus evidence"),
                       ("F_full", "F: full")):
        leak_rows.append(f"{_escape(label)} & {_number(main_ab.get(key))} & {_number(leak_ab.get(key))} " + r"\\")
    tables.append(table("Standard and leak-controlled benchmark comparison.", "tab:leakcontrol",
                        "Quantity & Standard & Leak controlled", leak_rows, "lrr", wide=True))

    # External evaluation.
    external_rows = []
    for name, label in (("diahalu", "DiaHalu, turn aligned"),
                        ("halueval_dialogue", "HaluEval pair, purity check"),
                        ("halueval_counterfactual", "HaluEval counterfactual")):
        item = (stats.get("external_eval", {}).get("datasets", {}) or {}).get(name, {})
        model = item.get("model", {}) or {}
        ci = model.get("auroc_ci", {}) or {}
        metrics = model.get("metrics", {}) or {}
        if metrics:
            external_rows.append(
                f"{_escape(label)} & {item.get('n_rows', '--')} & {_number(metrics.get('auroc'))} & "
                f"{_number(ci.get('lo'))}--{_number(ci.get('hi'))} & {_number(metrics.get('auprc'))} " + r"\\"
            )
    tables.append(table("External evaluation without retraining or recalibration.", "tab:external",
                        "Dataset & $n$ & AUROC & 95 percent CI & AUPRC",
                        external_rows, "lrrrr", wide=True))
    return "\n".join(tables)


def render() -> str:
    source = (PAPER / "main.tex").read_text(encoding="utf-8")
    start = source.index("\\begin{document}") + len("\\begin{document}")
    body = source[start:]

    # The title/author/date are part of the old one-column preamble. Keep the
    # body from \begin{document}, but substitute the conference front matter.
    body = re.sub(
        r"\\begin\{abstract\}.*?\\end\{abstract\}",
        lambda _match: ABSTRACT,
        body,
        count=1,
        flags=re.S,
    )
    # IEEE conference citations are numeric; the current draft uses optional
    # author-year labels in its thebibliography entries.
    body = re.sub(r"\\bibitem\[[^\]]*\]\{", r"\\bibitem{", body)
    body = body.replace(r"\input{tables.tex}", r"\input{conference_tables.tex}")
    body = body.replace(r"Tables~\ref{tab:gates} and~\ref{tab:actions}",
                        r"Table~\ref{tab:gates}")
    rendered = PREAMBLE + body

    # Guard against carrying template-prohibited material into the abstract.
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", rendered, re.S)
    title = re.search(r"\\title\{(.*?)\}", rendered, re.S)
    assert abstract and title
    for name, text in (("title", title.group(1)), ("abstract", abstract.group(1))):
        assert "$" not in text, f"math delimiter found in conference {name}"
        assert r"\footnote" not in text and r"\cite" not in text
        assert r"\%" not in text, f"symbolic percent found in conference {name}"
    return rendered


def main() -> int:
    out = PAPER / "conference.tex"
    out.write_text(render(), encoding="utf-8")
    tables = PAPER / "conference_tables.tex"
    tables.write_text(_conference_tables(), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {tables}")
    print("IEEE A4 conference layout; affiliation placeholder remains to be filled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
