"""Publication figures for the PHC-X report (matplotlib, Agg backend)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    precision_recall_curve,
    roc_curve,
)

COLORS = {
    "phcx": "#5aa9ff",
    "heuristic": "#37d399",
    "alt": "#f4c542",
    "neg": "#ff5c72",
    "muted": "#8b97ab",
}
BG = "#0e1117"
FG = "#dfe6f1"


def _style(ax) -> None:
    ax.set_facecolor("#161b24")
    for spine in ax.spines.values():
        spine.set_color("#263044")
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(BG)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def plot_roc(y: np.ndarray, curves: Dict[str, np.ndarray], out: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    _style(ax)
    for i, (name, p) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y, p)
        color = COLORS["phcx"] if i == 0 else (COLORS["heuristic"] if i == 1 else COLORS["alt"])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color=COLORS["muted"], lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Pre-generation risk ROC (test split)")
    ax.legend(facecolor="#1a2130", edgecolor="#263044", labelcolor=FG, fontsize=9)
    ax.grid(color="#212a3b", lw=0.5)
    return _save(fig, Path(out))


def plot_pr(y: np.ndarray, curves: Dict[str, np.ndarray], out: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    _style(ax)
    for i, (name, p) in enumerate(curves.items()):
        prec, rec, _ = precision_recall_curve(y, p)
        color = COLORS["phcx"] if i == 0 else (COLORS["heuristic"] if i == 1 else COLORS["alt"])
        ap = float(np.sum(np.diff(rec) * prec[:-1]))
        ax.plot(rec, prec, color=color, lw=2, label=f"{name} (AP={ap:.3f})")
    baseline = float(np.mean(y))
    ax.axhline(baseline, ls="--", color=COLORS["muted"], lw=1,
               label=f"base rate={baseline:.2f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision–recall (test split)")
    ax.legend(facecolor="#1a2130", edgecolor="#263044", labelcolor=FG, fontsize=9)
    ax.grid(color="#212a3b", lw=0.5)
    return _save(fig, Path(out))


def plot_reliability(p_raw: np.ndarray, p_cal: np.ndarray, y: np.ndarray,
                     out: str | Path, n_bins: int = 10) -> Path:
    from ..calibration.core import reliability_bins

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    _style(ax)
    ax.plot([0, 1], [0, 1], "--", color=COLORS["muted"], lw=1)
    for label, p, color in (("uncalibrated", p_raw, COLORS["neg"]),
                            ("isotonic", p_cal, COLORS["phcx"])):
        bins = reliability_bins(p, y, n_bins=n_bins)
        xs = [b["mean_predicted"] for b in bins]
        ys = [b["empirical_rate"] for b in bins]
        ax.plot(xs, ys, "o-", color=color, lw=2, ms=5, label=label)
    ax.set_xlabel("predicted risk")
    ax.set_ylabel("empirical hallucination rate")
    ax.set_title("Reliability (calibration) — test split")
    ax.legend(facecolor="#1a2130", edgecolor="#263044", labelcolor=FG, fontsize=9)
    ax.grid(color="#212a3b", lw=0.5)
    return _save(fig, Path(out))


def plot_ablation(runs: Dict[str, Dict[str, Any]], out: str | Path) -> Path:
    names = [k for k, v in runs.items() if v.get("test")]
    scores = [runs[k]["test"]["auroc"] for k in names]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _style(ax)
    order = np.argsort(scores)
    colors = [COLORS["phcx"] if names[i] == "F_full" else COLORS["alt"] for i in order]
    ax.barh([names[i] for i in order], [scores[i] for i in order], color=colors)
    for pos, i in enumerate(order):
        ax.text(scores[i] + 0.01, pos, f"{scores[i]:.3f}", va="center",
                color=FG, fontsize=9)
    ax.set_xlim(0.4, 1.05)
    ax.set_xlabel("AUROC")
    ax.set_title("Feature-group ablation (A → G)")
    ax.grid(color="#212a3b", lw=0.5, axis="x")
    return _save(fig, Path(out))


def plot_actions(benchmark: Dict[str, Any], out: str | Path) -> Path:
    dist = benchmark.get("action_distribution", {})
    labels = [k for k in dist if dist[k].get("n")]
    counts = [dist[k]["n"] for k in labels]
    rates = [dist[k].get("hallucination_rate") or 0.0 for k in labels]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    _style(ax)
    x = np.arange(len(labels))
    bars = ax.bar(x, counts, color=COLORS["phcx"], label="n turns")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f"{r * 100:.0f}%+", ha="center", color=COLORS["neg"], fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("test turns")
    ax.set_title("Policy action distribution (pos = hallucination rate)")
    ax.grid(color="#212a3b", lw=0.5, axis="y")
    return _save(fig, Path(out))


def plot_risk_hist(y: np.ndarray, p_model: np.ndarray, p_heur: Optional[np.ndarray],
                   out: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2 if p_heur is not None else 1,
                             figsize=(9.4 if p_heur is not None else 5.2, 4.0))
    axes = np.atleast_1d(axes)
    panels = [("PHC-X model", p_model)]
    if p_heur is not None:
        panels.append(("heuristic", p_heur))
    for ax, (title, p) in zip(axes, panels):
        _style(ax)
        bins = np.linspace(0, 1, 21)
        ax.hist(p[y == 0], bins=bins, color=COLORS["heuristic"], alpha=0.75, label="safe")
        ax.hist(p[y == 1], bins=bins, color=COLORS["neg"], alpha=0.75, label="hallucination")
        ax.set_xlabel("predicted risk")
        ax.set_ylabel("count")
        ax.set_title(f"Risk distribution — {title}")
        ax.legend(facecolor="#1a2130", edgecolor="#263044", labelcolor=FG, fontsize=9)
        ax.grid(color="#212a3b", lw=0.5)
    fig.tight_layout()
    return _save(fig, Path(out))


def make_all(
    y: np.ndarray,
    p_model: np.ndarray,
    p_heur: Optional[np.ndarray],
    p_raw: Optional[np.ndarray],
    benchmark: Optional[Dict[str, Any]],
    ablation: Optional[Dict[str, Any]],
    out_dir: str | Path = "reports/figures",
) -> List[Path]:
    out_dir = Path(out_dir)
    written: List[Path] = []
    curves = {"PHC-X": p_model}
    if p_heur is not None:
        curves["heuristic"] = p_heur
    written.append(plot_roc(y, curves, out_dir / "fig_roc.png"))
    written.append(plot_pr(y, curves, out_dir / "fig_pr.png"))
    if p_raw is not None:
        written.append(plot_reliability(p_raw, p_model, y,
                                        out_dir / "fig_reliability.png"))
    written.append(plot_risk_hist(y, p_model, p_heur, out_dir / "fig_risk_hist.png"))
    if benchmark:
        written.append(plot_actions(benchmark, out_dir / "fig_actions.png"))
    if ablation and ablation.get("runs"):
        written.append(plot_ablation(ablation["runs"], out_dir / "fig_ablation.png"))
    return written
