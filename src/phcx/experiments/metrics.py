"""Evaluation metrics, per-category breakdowns, and statistical tests.

Statistical helpers follow standard formulations:
- bootstrap percentile CIs (resampling test-set rows)
- McNemar's exact test (discordant paired predictions)
- DeLong's test for two correlated ROC AUCs
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..calibration.core import brier_score, expected_calibration_error, nll


# --------------------------------------------------------------------- metrics
def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def classification_metrics(
    y_true: np.ndarray,
    p: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    pred = (p >= threshold).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    out = {
        "n": int(y.size),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "threshold": float(threshold),
        "accuracy": _safe_div(tp + tn, y.size),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "auroc": auroc(y, p),
        "auprc": auprc(y, p),
        "brier": brier_score(p, y),
        "nll": nll(p, y),
        "ece": expected_calibration_error(p, y),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, int).reshape(-1)
    p = np.asarray(p, float).reshape(-1)
    pos, neg = p[y == 1], p[y == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    # Mann-Whitney U with tie handling via average ranks
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(p.size, dtype=float)
    ranks[order] = np.arange(1, p.size + 1, dtype=float)
    # average ranks for ties
    sorted_p = p[order]
    i = 0
    while i < p.size:
        j = i
        while j + 1 < p.size and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    sum_pos = ranks[y == 1].sum()
    n_pos, n_neg = float(pos.size), float(neg.size)
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision (step-wise AP, standard PR-curve summary)."""
    y = np.asarray(y, int).reshape(-1)
    p = np.asarray(p, float).reshape(-1)
    n_pos = int((y == 1).sum())
    if n_pos == 0 or n_pos == y.size:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, y.size + 1)
    return float(np.sum(precision * y_sorted) / n_pos)


def per_category_metrics(
    y: np.ndarray,
    p: np.ndarray,
    categories: Sequence[str],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    cats = np.asarray(list(categories))
    out: Dict[str, Dict[str, float]] = {}
    for c in sorted(set(cats.tolist())):
        mask = cats == c
        if mask.sum() == 0:
            continue
        out[str(c)] = classification_metrics(y[mask], p[mask], threshold)
    return out


def leak_check(
    y_true: np.ndarray,
    p: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Row-level separation of flagged vs. generated cases (dashboard gate)."""
    y = np.asarray(y_true, int)
    p = np.asarray(p, float)
    flagged = p >= threshold
    return {
        "flagged_rate": float(flagged.mean()) if y.size else 0.0,
        "pos_rate_flagged": float(y[flagged].mean()) if flagged.any() else 0.0,
        "pos_rate_generated": float(y[~flagged].mean()) if (~flagged).any() else 0.0,
        "positives_caught": float(y[flagged].sum() / y.sum()) if y.sum() else 0.0,
    }


# --------------------------------------------------------------- leakage probe
_ENTITY_TOKEN = re.compile(r"\b[A-Z][A-Za-z]*-\d+[A-Z]?\b")
_NUMBER_TOKEN = re.compile(r"\b\d[\d,\.]*%?\b")
_FAMILY_TOKEN = re.compile(r"\bfam_\d+\b")


def prompt_template(prompt: str) -> str:
    """The *template* of a prompt, with entity/quantity slots masked.

    Two prompts share a template when they ask the same kind of question of
    different entities.  On PHC-Conversation every template belongs to
    exactly one risk category, so the template identifies the category and
    therefore the label's base rate -- a leak that has to be measured, not
    assumed away.
    """
    p = _ENTITY_TOKEN.sub("<E>", prompt or "")
    p = _NUMBER_TOKEN.sub("<N>", p)
    p = _FAMILY_TOKEN.sub("<ID>", p)
    return re.sub(r"\s+", " ", p).strip().lower()


def template_lookup_scores(
    train_rows: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    label_key: str = "label",
) -> np.ndarray:
    """Score each row by the training base rate of its prompt template.

    This is the baseline that matters for a synthetic benchmark: it learns
    nothing about hallucination, only "prompts that look like this are
    positive X% of the time".  A trained model that cannot clearly beat it
    is reading the benchmark, not the conversation.
    """
    rates: Dict[str, List[float]] = {}
    for r in train_rows:
        rates.setdefault(prompt_template(str(r.get("prompt", ""))), []).append(
            float(r.get(label_key, 0) or 0)
        )
    global_rate = float(np.mean([r.get(label_key, 0) or 0 for r in train_rows])) if train_rows else 0.0
    out = np.zeros(len(rows), dtype=float)
    for i, r in enumerate(rows):
        values = rates.get(prompt_template(str(r.get("prompt", ""))))
        out[i] = float(np.mean(values)) if values else global_rate
    return out



# ------------------------------------------------------------------- bootstrap
def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = auroc,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    stats: List[float] = []
    n = y.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            v = float(metric(y[idx], p[idx]))
        except Exception:
            continue
        if np.isfinite(v):
            stats.append(v)
    if not stats:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "mean": float("nan"), "std": float("nan"), "n_boot": 0}
    arr = np.asarray(stats)
    point = float(metric(y, p))
    return {
        "point": round(point, 5),
        "lo": round(float(np.percentile(arr, 100 * alpha / 2)), 5),
        "hi": round(float(np.percentile(arr, 100 * (1 - alpha / 2))), 5),
        "mean": round(float(arr.mean()), 5),
        "std": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 5),
        "n_boot": int(arr.size),
        "confidence": float(1 - alpha),
    }


# ---------------------------------------------------------------------- tests
def mcnemar_test(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Exact McNemar test on discordant predictions of two classifiers."""
    y = np.asarray(y_true, int)
    a = (np.asarray(p_a, float) >= threshold).astype(int)
    b = (np.asarray(p_b, float) >= threshold).astype(int)
    correct_a = (a == y)
    correct_b = (b == y)
    b01 = int((correct_a & ~correct_b).sum())   # A right, B wrong
    b10 = int((~correct_a & correct_b).sum())   # A wrong, B right
    n_disc = b01 + b10
    if n_disc == 0:
        p_value = 1.0
    else:
        try:
            from scipy.stats import binomtest

            p_value = float(binomtest(b01, n_disc, 0.5).pvalue)
        except Exception:  # pragma: no cover - scipy always present
            # continuity-corrected chi-square fallback
            chi = (abs(b01 - b10) - 1) ** 2 / n_disc
            p_value = float(np.exp(-chi / 2.0))  # rough 1-df tail
    return {
        "b_A_right_B_wrong": b01,
        "b_A_wrong_B_right": b10,
        "n_discordant": n_disc,
        "chi2_continuity": round(
            ((abs(b01 - b10) - 1) ** 2 / n_disc) if n_disc else 0.0, 5),
        "p_value": round(p_value, 6),
        "method": "exact-binomial",
    }


def _midrank(v: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(v, method="average")


def delong_test(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
) -> Dict[str, Any]:
    """DeLong test for two correlated ROC AUCs (placements, no resampling)."""
    y = np.asarray(y_true, int).reshape(-1)
    s_a = np.asarray(p_a, float).reshape(-1)
    s_b = np.asarray(p_b, float).reshape(-1)
    pos = y == 1
    neg = ~pos
    M, N = int(pos.sum()), int(neg.sum())
    if M == 0 or N == 0:
        return {"auc_a": 0.5, "auc_b": 0.5, "delta": 0.0, "z": 0.0,
                "p_value": 1.0, "std": 0.0, "error": "single-class labels"}

    pa_pos, pa_neg = s_a[pos], s_a[neg]
    pb_pos, pb_neg = s_b[pos], s_b[neg]

    # placement values (ties count 0.5)
    pi1a = ((pa_pos[:, None] > pa_neg[None, :]).astype(float)
            + 0.5 * (pa_pos[:, None] == pa_neg[None, :])).mean(axis=1)
    pi1b = ((pb_pos[:, None] > pb_neg[None, :]).astype(float)
            + 0.5 * (pb_pos[:, None] == pb_neg[None, :])).mean(axis=1)
    pi0a = ((pa_neg[:, None] < pa_pos[None, :]).astype(float)
            + 0.5 * (pa_neg[:, None] == pa_pos[None, :])).mean(axis=1)
    pi0b = ((pb_neg[:, None] < pb_pos[None, :]).astype(float)
            + 0.5 * (pb_neg[:, None] == pb_pos[None, :])).mean(axis=1)

    auc_a = float(pi1a.mean())
    auc_b = float(pi1b.mean())
    delta = auc_a - auc_b

    # covariance across models of placement values (positives / negatives)
    P = np.vstack([pi1a, pi1b])          # 2 x M
    Q = np.vstack([pi0a, pi0b])          # 2 x N
    theta = P.mean(axis=1)               # 2
    mu = Q.mean(axis=1)                  # 2
    V10 = (P @ P.T) / M - np.outer(theta, theta)
    V01 = (Q @ Q.T) / N - np.outer(mu, mu)

    var = float(V10[0, 0] + V10[1, 1] - 2 * V10[0, 1]
                + V01[0, 0] + V01[1, 1] - 2 * V01[0, 1])
    std = float(np.sqrt(max(var, 0.0)))
    if std <= 1e-12:
        z, p_value = 0.0, 1.0
    else:
        z = delta / std
        from scipy.stats import norm

        p_value = float(2 * (1 - norm.cdf(abs(z))))
    return {
        "auc_a": round(auc_a, 5),
        "auc_b": round(auc_b, 5),
        "delta": round(delta, 5),
        "std": round(std, 5),
        "z": round(float(z), 5),
        "p_value": round(float(p_value), 6),
        "method": "delong-correlated",
    }


def significance_summary(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    name_a: str = "model_a",
    name_b: str = "model_b",
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired comparison package used by ablation and paper pipelines."""
    return {
        "a": name_a,
        "b": name_b,
        "auroc_a": round(auroc(y, p_a), 5),
        "auroc_b": round(auroc(y, p_b), 5),
        "bootstrap_auroc_a": bootstrap_ci(y, p_a, auroc, n_boot=n_boot, seed=seed),
        "bootstrap_auroc_b": bootstrap_ci(y, p_b, auroc, n_boot=n_boot, seed=seed),
        "delong": delong_test(y, p_a, p_b),
        "mcnemar": mcnemar_test(y, p_a, p_b, threshold),
    }
