"""Probability calibration (spec section 14).

Implements Platt scaling, isotonic regression and (probability) temperature
scaling, plus ECE / Brier / NLL metrics and reliability-diagram data.
Calibrators are ALWAYS fit on a held-out validation split -- never on test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np

from ..utils.io import ensure_dir


class Calibrator:
    method = "none"

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=float), 0.0, 1.0)

    def save(self, path: str | Path) -> None:
        ensure_dir(Path(path).parent)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Calibrator":
        return joblib.load(path)


class IdentityCalibrator(Calibrator):
    method = "none"


class PlattCalibrator(Calibrator):
    """Platt scaling: sigmoid(a*logit(p) + b)."""

    method = "platt"

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        if len(np.unique(y)) < 2:
            return self
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(logit, y)
        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        z = self.a * np.log(p / (1 - p)) + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


class IsotonicCalibrator(Calibrator):
    method = "isotonic"

    def __init__(self) -> None:
        self._iso = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        p = np.asarray(p, dtype=float)
        if len(np.unique(p)) < 3 or len(np.unique(y)) < 2:
            return self
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        self._iso = iso
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self._iso is None:
            return np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
        return np.clip(self._iso.transform(np.asarray(p, dtype=float)), 0.0, 1.0)


class TemperatureCalibrator(Calibrator):
    """Temperature scaling over logits (equivalently, power scaling of odds)."""

    method = "temperature"

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "TemperatureCalibrator":
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p))
        best_t, best_nll = 1.0, float("inf")
        for t in np.arange(0.25, 4.01, 0.05):
            z = logit / t
            q = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
            q = np.clip(q, 1e-7, 1 - 1e-7)
            nll = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
            if nll < best_nll:
                best_t, best_nll = float(t), nll
        self.temperature = best_t
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p)) / max(1e-6, self.temperature)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40)))


def make_calibrator(method: str) -> Calibrator:
    method = (method or "none").lower()
    if method == "platt":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    if method == "temperature":
        return TemperatureCalibrator()
    return IdentityCalibrator()


# ---------------------------------------------------------------- metrics
def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if not mask.any():
            continue
        ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> List[Dict[str, float]]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        out.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "mean_predicted": float(p[mask].mean()) if mask.any() else float("nan"),
                "empirical_rate": float(y[mask].mean()) if mask.any() else float("nan"),
                "count": int(mask.sum()),
            }
        )
    return [b for b in out if b["count"] > 0]


def calibration_report(p_raw: np.ndarray, p_cal: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    return {
        "uncalibrated": {
            "ece": round(expected_calibration_error(p_raw, y), 5),
            "brier": round(brier_score(p_raw, y), 5),
            "nll": round(nll(p_raw, y), 5),
        },
        "calibrated": {
            "ece": round(expected_calibration_error(p_cal, y), 5),
            "brier": round(brier_score(p_cal, y), 5),
            "nll": round(nll(p_cal, y), 5),
        },
        "reliability_uncalibrated": reliability_bins(p_raw, y),
        "reliability_calibrated": reliability_bins(p_cal, y),
    }


@dataclass
class ThresholdSearchResult:
    thresholds: Dict[str, float] = field(default_factory=dict)
    objective: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


def optimize_thresholds(
    p: np.ndarray,
    y: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 3.0,
    abstention_cost: float = 2.0,
) -> ThresholdSearchResult:
    """Cost-sensitive threshold search on VALIDATION data.

    Returns decision thresholds for retrieve/clarify actions plus the risk
    level cut-points.  Never call this on the test set.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    grid = np.arange(0.05, 0.96, 0.01)
    best_c, best_t = float("inf"), 0.5
    for t in grid:
        pred = (p >= t).astype(int)
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_c:
            best_c, best_t = float(cost), float(t)
    # separate, more conservative threshold for abstention/clarify
    best_abst, best_abst_c = best_t, float("inf")
    for t in np.arange(best_t, 1.0, 0.01):
        pred = (p >= t).astype(int)
        tp, fp = int(np.sum((pred == 1) & (y == 1))), int(np.sum((pred == 1) & (y == 0)))
        # abstaining is only worth it if it prevents misses at acceptable cost
        fn = int(np.sum((p < t) & (y == 1)))
        c = abstention_cost * tp + cost_fp * fp + cost_fn * fn * 0.5
        if c < best_abst_c:
            best_abst_c, best_abst = c, float(t)
    retrieve_t = best_t
    clarify_t = max(best_abst, best_t + 0.05, 0.6)
    return ThresholdSearchResult(
        thresholds={
            "decision": round(best_t, 3),
            "retrieve": round(retrieve_t, 3),
            "clarify": round(min(clarify_t, 0.95), 3),
            "low": round(min(0.25, best_t * 0.6), 3),
            "moderate": round(best_t, 3),
            "high": round(min(clarify_t, 0.75), 3),
        },
        objective=best_c,
        details={"cost_fp": cost_fp, "cost_fn": cost_fn, "grid_size": int(len(grid))},
    )
