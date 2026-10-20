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
    """Monotone isotonic calibration with *linear* out-of-range extrapolation.

    ``sklearn``'s ``out_of_bounds="clip"`` maps every score below the
    calibration range to exactly 0.0.  On a held-out split that produced a
    degenerate mass of 30% of test rows pinned at 0.0, which destroys the
    ranking (ties) and makes the deployed risk score report 0 for a prompt
    with a detected unsupported presupposition.  Here the mapping is
    extrapolated linearly from the two nearest knots instead, and floored at a
    small positive value so a score is never reported as exactly zero.
    """

    method = "isotonic"
    #: smallest probability the calibrator is allowed to emit
    floor = 1e-6

    def __init__(self) -> None:
        self._iso = None
        self._knots_x: np.ndarray = np.zeros(0)
        self._knots_y: np.ndarray = np.zeros(0)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        p = np.asarray(p, dtype=float)
        if len(np.unique(p)) < 3 or len(np.unique(y)) < 2:
            return self
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        self._iso = iso
        # the fitted step function's knots, used for extrapolation
        xs = np.asarray(iso.X_thresholds_, dtype=float)
        ys = np.asarray(iso.y_thresholds_, dtype=float)
        self._knots_x = xs
        self._knots_y = ys
        return self

    def _extrapolate(self, x: np.ndarray) -> np.ndarray:
        """Linear extrapolation below the first / above the last knot."""
        out = np.empty_like(x)
        low = x < self._knots_x[0]
        high = x > self._knots_x[-1]
        mid = ~(low | high)
        if low.any():
            x0, x1 = self._knots_x[0], self._knots_x[1]
            y0, y1 = self._knots_y[0], self._knots_y[1]
            slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
            out[low] = y0 + slope * (x[low] - x0)
        if high.any():
            x0, x1 = self._knots_x[-2], self._knots_x[-1]
            y0, y1 = self._knots_y[-2], self._knots_y[-1]
            slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
            out[high] = y1 + slope * (x[high] - x0)
        if mid.any():
            out[mid] = self._iso.predict(x[mid])
        return out

    def transform(self, p: np.ndarray) -> np.ndarray:
        x = np.asarray(p, dtype=float)
        # a calibrator pickled by an older version may not carry the knots
        knots_x = getattr(self, "_knots_x", None)
        knots_y = getattr(self, "_knots_y", None)
        if self._iso is None or knots_x is None or len(knots_x) < 2:
            if self._iso is None:
                return np.clip(x, 0.0, 1.0)
            return np.clip(self._iso.transform(x), self.floor, 1.0)
        self._knots_x = np.asarray(knots_x, dtype=float)
        self._knots_y = np.asarray(knots_y, dtype=float)
        return np.clip(self._extrapolate(x), self.floor, 1.0)


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


def cross_fitted_calibration(
    p: np.ndarray,
    y: np.ndarray,
    method: str = "isotonic",
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Out-of-fold calibrated probabilities for a held-out set.

    Isotonic regression interpolates its training set exactly, so *in-sample*
    calibrated probabilities make a model look sharper than it is.  The
    threshold search must not be tuned on such memorised values, so the
    calibrator is refit per fold and each fold is scored by a calibrator that
    never saw it.  The final calibrator is still fit on the whole split and is
    what gets applied to the test set.
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    n = p.size
    n_folds = max(2, int(n_folds))
    if n < 2 * n_folds or len(np.unique(y)) < 2:
        cal = make_calibrator(method)
        cal.fit(p, y)
        return cal.transform(p)

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    out = np.empty(n, dtype=float)
    for fold in folds:
        hold = np.ones(n, dtype=bool)
        hold[fold] = False
        if hold.sum() == 0 or len(np.unique(y[hold])) < 2:
            out[fold] = p[fold]
            continue
        cal = make_calibrator(method)
        cal.fit(p[hold], y[hold])
        out[fold] = cal.transform(p[fold])
    return np.clip(out, 0.0, 1.0)


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


#: one-sided 95% normal quantile: the coverage constraint must hold with
#: 95% confidence on a fresh sample of the same size, not merely touch the
#: limit on the split it was fitted on.
_COVERAGE_Z = 1.645


def enforce_min_coverage(
    thresholds: Dict[str, float],
    scores: Sequence[float],
    min_coverage: float,
    margin_stderr: bool = True,
) -> Dict[str, Any]:
    """Raise the *blocking* cut-point so enough turns are still answered.

    The intervention policy blocks a turn once its risk reaches the HIGH
    cut-point, so coverage (the fraction of turns still answered) is the
    fraction of scores below that cut.  The cost objective optimises the
    separate *decision* cut-point; this constrains the blocking one, which is
    what the deployment gate is about.

    The requirement is imposed on the validation split with a one-sided 95%
    safety margin, so that it still holds on a fresh sample of the same
    size: the split is a sample of deployment traffic, and a constraint that
    merely touches its limit on one split is satisfied by luck rather than by
    construction.  The margin is the standard error of a binomial proportion
    at the limit, scaled by 1.645 (the one-sided 95% normal quantile).
    """
    s = np.asarray(scores, dtype=float).reshape(-1)
    thr = dict(thresholds)
    if s.size == 0 or min_coverage is None:
        return {"thresholds": thr, "coverage": None, "adjusted": False}

    required = float(min_coverage)
    if margin_stderr and s.size > 1:
        # one-sided 95% margin on a binomial proportion at the limit
        required += _COVERAGE_Z * float(np.sqrt(0.25 / s.size))

    high = float(thr.get("high", 0.75))
    coverage = float((s < high).mean())
    if coverage >= required:
        return {"thresholds": thr, "coverage": coverage, "required": required,
                "adjusted": False}

    # the smallest observed score value that keeps enough turns below the cut
    target = int(np.ceil(required * s.size))
    target = min(max(target, 1), s.size)
    vals = np.sort(s)
    uniq = np.unique(s)
    above = uniq[uniq > vals[target - 1]]
    new_high = float(above[0]) if above.size else 0.999
    new_high = max(new_high, float(thr.get("moderate", 0.5)))
    thr["high"] = round(min(new_high, 0.999), 4)
    achieved = float((s < float(thr["high"])).mean())
    return {"thresholds": thr, "coverage": achieved, "required": required,
            "adjusted": True}


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

    The deployment requirement that a minimum fraction of turns must still be
    answered constrains the *blocking* cut-point, not the decision cut-point
    optimised here; use :func:`enforce_min_coverage` for that.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    grid = np.arange(0.05, 0.96, 0.01)

    def _counts(t: float) -> Tuple[int, int, int]:
        pred = (p >= t).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        return tp, fp, fn

    def _f1(t: float) -> float:
        tp, fp, fn = _counts(t)
        denom = 2 * tp + fp + fn
        return (2.0 * tp / denom) if denom else 0.0

    # Evaluate the whole grid, then pick the operating point.  The cost
    # surface is frequently flat over a wide range of thresholds; scanning
    # upward with a strict comparison would pin the operating point to the
    # grid floor purely as an artifact of scan order.  Instead every
    # cost-equivalent threshold is collected, and the choice is the one with
    # the best F1, with proximity to the middle of the optimal region used to
    # break ties.  The cost objective itself is left untouched.
    costs: List[Tuple[float, float]] = []
    for t in grid:
        _, fp, fn = _counts(float(t))
        costs.append((float(t), float(cost_fp * fp + cost_fn * fn)))
    best_c = min(c for _, c in costs)
    candidates = sorted(t for t, c in costs if c <= best_c + 1e-9)
    mid = float(np.median(candidates))
    best_t = max(candidates, key=lambda t: (round(_f1(t), 6), -abs(t - mid)))

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
    # The threshold ladder must stay monotone: low <= moderate <= high <= clarify.
    # Without this clamp a high decision threshold (e.g. 0.90) produced
    # high = 0.75 < moderate, i.e. an empty and inverted HIGH band.
    moderate = float(best_t)
    clarify = float(max(min(clarify_t, 0.95), moderate))
    low = float(min(0.25, best_t * 0.6))
    high = float(max(min(clarify, 0.75), moderate))
    return ThresholdSearchResult(
        thresholds={
            "decision": round(moderate, 3),
            "retrieve": round(retrieve_t, 3),
            "clarify": round(clarify, 3),
            "low": round(low, 3),
            "moderate": round(moderate, 3),
            "high": round(high, 3),
        },
        objective=best_c,
        details={"cost_fp": cost_fp, "cost_fn": cost_fn, "grid_size": int(len(grid))},
    )
