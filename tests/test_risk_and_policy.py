import numpy as np
import pytest

from phcx.calibration import (
    brier_score,
    expected_calibration_error,
    make_calibrator,
    nll,
    optimize_thresholds,
    reliability_bins,
)
from phcx.intervention import RiskPolicy, risk_level
from phcx.pipeline.risk_pipeline import heuristic_risk
from phcx.risk.predictor import RiskModel


def test_heuristic_risk_bounds_and_monotonicity():
    base = {"ev_risk": 0.0, "ev_cov": 0.9}
    assert 0.0 <= heuristic_risk(base) <= 1.0
    low = heuristic_risk({**base, "cons_contradiction_score": 0.1})
    high = heuristic_risk({**base, "cons_contradiction_score": 0.9})
    assert high > low
    # contamination increases risk
    c_high = heuristic_risk({**base, "prov_contamination": 0.9})
    assert c_high > heuristic_risk(base)


def test_risk_level_thresholds():
    assert risk_level(0.10) == "LOW"
    assert risk_level(0.30) == "MODERATE"
    assert risk_level(0.60) == "HIGH"
    assert risk_level(0.90) == "CRITICAL"


def test_policy_actions():
    p = RiskPolicy()
    assert p.decide(0.10, evidence_available=True).action == "GENERATE"
    assert p.decide(0.90, evidence_available=False).action in ("RETRIEVE", "CLARIFY_OR_ABSTAIN")
    d = p.decide(0.30, evidence_available=True).action
    assert d in ("GENERATE_WITH_EVIDENCE", "RETRIEVE")
    clarify = p.decide(0.30, evidence_available=False, prefer_clarify=True)
    assert clarify.action in ("CLARIFY_OR_ABSTAIN", "RETRIEVE")


def test_policy_logging():
    p = RiskPolicy()
    p.decide(0.8, conversation_id="c1")
    assert p.log() and p.log()[-1]["conversation_id"] == "c1"


def test_policy_from_yaml(tmp_path):
    import yaml

    cfg = {"thresholds": {"moderate": 0.2, "high": 0.4, "critical": 0.6},
           "overrides": {"user_preference": "aggressive", "latency_budget_ms": 250}}
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(cfg))
    p = RiskPolicy.from_yaml(str(path))
    assert p.thresholds["high"] == 0.4
    assert p.user_preference == "aggressive"
    assert p.latency_budget_ms == 250


def test_calibration_metrics():
    p = np.array([0.1, 0.4, 0.6, 0.9])
    y = np.array([0, 0, 1, 1])
    assert 0.0 <= expected_calibration_error(p, y) <= 1.0
    assert brier_score(p, y) >= 0.0
    assert nll(p, y) >= 0.0
    bins = reliability_bins(p, y, n_bins=4)
    assert bins and all(0 <= b["count"] for b in bins)


@pytest.mark.parametrize("method", ["platt", "isotonic", "temperature", "identity"])
def test_calibrators_fit_transform(method):
    rng = np.random.RandomState(0)
    logits = rng.normal(size=200)
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.rand(200) < p).astype(int)
    cal = make_calibrator(method).fit(p, y)
    out = cal.transform(p)
    assert out.shape == p.shape
    assert np.all((out >= 0) & (out <= 1))
    if method != "identity":
        assert brier_score(out, y) <= brier_score(p, y) + 1e-6


def test_optimize_thresholds():
    rng = np.random.RandomState(1)
    p = rng.rand(300)
    y = (rng.rand(300) < p).astype(int)
    res = optimize_thresholds(p, y)
    assert isinstance(res.thresholds, dict)
    assert 0.0 < res.thresholds["decision"] <= 1.0
    assert res.objective >= 0.0


def test_risk_model_train_predict_save(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.rand(200, 8)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    names = [f"f{i}" for i in range(8)]
    m = RiskModel("lr", seed=0).fit(X, y, feature_names=names)
    proba = m.predict_proba(X)
    assert proba.shape == (200,)
    assert np.all((proba >= 0) & (proba <= 1))
    # separable problem should learn something
    assert np.corrcoef(proba, y)[0, 1] > 0.5
    path = tmp_path / "model.joblib"
    m.save(path)
    m2 = RiskModel.load(path)
    assert np.allclose(m2.predict_proba(X), proba)


def test_risk_model_calibrated_probas():
    rng = np.random.RandomState(2)
    X = rng.rand(300, 5)
    y = (X[:, 0] > 0.5).astype(int)
    m = RiskModel("rf", seed=0).fit(X, y, X_val=X[:60], y_val=y[:60])
    proba = m.predict_proba(X, calibrated=True)
    assert np.all((proba >= 0) & (proba <= 1))
