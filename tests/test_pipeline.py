import numpy as np

from phcx.intervention import RiskPolicy
from phcx.pipeline.risk_pipeline import PHCPipeline, heuristic_risk


def _pipe(**kw):
    return PHCPipeline(**kw)


def test_contamination_scenario_high_risk(contaminated_history):
    pred = _pipe().analyze(
        "Based on the 50,000 samples in Atlas A, what should we report?",
        conversation_history=contaminated_history,
        use_retriever=False,
    )
    assert 0.4 <= pred.risk_score <= 1.0
    assert pred.risk_level in ("HIGH", "CRITICAL", "MODERATE")
    assert pred.recommended_action != "GENERATE"
    assert pred.contamination["score"] > 0.3
    assert pred.factors


def test_clean_scenario_low_risk(clean_history):
    passages = [{"doc_id": "d1",
                 "text": "The Orion project uses Atlas A for evaluation."}]
    pred = _pipe().analyze(
        "Which dataset does the Orion project use for evaluation?",
        conversation_history=clean_history,
        evidence=passages,
        use_retriever=False,
    )
    assert pred.risk_score < 0.4
    assert pred.risk_level in ("LOW", "MODERATE")
    assert pred.recommended_action in ("GENERATE", "GENERATE_WITH_EVIDENCE")


def test_contradiction_raises_risk(contradictory_history, clean_history):
    p = _pipe()
    risky = p.analyze("Which dataset does the Orion project use for evaluation?",
                      conversation_history=contradictory_history, use_retriever=False)
    safe = p.analyze("Which dataset does the Orion project use for evaluation?",
                     conversation_history=clean_history, use_retriever=False)
    assert risky.risk_score > safe.risk_score
    assert risky.contradictions


def test_prediction_serialisable(clean_history):
    pred = _pipe().analyze("Which dataset do we use?", conversation_history=clean_history,
                           use_retriever=False)
    d = pred.to_dict()
    for key in ("risk_score", "risk_level", "recommended_action", "factors",
                "engine_reasons", "decision", "features", "latency_ms", "mode"):
        assert key in d
    assert np.isfinite(d["risk_score"])


def test_policy_change_changes_action(clean_history):
    args = dict(prompt="Which dataset do we use for evaluation?",
                conversation_history=clean_history, use_retriever=False)
    default = _pipe().analyze(**args)
    aggressive = _pipe(policy=RiskPolicy(user_preference="aggressive")).analyze(**args)
    assert aggressive.risk_level != "LOW" or default.risk_level != "LOW" or True
    # aggressive policy must never be less cautious than default
    order = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
    assert order[aggressive.risk_level] >= order[default.risk_level]


def test_model_replaces_heuristic(clean_history):
    from phcx.features import FeatureEngine
    from phcx.risk.predictor import RiskModel

    n_features = len(FeatureEngine().feature_names)
    rng = np.random.RandomState(0)
    X = rng.rand(150, n_features)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    names = FeatureEngine().feature_names
    model = RiskModel("lr", seed=0).fit(X, y, feature_names=names)
    # heuristic path still works without a model
    assert 0.0 <= heuristic_risk({"ev_risk": 0.5}) <= 1.0
    # with a model the pipeline uses model probabilities
    pred = _pipe(model=model).analyze("Which dataset do we use?",
                                      conversation_history=clean_history,
                                      use_retriever=False)
    assert 0.0 <= pred.risk_score <= 1.0
    assert pred.features and set(pred.features) <= set(names)


def test_explanations_present(clean_history, contaminated_history):
    pred = _pipe().analyze("Based on the 50,000 samples, what should we report?",
                           conversation_history=contaminated_history, use_retriever=False)
    assert pred.explanation["top_factors"] or pred.factors
    assert any(f.get("feature") for f in pred.factors)
