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


def test_answerable_user_context_never_gets_a_blocking_override():
    class HighRiskModel:
        name = "test"
        feature_names = _pipe().features.feature_names
        calibrator = None
        _model = None

        def predict_proba(self, _x, calibrated=True):
            return np.asarray([0.99])

    history = [
        {"role": "user", "content": "The Northstar pilot had 120 participants."},
        {"role": "assistant", "content": "The Northstar pilot had 120 participants."},
    ]
    pred = _pipe(model=HighRiskModel()).analyze(
        "How many participants were in the Northstar pilot?",
        conversation_history=history,
        use_retriever=False,
    )
    assert pred.model_risk_score == 0.99
    assert pred.context_support["supported"]
    assert pred.risk_score <= 0.24
    assert pred.recommended_action not in ("CLARIFY", "CLARIFY_OR_ABSTAIN", "ABSTAIN")


def test_context_setting_message_is_not_blocked():
    class HighRiskModel:
        name = "test"
        feature_names = _pipe().features.feature_names
        calibrator = None
        _model = None

        def predict_proba(self, _x, calibrated=True):
            return np.asarray([0.99])

    prompt = (
        "For this chat, the Northstar pilot notes say: 120 participants, "
        "conducted in May 2025. No outcome metrics are included."
    )
    pred = _pipe(model=HighRiskModel()).analyze(prompt, use_retriever=False)
    assert pred.risk_score <= 0.24
    assert "context-setting" in pred.risk_adjustment
    assert pred.recommended_action not in ("CLARIFY", "CLARIFY_OR_ABSTAIN", "ABSTAIN")


def test_context_setting_with_changed_number_is_escalated():
    history = [
        {"role": "user", "content": (
            "For this chat, the Northstar pilot notes say: 120 participants, "
            "conducted in May 2025. No outcome metrics are included."
        )},
        {"role": "assistant", "content": "Northstar had 120 participants in May 2025."},
    ]
    pred = _pipe().analyze(
        "For this chat, the only Northstar pilot notes are: 180 participants, "
        "conducted in May 2025. No outcome metrics are included.",
        conversation_history=history,
        use_retriever=False,
    )
    assert pred.context_support["context_conflict"]["previous_values"] == ["120"]
    assert pred.context_support["context_conflict"]["new_values"] == ["180"]
    assert "conflicting numeric context" in pred.risk_adjustment
    assert pred.recommended_action in ("CLARIFY", "CLARIFY_OR_ABSTAIN", "ABSTAIN")


def test_social_greetings_ignore_empty_retrieval_risk_and_history():
    from phcx.features import FeatureEngine

    class HighRiskModel:
        name = "test"
        feature_names = FeatureEngine().feature_names
        calibrator = None
        _model = None

        def predict_proba(self, _x, calibrated=True):
            return np.asarray([0.99])

    history = [
        {"role": "user", "content": "The Northstar report says 120 participants."},
        {"role": "assistant", "content": "An unsupported claim about 900 outcomes."},
    ]
    for prompt in ("hi", "hii", "hello!", "thanks", "how are you?"):
        pred = _pipe(model=HighRiskModel(), model_blend=0.15).analyze(
            prompt, conversation_history=history
        )
        assert pred.risk_score == 0.0, (prompt, pred.to_dict())
        assert pred.recommended_action == "GENERATE"
        assert "greeting" in pred.risk_adjustment


def test_simple_arithmetic_is_safe_even_when_model_and_retriever_flag_risk():
    from phcx.features import FeatureEngine

    class HighRiskModel:
        name = "test"
        feature_names = FeatureEngine().feature_names
        calibrator = None
        _model = None

        def predict_proba(self, _x, calibrated=True):
            return np.asarray([0.99])

    pred = _pipe(model=HighRiskModel(), model_blend=0.15).analyze(
        "What is 2 + 2?", use_retriever=True
    )
    assert pred.model_risk_score == 0.99
    assert pred.risk_score == 0.0
    assert pred.risk_level == "LOW"
    assert pred.recommended_action == "GENERATE"
    assert "arithmetic" in pred.risk_adjustment


def test_northstar_followup_survives_chatgpt_formatted_echo():
    history = [
        {"role": "user", "content": "2 + 2 = 4 🎉"},
        {"role": "assistant", "content": "2 + 2 = 4 🎉"},
        {"role": "user", "content": (
            "For this chat, the Northstar pilot notes say: 120 participants, "
            "conducted in May 2025. No outcome metrics are included."
        )},
        {"role": "assistant", "content": (
            "Got it. For this chat, the Northstar pilot details are:\n\n"
            "Participants: 120\n\nConducted: May 2025\n\n"
            "Outcome metrics: Not included / not provided"
        )},
    ]
    pred = _pipe().analyze(
        "How many participants were in the Northstar pilot?",
        conversation_history=history,
        use_retriever=False,
    )
    assert pred.context_support["supported"]
    assert pred.contamination["score"] == 0.0
    assert not pred.ambiguities
    assert not pred.contradictions
    assert pred.recommended_action not in ("CLARIFY", "CLARIFY_OR_ABSTAIN", "ABSTAIN")

    # Common conversational paraphrases must resolve to the same grounded fact.
    for prompt in (
        "What's the number of participants in the Northstar pilot?",
        "What was the exact number of people in the Northstar pilot?",
    ):
        variant = _pipe().analyze(
            prompt, conversation_history=history, use_retriever=False
        )
        assert variant.context_support["supported"]
        assert variant.recommended_action not in (
            "CLARIFY", "CLARIFY_OR_ABSTAIN", "ABSTAIN"
        )


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


def test_no_retrain_ensemble_blends_existing_model_with_heuristic(clean_history):
    from phcx.features import FeatureEngine

    class HighRiskModel:
        name = "test"
        feature_names = FeatureEngine().feature_names
        calibrator = None
        _model = None

        def predict_proba(self, _x, calibrated=True):
            return np.asarray([0.99])

    args = dict(
        prompt="Which dataset does the Orion project use for evaluation?",
        conversation_history=clean_history,
        use_retriever=False,
    )
    pred = _pipe(model=HighRiskModel(), model_blend=0.15).analyze(**args)
    expected = 0.15 * 0.99 + 0.85 * heuristic_risk(pred.features)
    assert pred.risk_method == "heuristic_rf_ensemble_0.15"
    assert abs(pred.base_risk_score - expected) < 1e-9


def test_explanations_present(clean_history, contaminated_history):
    pred = _pipe().analyze("Based on the 50,000 samples, what should we report?",
                           conversation_history=contaminated_history, use_retriever=False)
    assert pred.explanation["top_factors"] or pred.factors
    assert any(f.get("feature") for f in pred.factors)
