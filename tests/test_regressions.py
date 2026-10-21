"""Regression tests for bugs found in the correctness sweep.

Each test pins a specific defect that was fixed, so the behaviour cannot
silently regress back.
"""
import numpy as np
import pytest

from phcx.calibration.core import optimize_thresholds
from phcx.entity import EntityTracker
from phcx.experiments.train import groups_from_config
from phcx.explain import Explainer
from phcx.features import FeatureEngine
from phcx.intervention import ACTIONS, RiskPolicy
from phcx.memory import ConversationMemory
from phcx.models.adapters import ModelAdapter
from phcx.pipeline import PHCPipeline
from phcx.risk import RiskModel
from phcx.temporal.engine import _FRESHNESS_RE, extract_temporal
from phcx.utils.text import entropy


# --------------------------------------------------------------- temporal
def test_temporal_expressions_use_word_boundaries():
    """'now' inside 'knowledge' and 'then' inside 'strengthened' are not dates."""
    assert extract_temporal("The knowledge graph is larger.") == []
    assert extract_temporal("We strengthened the model.") == []
    assert extract_temporal("The snowball melted.") == []
    # genuine occurrences are still detected
    exprs = extract_temporal("What is the current version right now?")
    assert {e["expr"] for e in exprs} >= {"current", "right now"}


def test_freshness_cues_use_word_boundaries():
    assert not _FRESHNESS_RE.search("The snowball melted.")
    assert _FRESHNESS_RE.search("Please update the model now.")


# --------------------------------------------------------------- calibration
def test_isotonic_calibrator_does_not_collapse_to_zero():
    """Out-of-range scores must not be clipped to exactly 0.0.

    sklearn's ``out_of_bounds="clip"`` pinned 30% of held-out rows at 0.0,
    which destroyed the ranking and reported a risk of 0 for prompts with a
    detected unsupported presupposition.
    """
    from phcx.calibration.core import IsotonicCalibrator

    p = np.concatenate([np.linspace(0.05, 0.45, 40), np.linspace(0.55, 0.95, 40)])
    y = (p >= 0.5).astype(int)
    cal = IsotonicCalibrator()
    cal.fit(p, y)
    out = cal.transform(np.array([0.01, 0.20, 0.50, 0.90]))
    assert out.min() > 0.0, "a score was mapped to exactly zero"
    assert np.all(np.diff(out) >= -1e-9), "the map is not monotone"


def test_enforce_min_coverage_raises_the_blocking_cut():
    """The coverage requirement must be enforced on the blocking cut-point."""
    from phcx.calibration.core import enforce_min_coverage

    scores = np.linspace(0.0, 1.0, 100)
    thr = {"decision": 0.4, "moderate": 0.4, "low": 0.2, "high": 0.5, "clarify": 0.9}
    out = enforce_min_coverage(thr, scores, 0.60)
    assert out["adjusted"] is True
    assert out["coverage"] >= 0.60
    assert out["thresholds"]["high"] > thr["high"]
    # already satisfied -> untouched
    out2 = enforce_min_coverage(dict(thr), scores, 0.10)
    assert out2["adjusted"] is False
    assert out2["thresholds"]["high"] == thr["high"]


def test_presupposition_engine_detects_question_form_factives():
    """"Isn't it true that X" presupposes X."""
    from phcx.presupposition import PresuppositionEngine

    props = PresuppositionEngine().extract(
        "Isn't it true that Paris doubled its population in 2025?"
    )
    assert props
    assert any(p.kind == "factive" for p in props)
    # a plain question must not trigger anything
    assert PresuppositionEngine().extract("What is the accuracy of Atlas A?") == []


def test_optimize_thresholds_returns_monotone_ladder():
    """A high decision threshold must not invert the HIGH band."""
    neg = np.linspace(0.80, 0.90, 40)
    pos = np.linspace(0.92, 0.98, 40)
    p = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros(40), np.ones(40)])
    t = optimize_thresholds(p, y, cost_fp=1.0, cost_fn=3.0).thresholds
    assert t["low"] <= t["moderate"] <= t["high"] <= t["clarify"]
    assert 0.0 < t["decision"] <= 1.0


def test_optimize_thresholds_not_pinned_to_grid_floor():
    """A flat cost region must not pin the operating point to the grid floor.

    The cost surface is flat over a wide range here; the search used to pick
    the lowest threshold purely because it scanned upwards with a strict
    comparison, which also made the operating point seed-dependent.
    """
    neg = np.linspace(0.02, 0.20, 60)
    pos = np.linspace(0.25, 0.90, 60)
    p = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros(60), np.ones(60)])
    t = optimize_thresholds(p, y, cost_fp=1.0, cost_fn=3.0).thresholds
    assert t["decision"] > 0.05, "threshold collapsed to the grid floor"
    assert t["decision"] < 0.95, "threshold collapsed to the grid ceiling"


def test_entropy_is_in_nats():
    """Entropy must use the natural log so it matches the model-logit entropy."""
    assert abs(entropy([0.5, 0.5]) - np.log(2.0)) < 1e-9


# --------------------------------------------------------------- policy
def test_abstain_is_reachable_under_aggressive_preference():
    """ABSTAIN must be emittable; the balanced default is unchanged."""
    assert "ABSTAIN" in ACTIONS
    assert RiskPolicy().decide(0.90).action == "CLARIFY_OR_ABSTAIN"
    assert RiskPolicy(user_preference="aggressive").decide(0.90).action == "ABSTAIN"
    assert RiskPolicy().decide(0.90, prefer_clarify=True).action == "CLARIFY"


def test_latency_budget_downgrades_the_action():
    """Exceeding the latency budget must downgrade to a cheaper action."""
    p = RiskPolicy(latency_budget_ms=0.0)
    d = p.decide(0.90, latency_ms=150.0)
    assert d.level == "HIGH"
    assert d.action == "RETRIEVE"
    # within budget the critical action is kept
    d2 = RiskPolicy(latency_budget_ms=800).decide(0.90, latency_ms=150.0)
    assert d2.level == "CRITICAL"


# --------------------------------------------------------------- explainer
def test_sklearn_mlp_explanation_is_not_empty():
    """sklearn MLP has no feature_importances_; the gradient path must work."""
    names = FeatureEngine().feature_names
    rng = np.random.default_rng(0)
    X = rng.random((120, len(names)))
    y = (X[:, 0] + 0.5 * X[:, 3] > 0.9).astype(int)
    model = RiskModel(name="mlp", seed=1)
    model.fit(X, y, names)
    ex = Explainer(model=model, feature_names=names)
    out = ex.explain(X[0], {n: float(X[0][i]) for i, n in enumerate(names)})
    assert out.top_factors, "MLP explanation produced no factors"
    assert any(abs(f.contribution) > 0 for f in out.top_factors)


# --------------------------------------------------------------- entity
def test_entity_tracker_ignores_common_capitalised_words():
    """'Okay' / 'Summarize' must not be typed as dataset entities."""
    hist = [
        {"role": "user", "content": "We use Corpus A for training."},
        {"role": "assistant", "content": "Okay, Corpus A is the training corpus."},
    ]
    res = EntityTracker().track(ConversationMemory.from_history(hist), "Summarize Corpus A")
    assert res.replacements == []
    assert res.score == 0.0


def test_entity_tracker_reports_role_ambiguity():
    """A definite role reference with >1 candidate must be flagged."""
    hist = [
        {"role": "user", "content": "We use Atlas A for training."},
        {"role": "assistant", "content": "Atlas A is the training dataset."},
        {"role": "user", "content": "We also prepared Bench B for evaluation."},
        {"role": "assistant", "content": "Bench B is the evaluation dataset."},
    ]
    res = EntityTracker().track(ConversationMemory.from_history(hist), "Summarize the dataset")
    assert res.ambiguous_references
    assert res.ambiguous_references[0]["n_candidates"] == 2
    assert res.score > 0.0


def test_entity_tracker_merges_qualified_mentions():
    """'Summarize Corpus B' in the prompt is the same entity as 'Corpus B'."""
    hist = [
        {"role": "user", "content": "We switched to Corpus B for evaluation."},
        {"role": "assistant", "content": "Corpus B is the evaluation corpus."},
    ]
    res = EntityTracker().track(ConversationMemory.from_history(hist), "Summarize Corpus B")
    names = [e["entity"] for r in res.replacements for e in r["entities"]]
    assert "summarize corpus b" not in names


# --------------------------------------------------------------- memory
def test_derived_from_edges_require_shared_significant_numbers():
    """Two sentences sharing only common words must not create a graph edge."""
    mem = ConversationMemory.from_history([
        {"role": "user", "content": "The model is ready for the run."},
        {"role": "assistant", "content": "The model is ready for the job."},
    ])
    derived = [e for e in mem.edges if e.edge_type.value == "derived-from"]
    assert derived == []

    mem2 = ConversationMemory.from_history([
        {"role": "user", "content": "Atlas A contains 50,000 samples."},
        {"role": "assistant", "content": "Atlas A contains 50,000 samples."},
    ])
    derived2 = [e for e in mem2.edges if e.edge_type.value == "derived-from"]
    assert derived2, "a reused numeric fact should create a derived-from edge"


# --------------------------------------------------------------- features
def test_use_flags_are_honoured():
    """`features.use_*` switches must actually change the active groups."""
    cfg = {"features": {"mode": "black_box"}}
    assert groups_from_config(cfg) == [
        "prompt", "history", "provenance", "consistency", "evidence"
    ]
    cfg["features"]["use_prompt"] = False
    assert "prompt" not in groups_from_config(cfg)
    cfg["features"]["use_temporal"] = False
    assert "consistency" not in groups_from_config(cfg)
    cfg["features"]["use_whitebox"] = True
    assert "model" in groups_from_config(cfg)


def test_prompt_has_negation_detects_contractions():
    from phcx.features.engine import FeatureContext
    ctx = FeatureContext(prompt="We don't have the numbers yet.",
                          memory=ConversationMemory())
    out = FeatureEngine()._prompt(ctx)
    assert out["prompt_has_negation"] == 1.0


# ------------------------------------------------------------- ambiguity
def test_ambiguity_does_not_flag_a_reference_that_matches_nothing():
    """'the held-out set' shares no content word with the conversation.

    The hashed n-gram similarity of two short unrelated strings is dominated
    by hash collisions, which used to make an unmatched reference look like
    an ambiguous one.
    """
    from phcx.ambiguity import AmbiguityEngine

    history = [
        {"role": "user", "content": "We use Atlas A for evaluation."},
        {"role": "assistant", "content": "Understood, Atlas A it is."},
    ]
    mem = ConversationMemory.from_history(history)
    res = AmbiguityEngine().analyze(
        "Summarize the Atlas A results on the held-out set.", mem)
    flagged = {r["reference"].lower() for r in res.references}
    assert "the held-out set" not in flagged
    assert res.score < 0.5, "an unmatched reference was scored as ambiguous"


def test_ambiguity_reference_keeps_hyphenated_words_intact():
    """The captured reference must be the whole noun phrase."""
    from phcx.ambiguity.engine import _NP_REF_RE, _TRAILING_STOP

    def refs(text):
        out = []
        for m in _NP_REF_RE.finditer(text):
            words = m.group(0).strip().split()
            while len(words) > 1 and words[-1].lower() in _TRAILING_STOP:
                words.pop()
            if len(words) >= 2:
                out.append(" ".join(words))
        return out

    got = refs("Summarize the Atlas A results on the held-out set.")
    assert "the held-out set" in got
    assert "the held" not in got
    # a trailing function word is not part of the noun phrase
    assert "that model configuration" in refs("Update that model configuration and report it.")
    assert not any(r.endswith(" and") for r in refs("Update that model configuration and report it."))


def test_history_window_keeps_size_features_on_the_full_conversation():
    """`history_turns` bounds content analysis, not the conversation size."""
    hist = [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": f"Turn {i} discusses the Atlas A training corpus."}
        for i in range(20)
    ]
    p = PHCPipeline(history_turns=5)
    pred = p.analyze("Summarize the Atlas A corpus results", conversation_history=hist)
    assert pred.features["history_turn_count"] == 20.0
    assert pred.features["history_claim_count"] == 20.0


# --------------------------------------------------------------- adapters
class _StubWhiteBoxAdapter(ModelAdapter):
    name = "stub"
    whitebox = True

    def available(self):
        return True

    def generate(self, prompt, history=None, max_tokens=8):
        return "stub"

    def whitebox_signals(self, prompt, history=None):
        return {}


def test_black_box_adapter_must_not_fake_white_box_signals():
    from phcx.models.adapters import ScriptedAdapter
    assert ScriptedAdapter().whitebox_signals("hello", []) == {}
    assert _StubWhiteBoxAdapter().whitebox is True
