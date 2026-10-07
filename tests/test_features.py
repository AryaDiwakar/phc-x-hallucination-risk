import numpy as np

from phcx.features import ABLATION_GROUPS, FEATURE_GROUPS, FeatureContext, FeatureEngine
from phcx.features.evidence import EvidenceEngine
from phcx.memory import ConversationMemory
from phcx.retrieval.local import RetrievedPassage


def _ctx(clean_history, contaminated_history):
    prompt = "Based on the 50,000 samples in Atlas A, what should we report?"
    mem = ConversationMemory.from_history(contaminated_history)
    ev = EvidenceEngine().analyze(prompt, mem, [])
    return FeatureContext(
        prompt=prompt,
        memory=mem,
        contamination_score=mem.contamination(prompt).score,
        contamination=mem.contamination(prompt),
        contradiction=None,
        entity=None,
        temporal=None,
        presupposition=None,
        ambiguity=None,
        evidence=ev,
        whitebox={},
    )


def test_feature_groups_cover_spec_sections():
    expected = {"prompt", "history", "provenance", "consistency", "evidence", "model"}
    assert expected == set(FEATURE_GROUPS)


def test_feature_names_stable_and_unique():
    eng = FeatureEngine()
    names = eng.feature_names
    assert len(names) == len(set(names))
    assert all(isinstance(n, str) for n in names)


def test_extract_returns_all_features(clean_history, contaminated_history):
    eng = FeatureEngine()
    feats = eng.extract(_ctx(clean_history, contaminated_history))
    for name in eng.feature_names:
        assert name in feats
        assert np.isfinite(feats[name])
    assert feats["prov_contamination"] > 0.3


def test_empty_history_has_no_semantic_drift():
    feats = FeatureEngine().extract(_ctx([], []))
    assert feats["history_turn_count"] == 0
    assert feats["history_semantic_distance"] == 0.0


def test_ablation_groups_are_subsets():
    all_groups = set(FEATURE_GROUPS)
    for name, groups in ABLATION_GROUPS.items():
        assert set(groups) <= all_groups, name
    assert ABLATION_GROUPS["G_full_whitebox"] == sorted(
        ABLATION_GROUPS["G_full_whitebox"], key=lambda g: list(FEATURE_GROUPS).index(g)
    )


def test_group_selection_changes_feature_set():
    full = FeatureEngine(groups=list(FEATURE_GROUPS)).feature_names
    no_wb = FeatureEngine(groups=[g for g in FEATURE_GROUPS if g != "model"]).feature_names
    assert set(no_wb) < set(full)


def test_evidence_engine_with_passages(clean_history):
    prompt = "Which dataset does the Orion project use for evaluation?"
    mem = ConversationMemory.from_history(clean_history)
    passages = [RetrievedPassage(doc_id="d1",
                                 text="The Orion project uses Atlas A for evaluation.",
                                 score=0.9)]
    ev = EvidenceEngine().analyze(prompt, mem, passages)
    assert ev.available
    assert ev.coverage_score > 0.3
    assert ev.score < 0.5  # good evidence should not look risky


def test_evidence_engine_without_evidence():
    prompt = "What is the budget of the Orion project?"
    mem = ConversationMemory.from_history([])
    ev = EvidenceEngine().analyze(prompt, mem, [])
    assert not ev.available
    assert ev.coverage_score == 0.0
