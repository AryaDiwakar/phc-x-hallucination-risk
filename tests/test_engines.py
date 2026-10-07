from phcx.ambiguity import AmbiguityEngine
from phcx.contradiction import ContradictionEngine
from phcx.entity import EntityTracker
from phcx.memory import ConversationMemory
from phcx.presupposition import PresuppositionEngine
from phcx.temporal import TemporalEngine


def test_contradiction_detects_conflicting_history(contradictory_history):
    mem = ConversationMemory.from_history(contradictory_history)
    res = ContradictionEngine().analyze(mem, "Which dataset does the Orion project use?")
    assert res.score >= 0.5
    assert res.pairs


def test_contradiction_clean_history(clean_history):
    mem = ConversationMemory.from_history(clean_history)
    res = ContradictionEngine().analyze(mem, "Which dataset does the Orion project use?")
    assert res.score < 0.5


def test_entity_drift_replacement():
    history = [
        {"role": "user", "content": "We use Atlas A for evaluation."},
        {"role": "assistant", "content": "Atlas A is our evaluation dataset."},
        {"role": "user", "content": "And now?"},
        {"role": "assistant", "content": "We switched everything to Nimbus B."},
    ]
    mem = ConversationMemory.from_history(history)
    res = EntityTracker().track(mem, "Which dataset should we benchmark on?")
    assert res.score > 0.0


def test_temporal_staleness_detected():
    history = [
        {"role": "user", "content": "What version are we on?"},
        {"role": "assistant", "content": "We are on PyTorch with the configuration from 2019."},
    ]
    mem = ConversationMemory.from_history(history, )
    # give the memory an old timestamp so staleness triggers
    for t in mem.turns:
        t.timestamp = 0.0
    res = TemporalEngine().analyze(mem, "What is the current version of our tooling?")
    assert res.score > 0.0
    # fresh evidence lowers the temporal risk
    res2 = TemporalEngine().analyze(
        mem, "What is the current version of our tooling?",
        fresh_evidence=["PyTorch 6.1 released in 2026."],
    )
    assert res2.score <= res.score


def test_presupposition_unsupported_change():
    prompt = "Why did accuracy improve from 70.0% to 85.0%?"
    res = PresuppositionEngine().analyze(prompt, ConversationMemory.from_history([]),
                                         evidence_texts=[])
    assert res.score > 0.3
    assert res.unsupported_count >= 1


def test_presupposition_supported_by_evidence():
    prompt = "Why did accuracy improve from 70.0% to 85.0%?"
    res = PresuppositionEngine().analyze(
        prompt, ConversationMemory.from_history([]),
        evidence_texts=["Accuracy improved from 70.0% to 85.0% after data cleaning."],
    )
    assert res.score < 0.3


def test_ambiguity_pronoun_two_antecedents():
    history = [
        {"role": "user", "content": "The Orion service depends on BERT."},
        {"role": "assistant", "content": "Noted, BERT serves the Orion service."},
        {"role": "user", "content": "And the batch job?"},
        {"role": "assistant", "content": "The batch job depends on RoBERTa."},
    ]
    mem = ConversationMemory.from_history(history)
    res = AmbiguityEngine().analyze("Update that model's configuration and report it.", mem)
    assert res.score > 0.0
    assert res.references or res.should_clarify
