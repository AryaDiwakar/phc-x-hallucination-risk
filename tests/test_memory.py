from phcx.memory import ConversationMemory, ProvenanceType, VerificationStatus
from phcx.provenance.classifier import classify_provenance, provenance_summary


def test_from_history_builds_claims(clean_history):
    mem = ConversationMemory.from_history(clean_history)
    assert len(mem.turns) == 4
    assert len(mem.all_claims) >= 3


def test_provenance_rules(clean_history, contaminated_history):
    mem = ConversationMemory.from_history(clean_history)
    verified = [c for c in mem.all_claims if c.source]
    assert verified, "sourced turns must keep their source"
    assert all(c.provenance_type == ProvenanceType.SOURCE_VERIFIED for c in verified)
    assert all(c.trust >= 0.45 for c in verified)

    mem2 = ConversationMemory.from_history(contaminated_history)
    model_claims = mem2.model_claims
    assert model_claims, "assistant claims without a source are MODEL_GENERATED"
    assert all(c.is_unverified for c in model_claims)


def test_user_assertions_are_not_model_claims(clean_history):
    mem = ConversationMemory.from_history(clean_history)
    users = mem.claims_by_speaker("user")
    assert users
    assert all(not c.is_model_claim for c in users)


def test_contamination_scores(contaminated_history, clean_history):
    prompt = "Based on the 50,000 samples in Atlas A, what should we report?"
    mem = ConversationMemory.from_history(contaminated_history)
    res = mem.contamination(prompt)
    assert res.score > 0.3
    assert res.dependent_claims
    assert "UNVERIFIED" in " ".join(res.reasons + res.explanations)

    clean_mem = ConversationMemory.from_history(clean_history)
    clean_prompt = "Which dataset does the Orion project use for evaluation?"
    assert clean_mem.contamination(clean_prompt).score < 0.2


def test_assistant_paraphrase_of_user_fact_is_not_contamination():
    history = [
        {"role": "user", "content": "The Northstar pilot had 120 participants."},
        {"role": "assistant", "content": "The Northstar pilot had 120 participants."},
    ]
    mem = ConversationMemory.from_history(history)
    result = mem.contamination("How many participants were in the Northstar pilot?")
    assert result.score == 0.0
    assert not result.dependent_claims


def test_assistant_addition_not_supported_by_user_stays_contamination():
    history = [
        {"role": "user", "content": "The Northstar pilot had 120 participants."},
        {"role": "assistant", "content": "The Northstar pilot had 180 participants."},
    ]
    mem = ConversationMemory.from_history(history)
    result = mem.contamination("How many participants were in the Northstar pilot?")
    assert result.score > 0.0


def test_natural_assistant_paraphrase_matches_full_user_source_turn():
    history = [
        {"role": "user", "content": (
            "For this chat, the Northstar pilot notes say: 120 participants, "
            "conducted in May 2025. No outcome metrics are included."
        )},
        {"role": "assistant", "content": (
            "The notes say the Northstar pilot had 120 participants and ran in May 2025."
        )},
    ]
    mem = ConversationMemory.from_history(history)
    result = mem.contamination("How many participants were in the Northstar pilot?")
    assert result.score == 0.0
    assert not result.dependent_claims


def test_relevant_claims(clean_history):
    mem = ConversationMemory.from_history(clean_history)
    rel = mem.relevant_claims("Which dataset do we use for evaluation?", top_k=3)
    assert rel and rel[0][1] > 0.1


def test_memory_serialisation_roundtrip(clean_history):
    mem = ConversationMemory.from_history(clean_history, conversation_id="c1")
    restored = ConversationMemory.from_dict(mem.to_dict())
    assert restored.conversation_id == "c1"
    assert len(restored.all_claims) == len(mem.all_claims)
    assert restored.summary()["total_claims"] == mem.summary()["total_claims"]


def test_provenance_summary(clean_history):
    mem = ConversationMemory.from_history(clean_history)
    s = provenance_summary(mem.all_claims)
    assert s["total_claims"] == len(mem.all_claims)
    assert 0.0 <= s["mean_trust"] <= 1.0


def test_classify_provenance_never_trusts_bare_model_claims():
    from phcx.memory.types import Claim, Turn

    turn = Turn(turn_id=1, speaker="assistant", text="The answer is 42 percent.")
    claim = Claim.create(text=turn.text, turn_id=1, speaker="assistant")
    out = classify_provenance(turn, claim)
    assert out.provenance_type in (ProvenanceType.MODEL_GENERATED, ProvenanceType.MODEL_INFERRED)
    assert out.verification_status in (VerificationStatus.UNVERIFIED, VerificationStatus.UNKNOWN)
