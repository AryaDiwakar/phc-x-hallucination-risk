import pytest

from phcx.dataset import PHCConversationGenerator


@pytest.fixture(scope="session")
def tiny_rows():
    gen = PHCConversationGenerator(samples_per_category=4, seed=7, n_families=40,
                                   long_context_extra_turns=(6, 9))
    return gen.generate()


@pytest.fixture()
def clean_history():
    return [
        {"role": "user", "content": "Tell me about the Orion project."},
        {"role": "assistant", "content": "The Orion project works on route planning.",
         "source": "doc:orion-spec", "verified": True},
        {"role": "user", "content": "Which dataset do we use?"},
        {"role": "assistant", "content": "The Orion project uses Atlas A for evaluation.",
         "source": "doc:orion-spec", "verified": True},
    ]


@pytest.fixture()
def contaminated_history():
    return [
        {"role": "user", "content": "How big is the Atlas A dataset?"},
        {"role": "assistant", "content": "The Atlas A dataset contains 50,000 samples."},
        {"role": "user", "content": "Thanks, noted."},
    ]


@pytest.fixture()
def contradictory_history():
    return [
        {"role": "user", "content": "Which dataset does the Orion project use?"},
        {"role": "assistant", "content": "The Orion project uses Atlas A for evaluation."},
        {"role": "user", "content": "Continue please."},
        {"role": "assistant", "content": "The Orion project uses Atlas B for evaluation."},
    ]
