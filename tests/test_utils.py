import math

from phcx.utils.io import load_json, read_jsonl, save_json, write_jsonl
from phcx.utils.seeded import seeded_rng
from phcx.utils.text import (
    content_tokens,
    entropy,
    jaccard,
    negation_present,
    sent_split,
    similarity,
    tokenize,
)


def test_tokenize_and_sentences():
    assert tokenize("Hello, world!") == ["hello", "world"]
    assert sent_split("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert content_tokens("the quick brown fox") == ["quick", "brown", "fox"]


def test_similarity_properties():
    a = "The project uses Atlas A for evaluation"
    assert similarity(a, a) == 1.0
    assert 0.0 <= similarity(a, "unrelated gardening advice") < 0.3
    assert similarity(a, "zzz") == similarity("zzz", a)


def test_entropy_and_jaccard():
    # entropy is in nats (natural log), matching the model-logit entropy
    assert abs(entropy([0.5, 0.5]) - math.log(2.0)) < 1e-9
    assert abs(entropy([1.0])) < 1e-9
    assert jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert jaccard(["a"], ["b"]) == 0.0


def test_negation_present():
    assert negation_present("This is not supported")
    assert not negation_present("this is supported")


def test_seeded_rng_deterministic():
    a = [seeded_rng(123).random() for _ in range(3)]
    b = [seeded_rng(123).random() for _ in range(3)]
    assert a == b
    assert a != [seeded_rng(124).random() for _ in range(3)]


def test_jsonl_roundtrip(tmp_path):
    rows = [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]
    p = tmp_path / "rows.jsonl"
    write_jsonl(p, rows)
    assert read_jsonl(p) == rows
    q = tmp_path / "obj.json"
    save_json(q, {"k": [1, 2]})
    assert load_json(q) == {"k": [1, 2]}
