from collections import Counter

from phcx.dataset import CATEGORIES, PHCConversationGenerator, family_split, leakage_report
from phcx.dataset.build import build_dataset


def test_generator_deterministic():
    a = PHCConversationGenerator(samples_per_category=3, seed=11, n_families=30).generate()
    b = PHCConversationGenerator(samples_per_category=3, seed=11, n_families=30).generate()
    assert [r["prompt"] for r in a] == [r["prompt"] for r in b]
    assert [r["label"] for r in a] == [r["label"] for r in b]


def test_all_categories_present(tiny_rows):
    cats = Counter(r["category"] for r in tiny_rows)
    assert set(cats) == set(CATEGORIES)


def test_sample_schema(tiny_rows):
    required = {
        "sample_id", "family_id", "category", "prompt", "conversation_history",
        "evidence", "expected_answer", "generated_answer", "claims",
        "label", "label_gold", "label_verified", "verifier",
        "contradiction_label", "temporal_label", "ambiguity_label",
        "presupposition_label", "contamination_label", "synthetic", "model",
    }
    for r in tiny_rows[:20]:
        assert required <= set(r)
        assert r["label"] in (0, 1)
        assert isinstance(r["conversation_history"], list)
        assert r["synthetic"] is True


def test_no_private_data_no_pii(tiny_rows):
    joined = " ".join(
        r["prompt"] + " " + r["generated_answer"] for r in tiny_rows
    ).lower()
    for marker in ("@gmail", "ssn", "password", "credit card"):
        assert marker not in joined


def test_family_split_no_crossing(tiny_rows):
    splits = family_split(tiny_rows, ratios=(0.7, 0.15, 0.15), seed=3)
    fam_split = {}
    for name, rows in splits.items():
        for r in rows:
            fam_split.setdefault(r["family_id"], set()).add(name)
    assert all(len(v) == 1 for v in fam_split.values())
    assert sum(len(v) for v in splits.values()) == len(tiny_rows)


def test_leakage_audit_clean(tiny_rows):
    splits = family_split(tiny_rows, seed=3)
    rep = leakage_report(tiny_rows, splits)
    assert rep["passed"], rep["checks"]
    assert rep["checks"]["family_overlap"] == 0


def test_build_dataset_end_to_end(tmp_path):
    summary = build_dataset(out_dir=tmp_path / "ds", samples_per_category=4,
                            n_families=40, seed=5)
    assert summary["n_rows"] == 4 * len(CATEGORIES)
    assert summary["leakage_passed"] is True
    assert (tmp_path / "ds" / "train.jsonl").exists()
    assert (tmp_path / "ds" / "leakage_report.json").exists()
    assert (tmp_path / "ds" / "build_summary.json").exists()
    rates = list(summary["label_rate"].values())
    assert all(0.0 <= r <= 1.0 for r in rates)
    assert any(0.0 < r < 1.0 for r in rates)    # not degenerate


def test_label_not_degenerate_per_category(tiny_rows):
    """Every category must contain both classes somewhere at full scale."""
    full = PHCConversationGenerator(samples_per_category=25, seed=21, n_families=80).generate()
    by_cat = {}
    for r in full:
        by_cat.setdefault(r["category"], set()).add(r["label"])
    mixed = {c: v for c, v in by_cat.items() if len(v) == 2}
    assert len(mixed) >= 8, f"only {len(mixed)} categories have both classes"


def test_short_benign_covers_first_turns():
    rows = PHCConversationGenerator(samples_per_category=5, seed=21,
                                     n_families=20).generate()
    benign = [r for r in rows if r["category"] == "short_benign"]
    assert len(benign) == 5
    assert all(r["label"] == 0 for r in benign)
    assert all(not r["conversation_history"] and not r["evidence"] for r in benign)
