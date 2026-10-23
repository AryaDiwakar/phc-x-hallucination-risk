"""Tests for the benchmark-leakage probe and the generalisation stress tests.

The probe exists because the original benchmark let the *prompt* identify
the risk category, so a lookup table over prompt templates could reach a
high AUROC without reading the conversation.  These tests pin the probe, the
leak-controlled variant and the two stress tests.
"""
import numpy as np
import pytest
import yaml

from phcx.dataset.generator import (
    CATEGORIES,
    CATEGORIES_CONTEXT_BORNE,
    NEUTRAL_PROMPTS,
    PHCConversationGenerator,
)
from phcx.experiments.metrics import (
    auroc,
    prompt_template,
    template_lookup_scores,
)
from phcx.experiments.stress import (
    run_cross_label_generalization,
    run_leave_one_category_out,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


# ------------------------------------------------------------- the probe
def test_prompt_template_masks_the_entity_and_number_slots():
    a = prompt_template("What is the GPU-hour cost of the Aurora-12A project?")
    b = prompt_template("What is the GPU-hour cost of the Granite-3Q project?")
    c = prompt_template("What is the GPU-hour cost of the Aurora-12A project?")
    assert a == b, "different entities must map to the same template"
    assert a == c
    assert "<e>" in a, "the entity slot must be masked"
    # numbers are masked too, so two different numeric prompts are one template
    assert prompt_template("BLEU improved from 68.7% to 71.5%") == \
        prompt_template("BLEU improved from 60.0% to 61.0%")


def test_template_lookup_scores_uses_the_training_rate_only():
    train = [
        {"prompt": "Ask about Aurora-12A please", "label": 1},
        {"prompt": "Ask about Granite-3Q please", "label": 0},
    ]
    test = [
        {"prompt": "Ask about Vega-1Q please", "label": 1},
        {"prompt": "Ask about Basalt-2P please", "label": 0},
    ]
    scores = template_lookup_scores(train, test)
    # both test prompts are unseen, so they fall back to the global rate
    assert scores.shape == (2,)
    assert abs(float(scores[0]) - 0.5) < 1e-9
    assert abs(float(scores[0]) - float(scores[1])) < 1e-9


# ------------------------------------------------- the leak-controlled variant
def test_leak_controlled_generator_does_not_leak_the_category():
    gen = PHCConversationGenerator(samples_per_category=3, seed=5, n_families=20,
                                   long_context_extra_turns=(4, 6),
                                   prompt_mode="neutral")
    rows = gen.generate()
    assert rows
    # only the context-borne categories are present
    assert {r["category"] for r in rows} <= set(CATEGORIES_CONTEXT_BORNE)
    # the prompts come from the shared pool, so there are at most three
    # distinct templates whatever the category
    templates = {prompt_template(r["prompt"]) for r in rows}
    assert len(templates) <= len(NEUTRAL_PROMPTS)


def test_leak_controlled_prompt_cannot_identify_the_category():
    """Every prompt template must appear in more than one category."""
    gen = PHCConversationGenerator(samples_per_category=6, seed=5, n_families=25,
                                   long_context_extra_turns=(4, 6),
                                   prompt_mode="neutral")
    rows = gen.generate()
    cats_per_template = {}
    for r in rows:
        cats_per_template.setdefault(prompt_template(r["prompt"]), set()).add(r["category"])
    for template, cats in cats_per_template.items():
        assert len(cats) > 1, (
            f"template {template!r} maps to a single category {cats}; the "
            "leak-controlled variant must not identify the risk type"
        )


def test_standard_generator_does_leak_the_category():
    """The control: the standard generator *does* identify the category.

    This is the defect the leak-controlled variant exists to remove, so the
    test asserts it is still present in the standard generator.
    """
    gen = PHCConversationGenerator(samples_per_category=8, seed=5, n_families=30,
                                   long_context_extra_turns=(4, 6))
    rows = gen.generate()
    assert {r["category"] for r in rows} == set(CATEGORIES)
    cats_per_template = {}
    for r in rows:
        cats_per_template.setdefault(prompt_template(r["prompt"]), set()).add(r["category"])
    single = sum(1 for c in cats_per_template.values() if len(c) == 1)
    assert single > 0.5 * len(cats_per_template), (
        "the standard generator is expected to leak the category through the "
        "prompt template"
    )


# --------------------------------------------------------------- stress tests
@pytest.fixture(scope="module")
def cfg():
    with open(ROOT / "configs" / "train.yaml") as fh:
        return yaml.safe_load(fh)


def test_cross_label_generalization_runs_and_reports_both_directions(cfg, tmp_path):
    out = tmp_path / "cross.json"
    result = run_cross_label_generalization(cfg, out_path=out, show_progress=False)
    assert set(result["results"]) == {
        "train_label_gold__test_label_verified",
        "train_label_verified__test_label_gold",
    }
    for entry in result["results"].values():
        # it must beat chance against a label source it never saw, and lose
        # AUROC against its own source
        assert entry["auroc_vs_other_source"] > 0.5
        assert entry["auroc_vs_other_source"] < entry["auroc_vs_own_source"]
    assert out.exists()


def test_leave_one_category_out_runs_and_reports_per_category(cfg, tmp_path):
    out = tmp_path / "loco.json"
    result = run_leave_one_category_out(cfg, out_path=out, show_progress=False)
    per = result["per_category"]
    assert len(per) >= 5
    assert result["mean_auroc"] is not None
    for entry in per:
        assert "category" in entry and "n" in entry
    assert out.exists()


def test_leakcontrol_model_selection_is_validation_only():
    """The leak-control model was selected without looking at its test set."""
    report_path = ROOT / "reports" / "model_selection_leakcontrol.json"
    if not report_path.exists():
        pytest.skip("run scripts/sweep_leakcontrol_models.py first")
    report = __import__("json").loads(report_path.read_text())
    assert report["validation_only"] is True
    assert report["test_split_used_for_selection"] is False
    assert report["selected"]["name"] == "xgb_d3"
    # The selected model has to beat the prior RF on validation, across seeds.
    candidates = {c["name"]: c for c in report["candidates"]}
    assert (candidates["xgb_d3"]["validation"]["auroc"]["mean"] >
            candidates["rf_current"]["validation"]["auroc"]["mean"])
