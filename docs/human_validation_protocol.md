# Human Validation Protocol

This protocol is ready for an independent annotation round. No human labels
are fabricated in the repository; `results/raw/external_eval.json` contains
only machine-computed results.

## Sampling

Sample 240 items, stratified equally across:

- PHC-Conversation risk categories
- DiaHalu domains
- PHC-X action: GENERATE, RETRIEVE, and CLARIFY/ABSTAIN
- Low, moderate, high, and critical score bands

Include 40 duplicate items for intra-rater consistency and blind the annotator

### Implementation

`scripts/prepare_human_validation.py` builds the packet:

```bash
python3 scripts/prepare_human_validation.py \
  --out data/annotations/human_validation_sample.jsonl \
  --key-out data/annotations/human_validation_key.jsonl \
  --n 240 --duplicates 40 --seed 42
```

Every candidate is scored with the trained checkpoint, so the action and the
score band are the model's own; items are then drawn round-robin over the
(category or domain) × action × band cells, which keeps every populated
stratum represented — a uniform shuffle would leave whole action/band
combinations empty whenever they are rare in the pool.

Two files are written:

- `human_validation_sample.jsonl` — the **blind** packet. It carries the
  prompt, history, evidence, proposed answer, the rubric and empty
  `annotator_label` / `rationale` fields. It contains no label, no score, no
  action and no stratum.
- `human_validation_key.jsonl` — the **unblinded key**: model score, level,
  action, band, stratum and gold label per `annotation_id`, plus the rubric
  version and the pre-registered analysis seed. This file is for the
  analysis step only and must never be given to the annotator.

The 40 duplicates are byte-identical copies of selected items under a new
`annotation_id`, with no marker that they are repeats, so that intra-rater
consistency can be measured.

## Annotation Questions

Each annotator receives the prompt, conversation history, evidence passages,

1. Is the answer factually supported by the supplied context? (`yes`, `no`,
   `uncertain`)
2. Does it contradict the conversation history? (`yes`, `no`, `uncertain`)
3. Does it contain an unsupported presupposition or entity/time drift?
   (`yes`, `no`, `uncertain`)
4. Would you allow generation without retrieval or clarification? (`allow`,
   `retrieve`, `clarify`, `abstain`)
5. Confidence (`1`–`5`) and a short rationale.

### Recommended addition (from the dry run, see below)

6. Is the answer's *reasoning* sound, and does it actually answer the
   question that was asked? (`yes`, `no`, `uncertain`)

Question 6 is needed because question 1 alone cannot reach a reasoning error:
when the facts are all present but the reasoning is wrong, "yes, the answer
is supported" is the *correct* answer to question 1 and still disagrees with
the gold label. Without question 6 the consensus label and the gold label
disagree by construction on those items.

## Analysis

Report Krippendorff's alpha or weighted Cohen's kappa, agreement by category,
human-vs-model AUROC/AUPRC, calibration by score band, action disagreement,
false-block rate, and the proportion of cases where evidence retrieval changes

The implementation is in `src/phcx/experiments/human_validation.py`:

- `simulate_annotations(...)` — the synthetic crowd (see below)
- `analyze_annotations(annotations, key, ...)` — the protocol's analysis

## Dry Run: a Synthetic Annotation Round

The analysis pipeline is exercised before any real annotator exists, using a
simulator whose "annotators" judge the visible content with **surface lexical
checks and per-annotator noise** — deliberately a different mechanism from
PHC-X, so that the human-vs-model comparison is not circular:

```bash
python3 scripts/simulate_human_annotations.py \
  --packet data/annotations/human_validation_sample.jsonl \
  --key data/annotations/human_validation_key.jsonl \
  --out results/raw/human_validation_simulated.jsonl \
  --report results/raw/human_validation_simulated_report.json \
  --annotators 100 --n-annotations 1000 --seed 42
```

Each simulated annotator has a stable personality: strictness about support,
caution about allowing generation, a slip rate, a confidence bias, and a
diligence level. Every annotation is marked `synthetic: true` and every file
carries a warning.

**This is not human validation and must never be reported as such.** The
paper continues to describe human validation as pending. What the dry run
gives is:

1. **A working analysis pipeline.** Inter-rater agreement
   (Krippendorff's alpha, Fleiss' kappa), human-vs-model AUROC/AUPRC,
   calibration by score band, action disagreement and the false-block rate
   are all computed end to end before anyone is paid to annotate.
2. **A noise floor.** The simulated crowd is deliberately noisy, so the
   agreement figures it produces are a *lower bound* on what a real study
   should expect. If real annotators agree less than the simulation does,
   the rubric is the problem.
3. **A protocol defect.** The reachability diagnostic splits the items by
   the DiaHalu error type and shows that a support-based rubric recovers
   fact-and-context errors at ~0.63 accuracy but reasoning errors at ~0.32 —
   and the gap is *constructive*, not annotator noise. This is what
   motivated question 6 above.

## Required Artifacts

The completed study must add an anonymized CSV/JSONL with annotator IDs
replaced by random codes, the rubric version, exclusion decisions, and a
pre-registered analysis seed. Until that file exists, the paper must describe
human validation as pending rather than claim it was performed.
