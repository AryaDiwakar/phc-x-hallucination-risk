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

## Analysis

Report Krippendorff's alpha or weighted Cohen's kappa, agreement by category,
human-vs-model AUROC/AUPRC, calibration by score band, action disagreement,
false-block rate, and the proportion of cases where evidence retrieval changes

## Required Artifacts

The completed study must add an anonymized CSV/JSONL with annotator IDs
replaced by random codes, the rubric version, exclusion decisions, and a
pre-registered analysis seed. Until that file exists, the paper must describe
human validation as pending rather than claim it was performed.
