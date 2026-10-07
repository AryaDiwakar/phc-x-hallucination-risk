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
