# PHC-X generated live-context stress run

Command: `python3 scripts/evaluate_live_context_10k.py`
Seed: `20261024` · Model: `rf:phcx_train_baseline_seed42`
Cases: 10,000 (2,000 per scenario) · Mean / p95 analysis latency: 15.7 / 26.3 ms

These are **template-generated, human-style examples** with labels assigned by
how each case was constructed. They are not human-authored, human-labeled, or
evidence of real-world hallucination prevention. The JSONL cases and full
per-case report are `data/stress/live_context_10k.jsonl` and
`reports/live_context_10k.json`.

| Scenario | Expected class | Non-blocking / intervention result |
|---|---|---:|
| Supported numeric follow-up | Safe | 100% non-blocking; 0% hard-blocked |
| Context-setting message | Safe | 100% generated; 0% hard-blocked |
| Missing fact | Risky | 100% got an intervention; 7.4% got `GENERATE_WITH_EVIDENCE` rather than retrieval/clarification |
| Conflicting numbers | Risky | 100% got an intervention; 49.7% asked/blocked, remainder retrieved |
| Ambiguous reference | Risky | 100% got an intervention; 75.3% asked/blocked, remainder retrieved |

Across the constructed 4,000 safe cases, the hard-block rate was 0%. Across
6,000 constructed risky cases, 100% received an intervention (including
`GENERATE_WITH_EVIDENCE`); 55.4% were hard-blocking actions. A hard block is
not always the best outcome: retrieval/evidence recommendations are also
interventions, and these generated labels cannot determine which response is
appropriate for real users.

## Important model result

The **raw trained classifier scored AUROC 0.5013** against these construction
labels—approximately chance. The effective pipeline's AUROC was 1.0, but that
number is dominated by the explicit source/context and ambiguity safeguards
that match these deliberately constructed scenarios. It is not evidence that
the random forest itself is strong, and it must not be represented as an
independent or human-validated benchmark result.

On the same generated cases, the transparent heuristic scored raw AUROC
0.6930 and ran at 0.9 ms mean latency. PHC-X therefore defaults the API to
the heuristic; the random-forest checkpoint is opt-in via
`PHCX_MODEL=auto phc serve`. This comparison is still generator-specific and
does not establish that the heuristic generalizes to real conversations.

A fixed 85/15 heuristic/RF blend scored **0.7432 base-pipeline AUROC** in the
same generated run (five-fold score-weight checks selected weights around
0.15–0.20; no model weights were trained). The source-aware safety layer's
constructed-label AUROC remains 1.0, for the same non-independent reasons
above. The blend is available experimentally with `PHCX_MODEL=ensemble phc
serve`; it is not the default because no human-labeled validation supports
that choice.

## No-retraining local LLM check

The already-installed Ollama models were also evaluated zero-shot (no
fine-tuning): Qwen 2.5 3B on 500 stratified examples and Llama 3.2 3B on 250.
Against the same construction labels, binary accuracy was 0.462 and 0.432;
risky-case recall was 0.103 and 0.053, respectively, at roughly 0.7–0.8 s per
case. These small-model results are not suitable as a drop-in risk gate. Some
labels may be overly strict—for example, a clearly marked correction can make
the latest number answerable—so this is diagnostic evidence, not a fair
human-quality ranking of those LLMs. Reproduction scripts and raw rows are
`scripts/evaluate_local_judge.py`, `reports/local_judge_qwen_500.json`, and
`reports/local_judge_llama_250.json`.

This run is useful as a regression test for the specific false positive that
motivated it (a supported count question being blocked) and for these five
scenario families. It does not validate paraphrases outside the generator,
multimodal history, independently sourced factual correctness, or deployment
behavior on a representative ChatGPT population. The next meaningful model
evaluation needs a consented, human-labeled sample of real conversations with
family/conversation-level held-out splits.
