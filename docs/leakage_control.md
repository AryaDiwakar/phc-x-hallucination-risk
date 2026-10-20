# Leakage Control

PHC-Conversation is synthetic, so the obvious reviewer attack is that the
benchmark rewards recognising *our own generator* rather than hallucination
risk. This document records the artifact we found, the variant that removes
it, and what the remaining numbers mean.

## The artifact

Each risk category draws its prompts from its own template pool. That means
the wording of the question identifies the category, and the categories have
very different base rates (0.00 for `short_benign`, 1.00 for several
templates). A lookup table that memorizes each template's training base rate
therefore predicts the label without reading anything else:

| Baseline | AUROC |
|---|---:|
| Category lookup table (train base rate per category) | 0.723 |
| **Template lookup table (train base rate per template)** | **0.808** |
| Prompt-only ablation (a learned model on prompt features only) | 0.853 |
| Full PHC-X | 0.986 |

So the prompt-only ablation of 0.853 was almost entirely a memorised lookup,
and the paper's phrase "conversation context—not the prompt alone—carries
most of the signal" understated how much of the headline was artifact.

The probe is now computed by `phcx.experiments.metrics.template_lookup_scores`
and reported in `stats.json` (`leakage_probe`, `template_lookup_baseline`),
in the main results table, and in the README.

## The leak-controlled variant

Same generator, same families, same label logic, same family split, same
model, same hyper-parameters, same protocol. Two changes only:

1. **Every category draws its prompt from one shared pool**
   (`NEUTRAL_PROMPTS`), so the prompt carries no category information.
2. **Only the context-borne categories are included.** A presupposition, a
   referential ambiguity and a freshness cue *are* properties of the
   question — no re-wording moves them into the conversation — so
   `temporal_inconsistency`, `unsupported_presupposition`,
   `referential_ambiguity` and `short_benign` are excluded.

```bash
phc build-dataset --config configs/dataset_leakcontrol.yaml \
  --out data/phc_conversation_leakcontrol
phc train --config configs/train_leakcontrol.yaml \
  --data data/phc_conversation_leakcontrol --seeds 42,123,2024
phc benchmark --config configs/train_leakcontrol.yaml \
  --data data/phc_conversation_leakcontrol
phc ablation --config configs/ablation.yaml \
  --data data/phc_conversation_leakcontrol
```

## The result

| | PHC-Conversation | leak-controlled |
|---|---:|---:|
| rows | 1,100 | 700 |
| categories | 11 | 7 |
| templates identifying one category | 84.2 % | 0.0 % |
| template lookup AUROC | 0.808 | 0.635 |
| A: prompt only | 0.853 | **0.485** |
| C: + history, provenance | 0.887 | 0.800 |
| D: + contradiction | 0.901 | 0.808 |
| E: + evidence | 0.984 | 0.965 |
| F: full | 0.986 | **0.963** |
| action-level gates | PASS | PASS |

Three things follow.

**The prompt-only number was an artifact.** It collapses to chance
(0.485) once the prompt cannot identify the risk type.

**The framework's discrimination was never resting on it.** The full model
moves only from 0.986 to 0.963, so the conversation state and the evidence
were carrying the signal all along — the artifact was available to
*baselines*, not to the model.

**The ablation ladder becomes interpretable.** With a clean prompt, the
order is the one the method section claims: history and provenance reach
0.800, contradiction adds 0.008, and evidence coverage is the single
largest contributor at 0.965.

## Generalisation stress tests

`phc stress` runs two further probes.

**Cross-label source.** PHC-Conversation carries two independent labels:
`label_gold` (by construction) and `label_verified` (from the post-hoc
verifier). They agree on only ~0.69 of the rows, so they behave like two
annotators.

| Trained on | Scored against | AUROC |
|---|---|---:|
| `label_gold` | `label_gold` (its own source) | 0.983 |
| `label_gold` | `label_verified` (unseen source) | **0.778** |
| `label_verified` | `label_verified` (its own source) | 0.989 |
| `label_verified` | `label_gold` (unseen source) | **0.694** |

The model keeps a substantial part of its discrimination against a label
source it has never seen, which is evidence that it learned something about
the conversation rather than about one particular labeller — and it
quantifies how much of the headline was label-source-specific.

**Leave-one-category-out.** Each risk type is held out in turn, so the
model has to detect a risk type it has never been trained on.

| Held-out category | n | AUROC |
|---|---:|---:|
| `entity_ambiguity` | 20 | 1.000 |
| `model_memory_contamination` | 12 | 1.000 |
| `unsupported_presupposition` | 11 | 1.000 |
| `contradictory_context` | 12 | 0.975 |
| `long_context_degradation` | 17 | 0.955 |
| `temporal_inconsistency` | 10 | 0.833 |
| `supported_context` | 18 | 0.805 |
| `referential_ambiguity` | 20 | 0.736 |
| `unsupported_context` | 18 | 0.377 |

Mean **0.853 ± 0.192** over 9 categories (two further categories have a
single-class held-out split, so no AUROC is defined). The framework transfers
to unseen risk types for most categories and fails on `unsupported_context`,
whose signal — the question's topic does not match the evidence — is
specific to having seen that category.

## What this does not fix

The leak-controlled variant is still synthetic, and it is still our own
generator. It removes one specific artifact and makes the context-based
claim measurable; it does not substitute for a benchmark with independent
labels. That is what the human-validation study is for, and it is still
pending.

## How the paper uses this

The paper and the README **lead with the leak-controlled number** (AUROC
0.963, n = 107) and report the standard benchmark (0.986, n = 161) beside
it, with the template-lookup probe in both columns. The reasoning: the
standard benchmark's 0.986 is a real measurement, but a reader cannot tell
from it how much of it came from the conversation. The leak-controlled
number answers that question directly, so it is the one the headline should
carry — and quoting 0.986 alone would invite exactly the objection this
document exists to pre-empt.
