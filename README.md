# PHC-X — Provenance-aware Conversational Pre-Generation Hallucination Risk Prediction & Intervention

PHC-X scores the **upcoming** assistant turn *before any token is generated*,
using conversation-level signals (history consistency, provenance of stored
claims, prompt presuppositions, retrieved evidence), calibrates that score on a
held-out split, and maps it to a cost-sensitive intervention:

```
GENERATE → GENERATE_WITH_EVIDENCE → RETRIEVE → CLARIFY → ABSTAIN
```

Everything is **offline-first and deterministic**: hashed n-gram similarity,
rule-based engines, seeded training, no network calls at runtime, no API keys.

Capstone proposal: Arya D (23BDS0023), Jeswanth G (23BDS0212),
N Sai Baba (23BDS0019).

---

## Headline results

**The number we stand behind is the leak-controlled one.** PHC-Conversation
is synthetic, and in the standard benchmark the wording of the question
identifies the risk category, so a lookup table over prompt templates
reaches AUROC 0.808 without reading the conversation. On the leak-controlled
variant — one shared prompt pool, so the question cannot identify the risk
type — the prompt-only ablation collapses to chance while the model holds:

| Metric | PHC-X, leak-controlled (n = 107) | PHC-X, standard (n = 161) | Heuristic (standard) |
|---|---|---|---|
| AUROC | **0.961** | 0.987 [0.972, 0.996] (1000× bootstrap CI) | 0.724 |
| AUPRC | **0.981** | 0.992 | 0.852 |
| F1 @ tuned threshold | **0.928** (P 0.970 / R 0.889) | 0.947 (P 0.942 / R 0.961) | 0.716 |
| ECE (10-bin) | 0.043 | 0.050 (0.092 uncalibrated) | 0.252 |
| Multi-seed AUROC | 0.962 ± 0.001 (seeds 42 / 123 / 2024) | 0.987 ± 0.000 | — |
| *Template lookup (leakage probe)* | *0.635* | *0.808* | *—* |
| *Prompt-only ablation* | *0.434 (chance)* | *0.853* | *—* |
| Action-level gates | **PASS** (coverage 52.3 %, 90.3 % caught, generated-turn rate 0.100 vs base 0.673) | **PASS** (coverage 52.8 %, 97.1 % caught, generated-turn rate 0.021 vs base 0.633) | — |

The prompt-only figure on the standard benchmark is a memorised lookup, not
a measure of what the prompt carries; the leak-controlled column is the
honest one, and it is the one the paper leads with. See
`docs/leakage_control.md`.

Significance vs. heuristic (standard benchmark): McNemar p < 0.001; DeLong
p ≈ 0.45 (AUROC CIs overlap — we report both).

**The leakage probe.** Each category draws its prompts from its own
template pool, so the wording of the question identifies the risk category.
A lookup table that memorizes each template's training base rate — and never
reads the conversation — already reaches **AUROC 0.808**. That is the number
a hallucination detector has to beat, and we report it in every table rather
than leave a reviewer to find it.

**The leak-controlled variant.** Same generator, same families, same label
logic, same family split, but every category draws its prompt from **one
shared pool**, so the prompt cannot identify the risk type (the four
prompt-borne categories — presupposition, referential ambiguity, freshness
cue, benign control — are excluded, because no re-wording moves those into
the conversation):

```bash
phc build-dataset --config configs/dataset_leakcontrol.yaml \
  --out data/phc_conversation_leakcontrol
phc train --config configs/train_leakcontrol.yaml \
  --data data/phc_conversation_leakcontrol --seeds 42,123,2024
phc ablation --config configs/ablation.yaml \
  --data data/phc_conversation_leakcontrol
```

| | PHC-Conversation | leak-controlled |
|---|---:|---:|
| Templates identifying one category | 84.2 % | 0.0 % |
| Template lookup AUROC | 0.808 | 0.635 |
| A: prompt only | 0.853 | **0.434** (chance) |
| E: + history, provenance, evidence | 0.984 | 0.964 |
| F: full | 0.987 | **0.961** |

The prompt-only ablation collapses to chance while the full model barely
moves. The original 0.853 was an artifact; the framework's discrimination
was already coming from the conversation.

**Generalisation stress tests** (`phc stress`):

- *Cross-label source* — training on `label_gold` and scoring against
  `label_verified` gives AUROC **0.779** (against 0.987 for its own source),
  and the reverse gives **0.694** (against 0.989). The model keeps much of
  its discrimination against a label source it has never seen.
- *Leave-one-category-out* — training on ten risk types and testing on the
  eleventh gives mean AUROC **0.804 ± 0.285** over 9 categories, from
  1.000 (`entity_ambiguity`) down to 0.195 (`supported_context`) — the
  framework transfers to risk types it has never seen for most categories,
  and fails on the two whose signal is specific to having seen them.

Ablation AUROC (standard benchmark): `A_prompt_only` 0.853 → `B` 0.853 →
`C` 0.887 → `D` 0.901 → `E_history_provenance_evidence` 0.984 →
`F_full` 0.986. On the leak-controlled variant the same ladder reads
0.434 → 0.789 → 0.793 → 0.804 → 0.964 → 0.961, and the ordering is the one
the method section claims: history and provenance get you to 0.800,
contradiction adds a little, and evidence coverage is the single largest
contributor.

The cost-sensitive threshold search is stable across seeds: multi-seed F1 is
0.943 ± 0.009 and the operating point varies by less than 0.07. The coverage
requirement (≥ 50 % of turns still answered) is imposed on the blocking
cut-point during the validation-only threshold search, with a one-sided 95 %
margin so that it holds on a fresh sample.

Latency: ~21 ms median / 23 ms p95 per analysis steady-state (SHAP warm-up is
done during API start-up; machine-local measurement).

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
phc --help
```

Requires Python ≥ 3.10. Optional: Ollama (`llama3.2:3b`) for local LLM
fallback, Node ≥ 18 for `node --check` syntax gates in tests.

## Quickstart

```bash
# 1. benchmark (already generated in data/phc_conversation/)
phc build-dataset --config configs/dataset.yaml

# 2. train + calibrate + evaluate (writes checkpoints/ and results/)
phc train --config configs/train.yaml --seeds 42,123,2024

# 3. action-level safety gates on the test split
phc benchmark --config configs/train.yaml

# 4. feature-group ablations A–G
phc ablation --config configs/ablation.yaml

# 5. full statistical report + paper figures
phc report                       # reports/stats.json + reports/figures/*.png

# 5b. paper: renders macros/tables from stats.json + compiles (tectonic)
phc paper                        # paper/main.pdf  (no LaTeX engine? --no-compile)

# 6. score one prompt end-to-end
phc risk --prompt "Isn't it true that Paris doubled its population in 2025?"

# 7. service + dashboard
phc serve                        # http://127.0.0.1:8000  (dashboard at /dashboard/)
```

Other subcommands: `evaluate` (re-score a checkpoint), `calibrate` (recalibrate
on val only), `stats` (paired significance between two checkpoints).

## Tests

```bash
python3 -m pytest tests -q        # 140 tests, ~65 s
```

Covers utils, memory, the five consistency engines, retrieval/features, risk
+ policy, dataset generation/leakage, end-to-end pipeline, API endpoints, the
browser extension manifest/sources, the **CLI** (every subcommand, against a
tiny dataset), the **human-validation packet** (blinding + stratification) and
the **annotation analysis** (agreement, calibration, false-block rate), and
one regression test per defect fixed in the correctness sweep.

## External Evaluation

The synthetic benchmark is supplementary, not the primary generalization
claim. Public snapshots of DiaHalu and HaluEval dialogue are evaluated with:

```bash
phc external-eval \
  --diahalu data/raw/diahalu_bench.jsonl \
  --halueval data/raw/halueva_dialogue.json \
  --halluhard data/raw/halluhard_research_questions.jsonl \
  --bootstrap 500
```

Current zero-retraining results are intentionally visible:

| Construction | Rows | PHC-X AUROC | 95% CI |
|---|---:|---:|---:|
| DiaHalu, turn-aligned | 5,923 turns | 0.429 | 0.403–0.456 |
| DiaHalu, stratified by history length | 10 strata | **0.510** | — |
| HaluEval right/hallucinated pair (purity check) | 20,000 | 0.500 | 0.493–0.508 |
| HaluEval counterfactual | 20,000 | **0.705** | 0.699–0.711 |
| HalluHard seeds | 250 | coverage only | N/A |

Three things are now separated cleanly:

- **DiaHalu is turn-aligned.** DiaHalu annotates *which* turn carries the
  error; the previous loader took the last turn, so for 410 of 1,103
  dialogues the label described a different turn than the prompt. Turns after
  an annotated error are excluded because they inherit the false premise.
- **The raw DiaHalu number is confounded** by conversation length, so the
  stratified figure (0.510) is the honest one: the system carries no
  DiaHalu signal.
- **The HaluEval pair is a purity check, not a generalization test.** Its
  AUROC is exactly 0.500, which *demonstrates* that the score cannot be
  using the response — the system is genuinely pre-generation. The
  counterfactual construction uses the same public prompts with only the
  context changed, and reaches 0.705: the context-coverage capability does
  transfer even though hallucination discrimination does not.

See `docs/external_evaluation.md` and `docs/human_validation_protocol.md`.

Optional local pilot baselines use Ollama and are explicitly labeled lexical
approximations, not official SelfCheckGPT or semantic-entropy reproductions.

## Human Validation (pending)

A blinded, stratified annotation packet is prepared, an annotator-facing
instrument exists, and the analysis pipeline has been rehearsed against a
**synthetic** annotation round:

```bash
python3 scripts/prepare_human_validation.py      # blind packet + unblinded key
python3 scripts/simulate_human_annotations.py     # 1,000 synthetic annotations
```

The simulator's annotators model the failure modes of real annotation: a
single latent judgement that drives correlated answers, selective reading of
long conversations, genuine ambiguity, fatigue and order effects, calibration
drift, carryover, and item-independent differences in strictness and caution.
They judge with a deliberately different mechanism from PHC-X, so the
human-vs-model comparison is not circular. Their rationales are written in an
annotator's voice.

**This is a dry run, not a study.** Every artifact is marked `synthetic: true`
and carries a warning; no human labels exist in this repository, and the
paper describes human validation as pending.

### Running the real study

```bash
phc annotate \
  --packet data/annotations/human_validation_sample.jsonl \
  --out data/annotations/annotations.jsonl

phc analyze-annotations \
  --annotations data/annotations/annotations.jsonl \
  --out results/raw/human_validation_report.json
```

`docs/annotator_instructions.md` is the document to hand to an annotator.
The session is resumable, the 40 intra-rater duplicates are not marked, and
annotator codes are random and opaque.

### What the dry run found

- **The pipeline works** end to end, before anyone is paid to annotate.
- **A rubric defect**: a support-based question cannot reach a *reasoning*
  error whose facts are all present, and the DiaHalu items carry no evidence
  at all, so the same question cannot be judged against a document. The
  protocol now also asks whether the reasoning is sound and whether the
  answer addresses the question.
- **A statistic that needed care**: the three yes/uncertain/no questions have
  a "don't know" in the middle, which is not a midpoint between yes and no.
  Krippendorff's ordinal metric assumes it is, and the ordinal and nominal
  figures diverge sharply. The analysis reports both and uses the **nominal**
  metric for those three scales, the ordinal one for the genuinely ordered
  action scale.

---

## Architecture

```
src/phcx/
├── utils/         text similarity, normalization, seeds, IO
├── memory/        ConversationMemory + claim store (append-only)
├── provenance/    claim origin, trust, verification state, contamination
├── presupposition/ entity/ambiguity/contradiction/temporal consistency engines
├── retrieval/     hashed n-gram index over evidence corpus
├── features/      FeatureEngine → 54-dim vector (groups: prompt, history,
│                  provenance, consistency, evidence, model)
├── risk/          RiskModel (RF / LR / GBM / heuristic), calibration, thresholds
├── intervention/  RiskPolicy: risk → action (YAML-configurable, monotone)
├── explain/       factor contributions + SHAP (fast, memoized)
├── verification/  evidence checks, post-hoc label audit
├── pipeline/      PHCPipeline.analyze() — single entry point (~50 ms)
├── dataset/       PHC-Conversation generator, family splits, leakage report
├── experiments/   matrix, metrics, train, ablation, benchmark, report, figures
├── api/           FastAPI service (api.main:app)
└── cli/           `phc` console entry point
```

Serving surfaces:

| Surface | Path | Notes |
|---|---|---|
| CLI | `phc …` | 11 subcommands |
| REST API | `api/main.py`, `phc serve` | `/v1/analyze`, `/v1/batch`, `/v1/policy`, `/v1/metrics`, `/v1/results`, decision log `data/decision_log.jsonl`; env: `PHCX_MODEL`, `PHCX_POLICY`, `PHCX_LOG`, `PHCX_RESULTS` |
| Dashboard | `dashboard/` served at `/dashboard/` | analyzer, policy sliders, live metrics + gates, results explorer |
| Extension | `extension/` (MV3) | intercepts Enter in chat UIs, non-blocking panel, blocking gate for CLARIFY/ABSTAIN; demo at `extension/demo/chat.html` against `phc serve` |

Model checkpoints auto-discover `checkpoints/phcx_train_baseline_seed42/model` (override
with `PHCX_MODEL`); results endpoints read `results/raw/*.json`.

## Data

`data/phc_conversation/` — 1100 prompts, 238 families, splits 768/171/161,
positive rates 0.560 / 0.556 / 0.634, `leakage_report.json: leakage_passed=true`
(no family crosses splits). `label = union(label_gold, label_verified)`;
post-hoc gold-vs-verified agreement ≈ 0.752 is reported, not hidden. The
benchmark contains ten risk categories plus `short_benign` first-turn cases.

## Results layout

```
results/raw/        phcx_lr.json, multiseed_phcx_train_baseline.json,
                    ablation_phcx_train_baseline.json,
                    benchmark_phcx_train_baseline.json, eval_*.json
checkpoints/        phcx_train_baseline_seed{42,123,2024}/model + test_meta.json
reports/            stats.json, figures/fig_{roc,pr,reliability,ablation,actions,risk_hist}.png
paper/              main.tex + generated macros.tex/tables.tex + main.pdf
                    (every number injected from reports/stats.json by `phc paper`)
```

## Design decisions & deviations from the original spec

- Package lives at `src/phcx/` (spec's flat top-level module names were folded
  in), CLI is `phc`, dataset dirs are `data/phc_conversation` + `configs/`.
- Similarity is **hashed n-gram TF cosine** (`phcx.utils.text`), not
  embeddings — deterministic and dependency-free; paraphrase robustness is
  limited by design (documented limitation).
- Rule-based engines instead of spaCy (not installed in this environment).
- `RiskPolicy.from_yaml` reads thresholds top-level but user/latency overrides
  from an `overrides:` mapping.
- Benchmarks gate on *action-level* safety (positive rate among non-blocking
  actions < base rate), not only AUROC.

## Limitations

Synthetic benchmark with a scripted assistant; white-box "model" features are
zeros outside serving (ablation G ≈ F offline); lexical rather than embedding
similarity. See `docs/novelty_matrix.md` § Limitations for the list stated in
the paper.

## References & novelty

- `docs/literature_review.md` — every citation individually verified against
  arXiv/ACL Anthology (2026-10-06); unverifiable claims are flagged.
- `docs/novelty_matrix.md` — capability-by-capability comparison.
- `docs/product_plugin.md` — plugin product path and acceptance criteria.
- Confirmed gap: no existing work predicts pre-generation risk from
  conversation-level signals (HalluGuard, ICLR 2026 lists it as future work).

## License

See `LICENSE`.
