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

## Headline results (test split, n = 161; run `phc report` to regenerate)

| Metric | PHC-X (RF) | Heuristic baseline |
|---|---|---|
| AUROC | **0.984** [0.966, 0.996] (1000× bootstrap CI) | 0.717 |
| AUPRC | **0.995** | 0.852 |
| F1 @ tuned threshold | **0.961** (P 0.961 / R 0.961) | 0.698 |
| ECE (10-bin) | **0.026** | 0.280 |
| Multi-seed AUROC | 0.980 ± 0.003 (seeds 42 / 123 / 2024) | — |

Ablation AUROC: `A_prompt_only` 0.846 → `B` 0.846 → `C` 0.889 → `D` 0.895 →
`E_history_provenance_evidence` 0.978 → `F_full` 0.984.
Benchmark gates **PASS**: coverage 50.9 %, positives caught 96.1 %,
safety margin 0.613 (blocking = CLARIFY/CLARIFY_OR_ABSTAIN/ABSTAIN).
Significance vs. heuristic: McNemar p < 0.001; DeLong p ≈ 0.46 (AUROC CIs
overlap — we report both).

Latency: ~22 ms median / 36 ms p95 per analysis steady-state (SHAP warm-up is
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
python3 -m pytest tests -q        # 71 tests, ~25 s
```

Covers utils, memory, the five consistency engines, retrieval/features, risk
+ policy, dataset generation/leakage, end-to-end pipeline, API endpoints, and
the browser extension manifest/sources.

## External Validation

The synthetic benchmark is supplementary, not the primary generalization
claim. Public snapshots of DiaHalu and HaluEval dialogue are evaluated with:

```bash
phc external-eval \
  --diahalu data/raw/diahalu_bench.jsonl \
  --halueval data/raw/halueva_dialogue.json \
  --halluhard data/raw/halluhard_research_questions.jsonl \
  --bootstrap 500
```

Current zero-retraining results are intentionally visible: DiaHalu AUROC
0.491 (1,103 rows), HaluEval paired-response AUROC 0.500 (20,000 rows), and
HalluHard coverage-only (250 unlabeled prompts). The HaluEval pairing shares
the same pre-generation inputs for correct and hallucinated responses, so it
cannot be treated as a valid next-turn propensity benchmark. See
`docs/external_evaluation.md` and `docs/human_validation_protocol.md`.

Optional local pilot baselines use Ollama and are explicitly labeled lexical
approximations, not official SelfCheckGPT or semantic-entropy reproductions.

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
