# External Evaluation

## Reproduction

The repository includes downloaded public-source snapshots under
`data/raw/`:

```bash
phc external-eval \
  --diahalu data/raw/diahalu_bench.jsonl \
  --halueval data/raw/halueva_dialogue.json \
  --halluhard data/raw/halluhard_research_questions.jsonl \
  --bootstrap 500
```

This writes `results/raw/external_eval.json`. The checkpoint is trained only
on PHC-Conversation; no external labels are used for fitting or calibration.

## Current Results

| Dataset | Rows | Status | PHC-X AUROC | 95% CI |
|---|---:|---|---:|---:|
| DiaHalu | 1,103 | scored | 0.491 | 0.482–0.499 |
| HaluEval dialogue | 20,000 paired responses | scored | 0.500 | 0.494–0.507 |
| HalluHard seeds | 250 | coverage only | N/A | N/A |

These are intentionally reported as negative/generalization results. DiaHalu
and HaluEval labels describe a response that already exists, while PHC-X sees
only the prompt, history, and evidence before generation. In HaluEval, the
correct and hallucinated responses share those exact pre-generation inputs, so
response-level discrimination is impossible by construction. This exposes a
label-alignment limitation rather than proving that PHC-X is a good
response-level detector.

## Baselines

The external harness includes transparent prompt-length and evidence-absence
baselines, plus the PHC-X heuristic. The optional local pilot is:

```bash
python3 - <<'PY'
from phcx.experiments.external import load_diahalu
from phcx.experiments.llm_baselines import evaluate_local_baselines

rows = load_diahalu("data/raw/diahalu_bench.jsonl", limit=20)
evaluate_local_baselines(
    rows,
    model_names=("llama3.2:3b", "qwen2.5:3b"),
    sample_count=2,
    limit=20,
    out_path="results/raw/local_llm_baselines.json",
)
PY
```

The pilot uses lexical approximations, not official SelfCheckGPT or semantic
entropy implementations. CoVe and RARR are post-generation methods and are
not apples-to-apples pre-generation classifiers; the paper records them as
non-comparable rather than manufacturing a score.

## Interpretation

PHC-Conversation is now a controlled supplementary benchmark. The primary
claim is the framework and its deployment path, not the synthetic AUROC.
External generalization remains an open result requiring a benchmark whose
labels represent the probability of the *next* turn hallucinating, rather
than labels attached only after a particular answer was generated.
