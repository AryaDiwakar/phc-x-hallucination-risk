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

| Construction | Rows | Status | PHC-X AUROC | 95% CI |
|---|---:|---|---:|---:|
| DiaHalu, turn-aligned | 5,923 turns | scored | 0.429 | 0.403–0.456 |
| DiaHalu, stratified by history length | 10 strata | scored | **0.507** | — |
| HaluEval right/hallucinated pair | 20,000 paired responses | scored | 0.500 | 0.493–0.508 |
| HaluEval counterfactual | 20,000 | scored | **0.709** | 0.702–0.716 |
| HalluHard seeds | 250 | coverage only | N/A | N/A |

### DiaHalu, turn-aligned

DiaHalu annotates *which* turn of a dialogue carries the error: `index` is
`[[turn, error_type], ...]` (e.g. `[[7, "Incoherence"]]`) and the explanation
names it (`"B7 contradicts B2"`). The previous loader took the **last** turn
of the dialogue, so for the 410 of 1,103 dialogues whose annotated turn is not
the last one the label described a different turn than the prompt. The loader
now evaluates the annotated turn: the prompt is the user turn that elicited
it, the history is everything before it, and the label describes that turn.

Turns *after* an annotated error are excluded. They inherit the false premise
the assistant just established, so they are genuinely risky but carry no
annotation; labelling them 0 injects noise that inverts the signal.

The raw AUROC of 0.429 is still confounded, because the annotated turn is
systematically earlier in the dialogue than the negatives (mean history 4.5
turns against 5.6), and conversation length is itself a risk signal. The
`stratified_by_history_length` field in the output reports the AUROC within
each history-length stratum and a positives-weighted average of them:
**0.507**. Once the confound is removed the system carries no DiaHalu signal.
We report the stratified number, not the raw one.

### HaluEval, as a purity check

The right/hallucinated pair shares its prompt, history and evidence exactly,
so no pre-generation system can separate it. The AUROC is 0.500 with a tight
confidence interval on 20,000 rows, which is the point: it demonstrates that
the score cannot be using the response, i.e. that the system is genuinely
pre-generation. It is a purity check, not a generalization test.

### HaluEval, counterfactual

To obtain a task that *is* decidable pre-generation, the same public prompts
are used and only the context changes:

- **supported** (label 0) — the record's own `knowledge` document
- **unsupported** (label 1) — another record's `knowledge` document

Evidence is present in both cases, so the task cannot be solved by noticing
that evidence is missing; it has to be solved by measuring how much of the
question the evidence actually covers. The system reaches **AUROC 0.709**
[0.702, 0.716]; the transparent heuristic baseline is slightly better (0.734).
This is a genuine external positive result, on external prompts, for the
capability the framework is built around.

An adversarial variant is available: `distractor="nearest"` substitutes the
*most similar* other knowledge document. That scores **below chance**
(0.281), because separating it from the true document requires verifying the
fact itself, which PHC-X does not claim to do. It is a useful boundary
result, not a fair test of the coverage machinery.

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

PHC-Conversation is a controlled supplementary benchmark. The primary claim is
the framework and its deployment path, not the synthetic AUROC.

The external results now say something more precise than "generalization
failed":

- **Discrimination of hallucinated *responses* does not transfer.** The
  synthetic AUROC of 0.986 is a property of the PHC-Conversation generator,
  and the stratified DiaHalu result (0.507) shows the framework does not
  detect externally-produced hallucinations.
- **The context-coverage capability does transfer.** On the same external
  prompts, the system reaches 0.709 AUROC at telling whether the supplied
  context actually covers the question.

What remains open is a benchmark whose labels represent the probability of
the *next* turn hallucinating, without labels attached only after a
particular answer was generated. The human-validation packet described in
`docs/human_validation_protocol.md` is the first step towards one.
