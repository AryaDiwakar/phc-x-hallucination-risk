# Showing PHC-X Works

A defence-ready plan: what to run, what to say, what the numbers are, and
what to concede. Every number below is from `reports/stats.json` and can be
regenerated with `phc report`.

---

## Phase 0 — Pre-flight (do this the day before)

### 0.1 Settle the branch

All the work is on `fix/correctness-sweep`; `master` is still the original
commit. Pick one:

```bash
git checkout master
git merge --no-ff fix/correctness-sweep     # keeps the history readable
git push origin master
```

or keep the branch and say so. **Do not leave `master` holding the broken
version** — everything below assumes the fixed code.

### 0.2 Verify from scratch

```bash
python3 -m pytest tests -q          # expect: 148 passed
```

```bash
phc train --config configs/train.yaml --seeds 42,123,2024
phc benchmark --config configs/train.yaml
phc report --config configs/train.yaml
phc paper
```

Everything is seeded, so the numbers must come out identical. If they don't,
something is stale.

### 0.3 Check the artifacts are current

```bash
python3 - <<'PY'
import os, glob
src = max(os.path.getmtime(f) for f in
           glob.glob('src/**/*.py', recursive=True) + glob.glob('api/*.py'))
for a in ('reports/stats.json', 'paper/main.pdf',
          'results/raw/benchmark_phcx_train_baseline.json'):
    print(f"  {a:52} {'FRESH' if os.path.getmtime(a) >= src else 'STALE'}")
PY
```

### 0.4 Practise the demo on the machine you'll present on

Notably: **Chrome 137+ ignores `--load-extension`**. Load the extension from
`chrome://extensions` → Developer mode → *Load unpacked* → the `extension/`
folder. Check it works *before* the day.

---

## Phase 1 — The three-minute demo

```bash
./scripts/demo.sh
```

That starts the service, checks the checkpoint loaded, serves the sandbox
page and prints the URLs. Then:

1. Open `http://127.0.0.1:8000/dashboard/` — the risk-operations view.
2. Open `http://127.0.0.1:8022/chat.html` in another tab, with the
   extension loaded.
3. Type **"How many samples does Atlas A contain?"** and press **Enter**.

### What you should see

The guard panel appears on the composer:

```
PHC-X   0.99 CRITICAL   CLARIFY_OR_ABSTAIN
critical risk: ask clarification or abstain
  ev_retrieval_similarity   +0.095
  ev_coverage               +0.041
  ev_supporting             +0.039
evidence coverage: 2%
                    [ send anyway ]  [ edit prompt ]
```

**And the message is not sent.** The conversation log still shows only the
original four turns.

### Why this demo is the right one

The sandbox conversation contains an assistant turn that asserted
*"Atlas A contains 50,000 samples"* with **no source**. The prompt asks
exactly about that number. So the system is not judging the question — it is
judging that the answer would rest on something the model asserted and never
verified.

Then click **"send anyway"** — the override works. And say the quiet part:
*the guard never stops the user; it makes them decide with the risk in
front of them.*

### The contrast, in one command

If you have a terminal, this is the sharpest single piece of evidence:

```bash
phc risk --prompt "How many samples does Atlas A contain?" --history /tmp/unsourced.json
phc risk --prompt "How many samples does Atlas A contain?" --history /tmp/sourced.json
```

where `unsourced.json` has the assistant turn plain, and `sourced.json` has
`"source":"doc:spec","verified":true` added.

**0.89 → 0.14.** One turn's provenance, half the risk score.

---

## Phase 2 — The evidence tour (one command per claim)

| Your claim | Run this | The number |
|---|---|---|
| It works | `phc risk --prompt "..."` | risk 0.89 → RETRIEVE, in **20 ms** |
| It beats a transparent baseline | `phc report` | AUROC **0.987** vs heuristic **0.724** |
| It is not a prompt classifier | `phc ablation --config configs/ablation.yaml` | prompt-only **0.853**, but a *template lookup table* also gets **0.808** — so that number is an artifact |
| …and here is the proof | `phc ablation --config configs/ablation.yaml --data data/phc_conversation_leakcontrol` | prompt-only collapses to **0.434** (chance) while the full model holds **0.961** |
| It is calibrated | `phc report` | ECE **0.050** (uncalibrated 0.092) |
| It is not seed-luck | `phc train --config configs/train.yaml --seeds 42,123,2024` | AUROC **0.9867 ± 0.0002** |
| The intervention is safe | `phc benchmark` | gates **PASS** — generated turns hallucinate at **0.021** vs base rate **0.633**; **97.1 %** of positives caught; **52.8 %** coverage |
| It generalises to unseen risk types | `phc stress` | leave-one-category-out **0.804 ± 0.285** |
| It isn't learning our own labeller | `phc stress` | scored against a label source it never saw: **0.779** |
| It transfers to external data | `phc external-eval --bootstrap 500` | context-coverage transfers: **0.705** on a HaluEval counterfactual |

**Lead with the leak-control row.** It is the strongest claim in the paper
and the one a reviewer will attack.

---

## Phase 3 — The questions you will be asked

### "Your benchmark is synthetic. Isn't 0.987 just measuring your own generator?"

**Yes, partly — and here's the measurement.** A lookup table over prompt
templates reaches **0.808** without reading the conversation, because each
category's template identifies the category. So we built a leak-controlled
variant with one shared prompt pool: the prompt-only ablation collapses to
**0.434** (chance) and the full model holds **0.961**. The signal was in the
conversation all along; the artifact was available to baselines, not to the
model.

### "Does it work on real data?"

Honest answer: **discrimination of already-generated hallucinations does not
transfer; context coverage does.** On turn-aligned DiaHalu the stratified
AUROC is **0.510** — no signal. On a HaluEval counterfactual (same public
prompts, context changed) it is **0.705** — real signal, for the capability
the framework is actually about.

### "The HaluEval pair scores exactly 0.500. Isn't that a failure?"

No — it's the opposite. The right/hallucinated pair shares its prompt,
history and evidence exactly, so no pre-generation system *can* separate it.
Getting exactly 0.500 proves the score isn't using the answer. It's a
**purity check**, and I report it as one.

### "What if the same model just memorised the template?"

That's the leak probe, and it's in every table. It also isn't memorisation
of a *model* — the model was trained on PHC-Conversation and applied
zero-shot to DiaHalu and HaluEval.

### "Why should I trust the calibration?"

Because the threshold is fit on validation only, the coverage requirement is
imposed as a constraint with a one-sided 95 % margin so it holds on a fresh
sample, and the same analysis reports the out-of-fold figure next to the
in-sample one. And the ECE is reported against a *template-lookup baseline*,
not against nothing.

### "How fast is it?"

**20.5 ms median**, 23 ms p95, warm, on one machine, including SHAP
attribution. No network calls, no API keys, no GPU.

### "Does the guard block the user?"

No. Blocking actions require an explicit *send anyway*, and if the service
is down the extension **fails open** after 3 seconds. A guard that bricks
your chat is worse than no guard.

---

## Phase 4 — What to concede

Conceding these buys you credibility for the rest. Have them ready, don't
defend them:

1. **Human validation is pending.** The packet, the annotator instrument and
   the analysis pipeline exist and have been rehearsed against a synthetic
   round, but *no human labels exist yet*. Say it plainly.
2. **The rubric has a known defect**, found by the dry run: a support-based
   question cannot reach a *reasoning* error whose facts are all present.
   The protocol has been amended with a question about whether the reasoning
   is sound.
3. **DiaHalu shows no transfer**, and the stratified number (0.510) is the
   honest one, not the raw 0.431.
4. **The prompt-only ablation of 0.853 is an artifact** — quote 0.434
   instead.
5. **The benchmark is still ours.** The leak-controlled variant removes one
   specific artifact; it is not a substitute for independent labels.

---

## Phase 5 — If the demo breaks

| Problem | Fix |
|---|---|
| No panel appears | extension not loaded; or the page's markup doesn't match the history selectors — check for `data-message-author-role` |
| "service offline" | `phc serve` isn't running, or the popup's endpoint is wrong |
| Extension loads, risk always ~0 | same markup problem: the guard is reading an empty conversation |
| `--load-extension` does nothing | expected on Chrome 137+; load it from `chrome://extensions` |
| No `phc` command | `source .venv/bin/activate`, or `pip install -e .` |
| Paper won't build | `tectonic` not installed → `phc paper --no-compile` |
| Machine is offline | the demo is fully local; nothing needs the network |

**Fallback if Chrome misbehaves:** the demo works without a browser —
`phc risk` against the same two history files tells the identical story
(0.89 vs 0.14).

---

## One-sentence summary to close on

> *The risk of a bad next turn lives in the conversation, not in the prompt —
> and here's a benchmark that cannot be solved by pattern-matching the
> question.*
