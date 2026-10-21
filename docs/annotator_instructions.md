# Annotator Instructions

You are annotating a sample of conversational turns for a research project
on pre-generation hallucination risk. The system under study proposes an
answer for each turn; your job is to judge the turn, **not** to judge the
system, and not to rewrite the answer.

Everything is local. The packet never leaves the machine, and your
annotator code is random — no name, no email, nothing that identifies you
is written down.

## How to run a session

```bash
phc annotate \
  --packet data/annotations/human_validation_sample.jsonl \
  --out data/annotations/annotations.jsonl
```

The session prints your random code (e.g. `A4271`), then shows one item at
a time. Re-running the same command with the same `--out` resumes where you
left off. If you want a fixed code (so you can annotate from two machines),
pass `--code A4271` yourself.

To preview a single item without answering it: `--dry-run`.

## What you see

Each item has four parts:

- **PROMPT** — the user's question in the turn being judged.
- **CONVERSATION SO FAR** — the turns before it. This may be long; you do
  not need to read every word, but do read enough to know what the assistant
  has already said.
- **EVIDENCE** — the passages the system was given, if any. **Some items
  have none.** That is deliberate: a question can be unanswerable from the
  conversation alone, and you are being asked to notice that.
- **PROPOSED ANSWER** — the answer the system would give.

## The five questions

1. **Is the answer factually supported by the supplied context?**
   `yes` / `no` / `uncertain`

   Judge against the evidence *and* the conversation. If the answer states a
   fact that appears nowhere in either, that is `no`. If it is there, `yes`.
   If the context is too vague to tell, `uncertain` — and please use it.
   "Uncertain" is a real answer, not a failure.

2. **Does it contradict the conversation history?**
   `yes` / `no` / `uncertain`

   Look for a number, a name or a claim in the answer that conflicts with
   something said earlier. This is the hardest question in the set: most
   answers do not conflict, and "uncertain" is a legitimate answer when you
   are not sure.

3. **Does it contain an unsupported presupposition or entity/time drift?**
   `yes` / `no` / `uncertain`

   A presupposition is something the answer takes for granted without the
   context establishing it ("we improved from 70% to 85%" presupposes that
   the improvement happened). Drift is a silent switch of the dataset, model
   or version the conversation was about.

4. **Would you allow generation without retrieval or clarification?**
   `allow` / `retrieve` / `clarify` / `abstain`

   This is a judgement about *this system, right now*. `allow` if the
   answer can go out as it is. `retrieve` if more context should be fetched
   first. `clarify` if the user should be asked. `abstain` if the question
   should not be answered at all.

5. **Confidence (1–5) and a short rationale.**

   1 = a guess, 3 = reasonably sure, 5 = obvious. Then one line of free
   text saying why. The rationale is read by a human, not by a model, so
   write it for a person: "the answer names a dataset that appears nowhere
   in the conversation" beats "unsupported".

## Things that will trip you up

- **Do not look up the answer.** Judge from the material in front of you.
  If you do not know a fact, that is a reason to say `uncertain`, not to
  guess.
- **Some items appear twice.** That is deliberate, and the repeats are not
  marked. Answer each one on its own merits — that is how intra-rater
  consistency gets measured.
- **The proposed answer is sometimes right.** It is not always wrong. Do
  not assume a trap.
- **Judge the turn, not the topic.** A perfectly reasonable question can
  have an unanswerable answer, and vice versa.

## What happens to your answers

Your annotations are written to `data/annotations/annotations.jsonl` with
your random code, and analysed with:

```bash
phc analyze-annotations \
  --annotations data/annotations/annotations.jsonl \
  --out results/raw/human_validation_report.json
```

The analysis reports inter-rater agreement, agreement per category,
human-vs-model AUROC, calibration by risk band, how often the system and
the annotators disagree about the action, and the false-block rate. No
individual annotation is published, and the annotator code is not linked to
anyone.

## Time

Roughly one to two minutes per item. A full session of 280 items is a
few hours; if you can only do 40, that is fine — the analysis handles
uneven coverage, and partial sessions are more useful than none.
