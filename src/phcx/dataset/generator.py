"""PHC-Conversation benchmark generator (spec section 19).

OUR RESEARCH DATASET -- not an established external benchmark.

Constructs controlled multi-turn conversations with known ground truth across
11 categories (10 risk categories plus a benign control).  For every sample we
store:

    prompt, history, evidence, expected answer, generated (simulated) answer,
    claims, hallucination label, provenance / contradiction / temporal /
    ambiguity / presupposition / contamination labels, category, family id.

Labels:
    label_gold      -- by construction (would the default policy hallucinate?)
    label_verified  -- from the post-hoc Verifier applied to the generated answer
    label           -- training/evaluation target = label_verified when available
                       else label_gold (always recorded, agreement reported)

All content is synthetic; no private user conversations are collected.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..memory import ConversationMemory
from ..utils.text import sent_split, similarity
from .facts import FactFamily, build_families

CATEGORIES = [
    "supported_context",
    "unsupported_context",
    "contradictory_context",
    "model_memory_contamination",
    "entity_ambiguity",
    "temporal_inconsistency",
    "unsupported_presupposition",
    "referential_ambiguity",
    "evidence_conflict",
    "long_context_degradation",
    "short_benign",
]

_TURNS = ("user", "assistant")

#: Categories whose risk signal lives in the conversation state and the
#: evidence rather than in the wording of the prompt.  The leak-controlled
#: benchmark is built from these with one shared prompt pool, so that the
#: prompt cannot identify the risk type.  The four excluded categories are
#: excluded on purpose: a presupposition, a referential ambiguity and a
#: freshness cue *are* properties of the question, and no amount of
#: re-wording can move them into the context.
CATEGORIES_CONTEXT_BORNE = [
    "supported_context",
    "unsupported_context",
    "contradictory_context",
    "model_memory_contamination",
    "entity_ambiguity",
    "evidence_conflict",
    "long_context_degradation",
]

#: The prompt pool of the leak-controlled benchmark.  Every category draws
#: from the same pool, so a lookup table over prompt templates learns
#: nothing: after masking the entity slot all of these are one template.
NEUTRAL_PROMPTS = [
    "What should you say next about the {topic} project?",
    "Please answer the question about the {topic} project.",
    "Continue the conversation about the {topic} project.",
]


def _u(text: str) -> Dict[str, str]:
    return {"role": "user", "content": text}


def _a(text: str, source: str = "") -> Dict[str, Any]:
    d: Dict[str, Any] = {"role": "assistant", "content": text}
    if source:
        d["source"] = source
        d["verified"] = True   # claim attributed to and checked against a doc
    return d


def _doc(doc_id: str, text: str, source: str = "internal-docs", reliability: float = 0.85,
         timestamp: str = "") -> Dict[str, Any]:
    return {"doc_id": doc_id, "text": text, "source": source,
            "reliability": reliability, "timestamp": timestamp}


@dataclass
class _Case:
    history: List[Dict[str, Any]]
    prompt: str
    evidence: List[Dict[str, Any]]
    expected_answer: str
    generated_answer: str
    labels: Dict[str, int] = field(default_factory=dict)
    notes: str = ""


class SimulatedAssistant:
    """Deterministic stand-in for an LLM with typical (unverified-memory
    reusing) behaviour.  Used only to produce the generated answer that the
    post-hoc Verifier then labels.  It is NOT used at prediction time.
    """

    def __init__(self, carefulness: float = 0.5, seed: int = 0):
        self.carefulness = carefulness
        self.rng = random.Random(seed)

    # prefixes that mark instructions/acknowledgements rather than facts
    _INSTR = (
        "record ", "remember", "note ", "keep ", "tell me", "run ", "please",
        "summarize", "update ", "thanks", "okay", "great", "good ", "continue",
        "by the way", "just ", "we built", "and ", "sure", "surely", "nothing new",
        "based on our conversation",
    )

    @classmethod
    def _answerable_claim(cls, c) -> bool:
        t = c.text.strip().lower()
        return not t.startswith(cls._INSTR)

    @staticmethod
    def _answerable_sentence(sent: str) -> bool:
        from ..utils.text import content_tokens

        t = sent.strip()
        if not t or t.endswith(":") or len(content_tokens(t)) < 3:
            return False
        return True

    @staticmethod
    def _mask(text: str, family: FactFamily) -> str:
        """Remove family-specific names so responsiveness is judged on the
        question's content, not on shared entity tokens."""
        t = text
        for tok in (family.topic, family.alt_dataset, family.dataset,
                    family.alt_model, family.model, family.tool):
            if tok:
                t = t.replace(tok, " ")
        return " ".join(t.split())

    def answer(self, case: _Case, family: FactFamily, category: str) -> str:
        mem = ConversationMemory.from_history(case.history)
        ev_texts = [e["text"] for e in case.evidence]
        contam = mem.contamination(case.prompt)

        # 1) reuse unverified model memory (classic contamination behaviour)
        if contam.score >= 0.30 and self.rng.random() > self.carefulness * 0.6:
            top = contam.dependent_claims[0]
            return f"Based on our conversation, {top.text[0].lower() + top.text[1:]}"

        pmask = self._mask(case.prompt, family)

        # 2) answer from evidence: quote the single best-matching *sentence*.
        #    The doc must be relevant to the prompt as a whole; inside a
        #    relevant doc even a loosely-overlapping sentence counts.
        if ev_texts:
            doc_relevant = max(similarity(case.prompt, t) for t in ev_texts) >= 0.30
            gate = 0.08 if doc_relevant else 0.18
            best_sent, best_s = "", 0.0
            for t in ev_texts:
                for sent in sent_split(t):
                    if not self._answerable_sentence(sent):
                        continue
                    sc = similarity(case.prompt, sent)
                    if sc > best_s:
                        best_s, best_sent = sc, sent
            if best_s >= gate:
                return best_sent

        # 3) confident answer from memory that actually addresses the question
        #    (may be stale, contradicted, or unverified -> the verifier decides)
        rel = []
        for c, sc in mem.relevant_claims(case.prompt, top_k=8, min_sim=0.20):
            if not self._answerable_claim(c) or sc < 0.42:
                continue
            if similarity(pmask, self._mask(c.text, family)) >= 0.25:
                rel.append((c, sc))
        if rel:
            return rel[0][0].text

        # 4) fabricate a responsive but unsupported answer (hallucination)
        if category == "unsupported_context":
            return (
                f"The {family.topic} training job used {int(family.value_a)} GPU hours "
                f"and {int(family.value_c)} GB of memory."
            )
        if category == "unsupported_presupposition":
            return (
                f"The improvement was driven by better feature engineering and a larger "
                f"{family.dataset} training split."
            )
        if category == "temporal_inconsistency":
            return f"The current version is {family.value_c}% with the latest release from {family.alt_year}."
        if category in ("entity_ambiguity", "referential_ambiguity", "long_context_degradation",
                        "contradictory_context"):
            return f"We use {family.alt_dataset} for evaluation."
        return (
            f"The {family.metric} on {family.dataset} is {family.value_c}%, "
            f"which is the most recent figure we have."
        )


class PHCConversationGenerator:
    def __init__(
        self,
        samples_per_category: int = 60,
        seed: int = 42,
        n_families: int = 120,
        long_context_extra_turns: Tuple[int, ...] = (30, 45, 60),
        carefulness: float = 0.5,
        prompt_mode: str = "category",
    ):
        self.samples_per_category = samples_per_category
        self.seed = seed
        self.families = build_families(n_families, seed=seed)
        self.long_context_extra_turns = long_context_extra_turns
        self.assistant = SimulatedAssistant(carefulness=carefulness, seed=seed)
        self.prompt_mode = prompt_mode

    # ------------------------------------------------------------------ api
    def generate(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        rng = random.Random(self.seed)
        categories = (CATEGORIES if self.prompt_mode == "category"
                      else CATEGORIES_CONTEXT_BORNE)
        for cat in categories:
            k = self.samples_per_category
            for i in range(k):
                fam = self.families[rng.randrange(len(self.families))]
                case = self._build(cat, fam, rng)
                if case is None:
                    continue
                if self.prompt_mode == "neutral":
                    # the prompt is drawn from a pool shared by every
                    # category, so it cannot identify the risk type
                    case.prompt = rng.choice(NEUTRAL_PROMPTS).format(topic=fam.topic)
                out.append(self._package(case, fam, cat, rng))
        rng.shuffle(out)
        for i, r in enumerate(out):
            r["sample_id"] = f"phc_{i:05d}"
        return out

    # ------------------------------------------------------------- packaging
    def _package(self, case: _Case, fam: FactFamily, cat: str, rng: random.Random) -> Dict[str, Any]:
        from ..verification import Verifier

        gen = case.generated_answer or self.assistant.answer(case, fam, cat)
        case.generated_answer = gen
        from ..retrieval.local import RetrievedPassage

        passages = [
            RetrievedPassage(doc_id=e["doc_id"], text=e["text"], score=1.0,
                             source=e.get("source", "internal-docs"),
                             timestamp=e.get("timestamp") or None,
                             reliability=e.get("reliability", 0.85))
            for e in case.evidence
        ]
        ver = Verifier().verify(
            gen, prompt=case.prompt,
            memory=ConversationMemory.from_history(case.history),
            passages=passages,
        )
        label_verified = int(ver.label)
        label_gold = int(case.labels.get("hallucination", label_verified))
        label = label_verified if (label_verified or label_gold) else 0
        # if gold says hallucination but verifier missed it, keep gold as target
        # (verifier false-negatives are recorded, not silently dropped)
        if label_gold == 1 and label_verified == 0:
            label = 1

        return {
            "family_id": fam.family_id,
            "category": cat,
            "synthetic": True,
            "prompt": case.prompt,
            "conversation_history": case.history,
            "evidence": case.evidence,
            "expected_answer": case.expected_answer,
            "generated_answer": gen,
            "claims": [c.text for c in _claims_of(gen)],
            "label": label,
            "label_gold": label_gold,
            "label_verified": label_verified,
            "verifier": ver.to_dict(),
            "provenance_labels": case.labels.get("provenance", {}),
            "contradiction_label": int(case.labels.get("contradiction", 0)),
            "temporal_label": int(case.labels.get("temporal", 0)),
            "ambiguity_label": int(case.labels.get("ambiguity", 0)),
            "presupposition_label": int(case.labels.get("presupposition", 0)),
            "contamination_label": int(case.labels.get("contamination", 0)),
            "model": "simulated-scripted",
            "n_turns": len(case.history),
            "notes": case.notes,
        }

    # --------------------------------------------------------- category build
    def _build(self, cat: str, fam: FactFamily, rng: random.Random) -> Optional[_Case]:
        builder = getattr(self, f"_cat_{cat}", None)
        if builder is None:
            raise NotImplementedError(cat)
        return builder(fam, rng)

    # 0 ------------------------------------------------------ short benign turn
    def _cat_short_benign(self, fam: FactFamily, rng: random.Random) -> _Case:
        """Cover first-turn greetings and other non-factual chat prompts.

        The family token keeps synthetic prompts unique while the empty
        history/evidence reflects the real first-turn deployment case.
        """
        variant = rng.choice(("hi", "hello", "thanks", "okay", "hey"))
        prompt = f"{variant} {fam.family_id}"
        answers = {
            "hi": "Hello! How can I help you today?",
            "hello": "Hello! How can I help you today?",
            "thanks": "You're welcome!",
            "okay": "Sounds good.",
            "hey": "Hey! How can I help?",
        }
        return _Case(
            history=[],
            prompt=prompt,
            evidence=[],
            expected_answer=answers[variant],
            generated_answer=answers[variant],
            labels={
                "hallucination": 0,
                "contradiction": 0,
                "temporal": 0,
                "ambiguity": 0,
                "presupposition": 0,
                "contamination": 0,
                "provenance": {},
            },
            notes="benign first-turn social prompt",
        )

    # 1 ---------------------------------------------------- supported context
    def _cat_supported_context(self, fam: FactFamily, rng: random.Random) -> _Case:
        source = f"doc:{fam.family_id}-spec"
        history = [
            _u(f"Tell me about the {fam.topic} project."),
            _a(f"The {fam.topic} project works on {fam.domain}.", source),
            _u("Which dataset do we use?"),
            _a(f"The project uses {fam.dataset} for evaluation.", source),
            _u("And the baseline?"),
            _a(f"The baseline model is {fam.model}.", source),
        ]
        prompt_q = rng.choice([
            f"Which dataset does the {fam.topic} project use for evaluation?",
            f"Which dataset does the {fam.topic} project evaluate on?",
            f"Which model is the baseline for the {fam.topic} project?",
        ])
        evidence = [
            _doc(f"{fam.family_id}-spec",
                 f"The {fam.topic} project works on {fam.domain} and evaluates on {fam.dataset}. "
                 f"The baseline model is {fam.model}. "
                 f"{fam.metric} on {fam.dataset} is {fam.value_a}%.",
                 timestamp="2026-05-01"),
        ]
        expected = fam.dataset if "dataset" in prompt_q.lower() else fam.model
        labels = {"hallucination": 0, "contradiction": 0, "temporal": 0,
                  "ambiguity": 0, "presupposition": 0, "contamination": 0,
                  "provenance": {"verified": True}}
        # hard positive: well-sourced context, but the question needs information
        # that exists nowhere -> high coverage signals yet a fabricated answer
        if rng.random() < 0.35:
            prompt_q = f"What is the GPU-hour training cost of the {fam.topic} project?"
            expected = "Not answerable; the system should abstain or retrieve."
            labels = {"hallucination": 1, "contradiction": 0, "temporal": 0,
                      "ambiguity": 0, "presupposition": 0, "contamination": 0,
                      "provenance": {"verified": True, "missing_info": True}}
        return _Case(history, prompt_q, evidence,
                     expected, "", labels, notes="supported context")

    # 2 -------------------------------------------------- unsupported context
    def _cat_unsupported_context(self, fam: FactFamily, rng: random.Random) -> _Case:
        history = [
            _u(f"We built the {fam.topic} pipeline on {fam.tool}."),
            _a(f"The training pipeline is built with {fam.tool}."),
            _u("Great, keep going."),
            _a("Sure, continuing with the next step."),
        ]
        prompt = rng.choice([
            f"What is the memory footprint of the {fam.topic} training job?",
            f"How many GPU hours did the {fam.topic} pretraining take?",
            f"What was the budget approved for the {fam.topic} project?",
        ])
        evidence = [
            _doc(f"{fam.family_id}-ops", f"{fam.topic} uses {fam.tool} for training.", timestamp="2026-04-11")
        ]
        expected = "The context does not contain this information; the system should abstain or retrieve."
        labels = {"hallucination": 1, "contradiction": 0, "temporal": 0, "ambiguity": 0,
                  "presupposition": 0, "contamination": 0,
                  "provenance": {"verified": False, "missing_info": True}}
        # hard negative: evidence actually answers it
        if rng.random() < 0.35:
            evidence = [_doc(f"{fam.family_id}-ops2",
                             f"The {fam.topic} training job consumed 40 GPU hours and 1.2 TB memory.",
                             timestamp="2026-06-01")]
            labels["hallucination"] = 0
        return _Case(history, prompt, evidence,
                     "Not answerable from context without retrieval.", "", labels,
                     notes="unsupported context")

    # 3 ------------------------------------------------- contradictory context
    def _cat_contradictory_context(self, fam: FactFamily, rng: random.Random) -> _Case:
        history = [
            _u("Which dataset does the project use?"),
            _a(f"The project uses {fam.dataset} for evaluation."),
            _u("Please continue the analysis."),
            _a("Continuing with the evaluation plan."),
            _u("Remind me about the setup later."),
            _a(f"The project uses {fam.alt_dataset} for evaluation."),
        ]
        prompt = rng.choice([
            f"Summarize the evaluation plan for the {fam.topic} project.",
            f"Which dataset are we evaluating {fam.topic} on?",
            f"Run the evaluation for {fam.topic} and report {fam.metric}.",
        ])
        evidence = [] if rng.random() < 0.5 else [
            _doc(f"{fam.family_id}-spec",
                 f"The evaluation dataset for the {fam.topic} project is {fam.dataset}.",
                 timestamp="2026-05-01")
        ]
        labels = {"hallucination": 1 if not evidence else 0, "contradiction": 1,
                  "temporal": 0, "ambiguity": 1, "presupposition": 0,
                  "contamination": 0, "provenance": {"verified": bool(evidence)}}
        expected = "Detect conflicting memory and clarify which dataset is correct."
        return _Case(history, prompt, evidence, expected, "", labels,
                     notes="contradictory context")

    # 4 ----------------------------------------------- model memory contamination
    def _cat_model_memory_contamination(self, fam: FactFamily, rng: random.Random) -> _Case:
        false_claim = f"The {fam.dataset} contains 50,000 samples."
        history = [
            _u(f"The project uses {fam.dataset} for evaluation, right?"),
            _a(f"The project uses {fam.dataset} for evaluation."),
            _u("How big is it?"),
            _a(false_claim),          # MODEL_GENERATED, unverified
            _u("Thanks, noted."),
        ]
        prompt = rng.choice([
            f"How many samples did {fam.dataset} contain? Analyze the 50,000 samples for {fam.topic}.",
            f"Based on the 50,000 samples in {fam.dataset}, what should the {fam.topic} report say?",
            f"Summarize statistics over the 50,000 samples of {fam.dataset} for {fam.topic}.",
        ])
        if rng.random() < 0.35:
            # hard negative: the claim WAS verified (sourced turn + confirming datasheet)
            history = [
                _u(f"The project uses {fam.dataset} for evaluation, right?"),
                _a(f"The project uses {fam.dataset} for evaluation."),
                _u("How big is it?"),
                _a(false_claim, source=f"doc:{fam.family_id}-datasheet"),
                _u("Thanks, noted."),
            ]
            evidence = [
                _doc(f"{fam.family_id}-datasheet",
                     f"The {fam.topic} datasheet for {fam.dataset} lists 50,000 samples in 3 splits.",
                     timestamp="2026-03-19"),
            ]
            labels = {"hallucination": 0, "contradiction": 0, "temporal": 0,
                      "ambiguity": 0, "presupposition": 0, "contamination": 1,
                      "provenance": {"model_generated": True, "verified": True}}
        else:
            evidence = [] if rng.random() < 0.5 else [
                _doc(f"{fam.family_id}-datasheet",
                     f"The {fam.topic} datasheet for {fam.dataset} lists 12,000 annotated samples.",
                     timestamp="2026-03-19")
            ]
            # when the datasheet is present, the false claim is contradicted
            labels = {"hallucination": 0 if evidence else 1, "contradiction": int(bool(evidence)),
                      "temporal": 0, "ambiguity": 0, "presupposition": 0,
                      "contamination": 1, "provenance": {"model_generated": True,
                                                         "verified": bool(evidence)}}
        expected = ("Flag MODEL_GENERATED + UNVERIFIED dependency; verify before answering.")
        return _Case(history, prompt, evidence, expected, "", labels,
                     notes="model-memory contamination")

    # 5 ------------------------------------------------------ entity ambiguity
    def _cat_entity_ambiguity(self, fam: FactFamily, rng: random.Random) -> _Case:
        history = [
            _u(f"We maintain two datasets: {fam.dataset} and {fam.alt_dataset}."),
            _a(f"Noted: {fam.dataset} and {fam.alt_dataset} are both used by the project."),
            _u("Which one has more rows?"),
            _a(f"{fam.dataset} has more rows in the last snapshot."),
            _u("Continue please."),
        ]
        prompt = rng.choice([
            f"Summarize the {fam.topic} dataset statistics.",
            f"Which {fam.topic} dataset should we benchmark on?",
            f"Compare the {fam.topic} dataset results.",
        ])
        evidence = [
            _doc(f"{fam.family_id}-ds1",
                 f"{fam.dataset} contains 12,000 rows and reaches {fam.metric} {fam.value_a}%.",
                 timestamp="2026-02-02"),
            _doc(f"{fam.family_id}-ds2",
                 f"{fam.alt_dataset} contains 9,500 rows and reaches {fam.metric} {fam.value_c}%.",
                 timestamp="2026-02-03"),
        ] if rng.random() < 0.5 else []
        labels = {"hallucination": 1 if not evidence else 0, "contradiction": 0,
                  "temporal": 0, "ambiguity": 1, "presupposition": 0,
                  "contamination": 0, "provenance": {"verified": bool(evidence)}}
        expected = "Disambiguate which dataset is meant before answering."
        return _Case(history, prompt, evidence, expected, "", labels, notes="entity ambiguity")

    # 6 -------------------------------------------------- temporal inconsistency
    def _cat_temporal_inconsistency(self, fam: FactFamily, rng: random.Random) -> _Case:
        history = [
            _u("What version of the tooling are we on?"),
            _a(f"We are on {fam.tool} with the configuration from {fam.year}."),
            _u("And the project status?"),
            _a(f"The project started in {fam.year} and is unchanged since then."),
        ]
        prompt = rng.choice([
            f"What is the current version of the {fam.topic} tooling?",
            f"What is the latest configuration for the {fam.topic} project?",
            f"Give me the current status of the {fam.topic} project.",
        ])
        evidence = [] if rng.random() < 0.5 else [
            _doc(f"{fam.family_id}-release",
                 f"The {fam.topic} tooling now runs {fam.tool} version 6.1, released in 2026.",
                 timestamp="2026-09-15")
        ]
        labels = {"hallucination": 1 if not evidence else 0, "contradiction": 0,
                  "temporal": 1, "ambiguity": 0, "presupposition": 0,
                  "contamination": 0, "provenance": {"verified": bool(evidence)}}
        if rng.random() < 0.40:
            # hard negative: memory is already refreshed from a trusted source
            history.append(_u("Did anything change since then?"))
            history.append(_a(
                f"{fam.tool} was upgraded to version 6.1 in 2026 for {fam.topic}.",
                source=f"doc:{fam.family_id}-release"))
            evidence = [
                _doc(f"{fam.family_id}-release",
                     f"The {fam.topic} tooling now runs {fam.tool} version 6.1, released in 2026.",
                     timestamp="2026-09-15"),
            ]
            labels = {"hallucination": 0, "contradiction": 0, "temporal": 0,
                      "ambiguity": 0, "presupposition": 0, "contamination": 0,
                      "provenance": {"verified": True}}
        expected = ("Treat memory as stale; retrieve current source before answering.")
        return _Case(history, prompt, evidence, expected, "", labels, notes="temporal inconsistency")

    # 7 ---------------------------------------------- unsupported presupposition
    def _cat_unsupported_presupposition(self, fam: FactFamily, rng: random.Random) -> _Case:
        history = [
            _u(f"Record the {fam.metric} for {fam.topic}."),
            _a(f"{fam.metric} on {fam.dataset} is {fam.value_a}%."),
            _u("Keep tracking it."),
            _a("Understood, tracking continues."),
        ]
        prompt = rng.choice([
            f"Why did {fam.metric} improve from {fam.value_a}% to {fam.value_b}% on {fam.topic}?",
            f"What caused the {fam.metric} jump to {fam.value_b}% for {fam.topic}?",
            f"How did the {fam.topic} project achieve {fam.value_b}% {fam.metric}?",
        ])
        evidence = [] if rng.random() < 0.5 else [
            _doc(f"{fam.family_id}-report",
                 f"{fam.topic} weekly report: {fam.metric} improved from {fam.value_a}% "
                 f"to {fam.value_b}% after data cleaning.", timestamp="2026-06-20")
        ]
        labels = {"hallucination": 1 if not evidence else 0, "contradiction": 0,
                  "temporal": 0, "ambiguity": 0, "presupposition": 1,
                  "contamination": 0, "provenance": {"verified": bool(evidence)}}
        expected = "Detect the unsupported presupposition; ask for evidence or clarify."
        return _Case(history, prompt, evidence, expected, "", labels,
                     notes="unsupported presupposition")

    # 8 ---------------------------------------------------- referential ambiguity
    def _cat_referential_ambiguity(self, fam: FactFamily, rng: random.Random) -> _Case:
        history = [
            _u(f"The {fam.topic} service depends on {fam.model}."),
            _a(f"Noted, {fam.model} is used by the {fam.topic} service."),
            _u("And the batch job?"),
            _a(f"The batch job depends on {fam.alt_model}."),
            _u("Okay, continue."),
            _a("Continuing."),
        ]
        prompt = rng.choice([
            f"What is its accuracy on the held-out set for {fam.topic}?",
            f"How does it handle long inputs in the {fam.topic} service?",
            f"Update that model's configuration for {fam.topic} and report the result.",
        ])
        evidence = [
            _doc(f"{fam.family_id}-eval",
                 f"{fam.model} held-out {fam.metric} is {fam.value_a}%; "
                 f"{fam.alt_model} held-out {fam.metric} is {fam.value_c}%.",
                 timestamp="2026-05-30"),
        ] if rng.random() < 0.5 else []
        labels = {"hallucination": 1 if not evidence else 0, "contradiction": 0,
                  "temporal": 0, "ambiguity": 1, "presupposition": 0,
                  "contamination": 0, "provenance": {"verified": bool(evidence)}}
        if rng.random() < 0.35:
            # hard negative: the reference is disambiguated in conversation
            history.append(_u("Just to be sure, which one did I mean?"))
            history.append(_a(
                f"By 'it' you mean the {fam.model} baseline used by the {fam.topic} service.",
                source=f"doc:{fam.family_id}-eval"))
            labels = {"hallucination": 0, "contradiction": 0, "temporal": 0,
                      "ambiguity": 0, "presupposition": 0, "contamination": 0,
                      "provenance": {"verified": True}}
        expected = "Resolve 'it/that' to a unique antecedent or ask for clarification."
        return _Case(history, prompt, evidence, expected, "", labels,
                     notes="referential ambiguity")

    # 9 ------------------------------------------------------- evidence conflict
    def _cat_evidence_conflict(self, fam: FactFamily, rng: random.Random) -> _Case:
        history = [
            _u("What metric did we report last week?"),
            _a(f"We reported {fam.metric} = {fam.value_a}% on {fam.dataset}."),
            _u("Good, keep that as the reference."),
            _a("Keeping it as the reference figure."),
        ]
        prompt = rng.choice([
            f"What {fam.metric} should we publish for {fam.dataset} in {fam.topic}?",
            f"Confirm the {fam.metric} result for the {fam.topic} paper.",
            f"Report the {fam.metric} we should submit for {fam.topic}.",
        ])
        evidence = [
            _doc(f"{fam.family_id}-neweval",
                 f"A re-run on {fam.dataset} reports {fam.metric} of {fam.value_b}%, "
                 f"which supersedes the earlier figure.", timestamp="2026-07-08"),
        ]
        labels = {"hallucination": 1, "contradiction": 1, "temporal": 0, "ambiguity": 0,
                  "presupposition": 0, "contamination": 0,
                  "provenance": {"verified": True, "conflict": True}}
        if rng.random() < 0.35:
            # hard negative: memory already updated to the re-run figure
            history = [
                _u("What metric did we report last week?"),
                _a(f"We reported {fam.metric} = {fam.value_b}% on {fam.dataset}.",
                   source=f"doc:{fam.family_id}-neweval"),
                _u("Good, keep that as the reference."),
                _a("Keeping it as the reference figure."),
            ]
            labels = {"hallucination": 0, "contradiction": 0, "temporal": 0, "ambiguity": 0,
                      "presupposition": 0, "contamination": 0,
                      "provenance": {"verified": True, "conflict": False}}
        expected = ("Flag evidence/memory conflict and ask which figure to use.")
        return _Case(history, prompt, evidence, expected, "", labels, notes="evidence conflict")

    # 10 ----------------------------------------------- long-context degradation
    def _cat_long_context_degradation(self, fam: FactFamily, rng: random.Random) -> _Case:
        n_distract = rng.choice(self.long_context_extra_turns)
        history: List[Dict[str, Any]] = [
            _u(f"Remember this: the {fam.topic} project uses {fam.dataset}."),
            _a(f"The project uses {fam.dataset} for evaluation."),
        ]
        filler_topics = ["weather", "lunch", "git commits", "meetings", "logs", "tickets"]
        for i in range(n_distract):
            topic = filler_topics[i % len(filler_topics)]
            if i % 2 == 0:
                history.append(_u(f"By the way, anything new about the {topic}?"))
            else:
                history.append(_a(f"Nothing new on the {topic}; we can move on."))
        # late-turn drift: a conflicting statement appears far later
        if rng.random() < 0.6:
            history.append(_u("Quick reminder of the dataset?"))
            history.append(_a(f"We use {fam.alt_dataset} for evaluation."))
        prompt = rng.choice([
            f"Which dataset does the {fam.topic} project use for evaluation? Answer briefly.",
            f"Remind me which dataset {fam.topic} evaluates on.",
        ])
        evidence = [] if rng.random() < 0.5 else [
            _doc(f"{fam.family_id}-spec",
                 f"The evaluation dataset for the {fam.topic} project is {fam.dataset}.",
                 timestamp="2026-05-01")
        ]
        labels = {"hallucination": 1 if not evidence else 0,
                  "contradiction": 1 if any(fam.alt_dataset in str(h) for h in history) else 0,
                  "temporal": 0, "ambiguity": 0, "presupposition": 0,
                  "contamination": 0, "provenance": {"verified": bool(evidence)}}
        expected = "Recover the early fact despite long distractor context."
        return _Case(history, prompt, evidence, expected, "", labels,
                     notes=f"long context ({n_distract} distractor turns)")


def _claims_of(text: str):
    from ..memory.extract import extract_claims
    from ..memory.types import Turn

    t = Turn(turn_id=1, speaker="assistant", text=text)
    return extract_claims(t)
