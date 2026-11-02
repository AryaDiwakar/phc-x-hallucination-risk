"""Referential ambiguity detection (spec section 10).

Resolves demonstratives, pronouns and definite descriptions against
conversation history.  When several antecedents are nearly equally plausible
-- or when a reference cannot be resolved at all -- the ambiguity score rises
and the recommended action becomes CLARIFY.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..entity.tracker import EntityTracker
from ..memory import ConversationMemory
from ..utils.text import content_tokens, similarity

_NP_REF_RE = re.compile(
    r"\b("
    r"(?:the|this|that|our|my|your|previous|earlier|last|above|mentioned|aforementioned)"
    # a noun phrase of up to three words, each possibly hyphenated.  The
    # quantifier must not stop at an internal word boundary, or "the
    # held-out set" would be captured as the meaningless fragment "the held".
    r"\s+(?:[\w]+(?:-[\w]+)*)(?:\s+(?:[\w]+(?:-[\w]+)*)){0,2}"
    r"|that dataset|the dataset|that model|the model|the algorithm|our project|"
    r"previous(?:ly)?\s+(?:answer|result|model|dataset|claim|statement|approach)|"
    r"earlier\s+(?:answer|result|model|dataset|claim|statement|approach)"
    r")\b",
    re.I,
)
_BARE_RE = re.compile(r"\b(it|they|them|these|those|he|she)\b", re.I)
_BARE = {"it", "they", "them", "these", "those", "he", "she"}
_STOP_REF = {"the dataset", "the model", "this", "that", "it"}
#: function words that must not end a captured noun phrase ("that model
#: configuration and" -> "that model configuration")
_TRAILING_STOP = {
    "and", "or", "but", "to", "for", "of", "in", "on", "at", "with", "from",
    "the", "a", "an", "is", "are", "was", "were", "as", "by", "that", "this",
}


@dataclass
class AmbiguityResult:
    score: float = 0.0
    references: List[Dict[str, Any]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    should_clarify: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "references": self.references[:8],
            "reasons": self.reasons,
            "should_clarify": self.should_clarify,
        }


class AmbiguityEngine:
    def __init__(self, margin: float = 0.12, min_candidates: int = 2,
                 min_relevance: float = 0.10):
        self.margin = margin
        self.min_candidates = min_candidates
        #: a candidate must be at least this relevant to count as a plausible
        #: antecedent; below it the reference is unmatched, not ambiguous
        self.min_relevance = min_relevance
        self._entities = EntityTracker()

    def _candidates(self, memory: ConversationMemory) -> List[Tuple[str, int, str]]:
        pool: List[Tuple[str, int, str]] = []
        for t in memory.turns:
            for s in re.split(r"(?<=[.!?;])\s+|\n+", t.text):
                s = s.strip()
                if s:
                    pool.append((s, t.turn_id, t.speaker))
        return pool

    def analyze(self, prompt: str, memory: ConversationMemory) -> AmbiguityResult:
        res = AmbiguityResult()
        if not prompt:
            return res
        pool = self._candidates(memory)
        if not pool:
            return res

        # entity inventory of the conversation (for definite NP resolution)
        ent_mentions: Dict[str, List[int]] = {}
        for t in memory.turns:
            for m in self._entities.extract(t.text, t.turn_id, t.speaker):
                ent_mentions.setdefault(m.entity.lower(), []).append(t.turn_id)
        prompt_ents = {m.entity.lower() for m in self._entities.extract(prompt, -1, "user")}

        def _score_reference(ref: str, context_query: str) -> List[Dict[str, Any]]:
            scored = sorted(
                ((similarity(context_query, s), s, tid, spk) for s, tid, spk in pool),
                key=lambda x: -x[0],
            )
            top = [x for x in scored[:6] if x[0] > 0.08]
            return [
                {"text": c[:120], "turn": tid, "speaker": spk, "relevance": round(sc, 3)}
                for sc, c, tid, spk in top[:3]
            ]

        def _mentions_reference(ref: str, sentence: str) -> bool:
            """Does the sentence share a content word with the reference?

            A definite reference that names something specific ("the held-out
            set", "the Atlas A results") can only be ambiguous between
            statements that mention it.  Without this guard the hashed
            n-gram similarity of two short, unrelated strings is dominated by
            hash collisions, and a reference that matches nothing in the
            conversation gets flagged as ambiguous.
            """
            ref_words = {w for w in content_tokens(ref) if len(w) > 2}
            if not ref_words:
                return True
            sent_words = set(content_tokens(sentence))
            return bool(ref_words & sent_words)

        # ---------------- (1) definite / demonstrative NPs ---------------
        for m in _NP_REF_RE.finditer(prompt):
            ref = m.group(0).strip()
            # drop trailing function words the noun phrase ran into
            words = ref.split()
            while len(words) > 1 and words[-1].lower() in _TRAILING_STOP:
                words.pop()
            ref = " ".join(words)
            if len(ref.split()) < 2:
                continue
            if ref.lower() in _STOP_REF and len(ref.split()) < 3:
                # generic "the dataset"/"the model": resolve via entity inventory
                head = ref.lower().replace("the ", "").replace("that ", "")
                hits = [
                    (e, turns) for e, turns in ent_mentions.items()
                    if head[:5] in e or e[:5] in head
                ]
                if len(hits) >= 2:
                    res.references.append(
                        {
                            "reference": ref,
                            "candidates": [
                                {"entity": e, "turns": sorted(set(t))[:6]} for e, t in hits[:3]
                            ],
                            "margin": 0.0,
                            "type": "entity_slot",
                        }
                    )
                    res.reasons.append(
                        f"\"{ref}\" matches {len(hits)} distinct entities: "
                        + ", ".join(e for e, _ in hits[:3])
                    )
                continue
            # noun-headed reference: if the reference itself carries the
            # entity name and every candidate is about that same entity, the
            # referent is pinned -> not ambiguous (e.g. "the Orion project").
            ref_ents = {m2.entity.lower() for m2 in self._entities.extract(ref, -1, "user")}
            cands = _score_reference(ref, prompt)
            if ref_ents and cands and all(
                any(e in c["text"].lower() for e in ref_ents) for c in cands
            ):
                continue
            # a specific reference can only be ambiguous between statements
            # that actually mention it
            cands = [c for c in cands if _mentions_reference(ref, c["text"])]
            # Repeated mentions of the same explicit entity in the user and
            # assistant turns are supporting context, not competing
            # antecedents. Only treat a noun phrase as ambiguous when its
            # identifying words do not pin the same entity in each candidate.
            ref_words = {w for w in content_tokens(ref) if len(w) > 2}
            if len(ref_words) >= 2 and cands and all(
                ref_words.issubset(set(content_tokens(c["text"]))) for c in cands
            ):
                continue
            if len(cands) >= self.min_candidates:
                s1, s2 = cands[0]["relevance"], cands[1]["relevance"]
                margin = round(s1 - s2, 4)
                # A reference is only *ambiguous* if the leading candidate is
                # actually plausible.  A reference that matches nothing (say
                # "the held-out set" in a conversation about datasets) has
                # near-zero relevance and a tiny margin by chance, and
                # flagging it as ambiguous is a false positive.  The
                # bare-pronoun branch below applies the same guard.
                if s1 > self.min_relevance and margin < self.margin:
                    res.references.append({"reference": ref, "candidates": cands, "margin": margin, "type": "np"})
                    res.reasons.append(
                        f"\"{ref}\" has {len(cands)} plausible antecedents (margin={margin})"
                    )

        # ---------------- (2) bare pronouns --------------------------------
        for m in _BARE_RE.finditer(prompt):
            ref = m.group(0).lower()
            cands = _score_reference(ref, prompt)
            if len(cands) >= self.min_candidates and cands[0]["relevance"] > self.min_relevance:
                s1, s2 = cands[0]["relevance"], cands[1]["relevance"]
                margin = round(s1 - s2, 4)
                if margin < self.margin:
                    res.references.append(
                        {"reference": ref, "candidates": cands, "margin": margin, "type": "pronoun"}
                    )
                    res.reasons.append(
                        f"\"{ref}\" has {len(cands)} plausible antecedents (margin={margin})"
                    )
            elif len(ent_mentions) >= 2:
                # pronoun cannot be resolved to a single entity although several exist
                res.references.append(
                    {
                        "reference": ref,
                        "candidates": [
                            {"entity": e, "turns": sorted(set(v))[:6]}
                            for e, v in list(ent_mentions.items())[:3]
                        ],
                        "margin": 0.0,
                        "type": "unresolved_pronoun",
                    }
                )
                res.reasons.append(
                    f"\"{ref}\" cannot be resolved uniquely; {len(ent_mentions)} entities in scope"
                )

        # ---------------- score -------------------------------------------
        if res.references:
            numeric_margins = [r["margin"] for r in res.references if r.get("type") != "unresolved_pronoun"]
            worst = min(numeric_margins) if numeric_margins else 0.0
            unresolved = any(r.get("type") == "unresolved_pronoun" for r in res.references)
            base = 0.4 if unresolved else 0.35
            res.score = float(min(1.0, base + 0.6 * max(0.0, 1.0 - worst / max(self.margin, 1e-6))))
            res.should_clarify = True
        return res
