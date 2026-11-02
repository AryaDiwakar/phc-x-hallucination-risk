"""ConversationMemory: provenance-aware store over conversational claims.

Design rules
------------
1. Every statement is stored as a :class:`Claim` with explicit provenance.
2. Model-generated claims are *never* trusted by default.
3. The memory exposes a contamination score when a new prompt depends on an
   unverified model-generated claim (spec section 5).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..utils.text import content_tokens, similarity, tokenize
from .extract import extract_claims, _naive_entities
from .types import (
    Claim,
    EdgeType,
    MemoryEdge,
    ProvenanceType,
    Turn,
    VerificationStatus,
)

__all__ = ["ConversationMemory", "ContaminationResult"]


@dataclass
class ContaminationResult:
    score: float = 0.0
    dependent_claims: List[Claim] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "dependent_claims": [
                {
                    "claim_id": c.claim_id,
                    "text": c.text,
                    "turn_id": c.turn_id,
                    "provenance": c.provenance_type.value,
                    "verification": c.verification_status.value,
                    "trust": round(c.trust, 3),
                }
                for c in self.dependent_claims
            ],
            "reasons": self.reasons,
            "explanations": self.explanations,
        }


_NUM_TOKEN_RE = re.compile(r"^\d[\d,]*\.?\d*%?$")


def _significant_numbers(text: str) -> set:
    """Numeric tokens in ``text`` that carry enough specificity to be a fact."""
    out = set()
    for t in tokenize(text or ""):
        if not _NUM_TOKEN_RE.match(t):
            continue
        try:
            v = float(t.rstrip("%").replace(",", ""))
        except ValueError:
            continue
        if v >= 10:
            out.add(t)
    return out


class ConversationMemory:
    def __init__(self, conversation_id: str = "conv", turns: Optional[Sequence[Turn]] = None):
        self.conversation_id = conversation_id
        self.turns: List[Turn] = []
        self.edges: List[MemoryEdge] = []
        self.created_at = time.time()
        if turns:
            for t in turns:
                self.add_turn(t, extract=True)

    # ------------------------------------------------------------------ build
    def add_turn(
        self,
        turn: Turn,
        extract: bool = True,
        source: Optional[str] = None,
        retrieved: bool = False,
        verified: bool = False,
    ) -> Turn:
        from ..provenance.classifier import annotate_turn

        if extract and not turn.claims:
            turn.claims = extract_claims(turn)
        annotate_turn(turn, source=source, retrieved=retrieved, verified=verified)
        self.turns.append(turn)
        self._link_new_turn(turn)
        return turn

    @staticmethod
    def from_history(
        history: Sequence[Dict[str, Any]],
        conversation_id: str = "conv",
        default_speaker: str = "user",
    ) -> "ConversationMemory":
        """Build memory from API-style ``[{"role"/"speaker", "content"/"text"}]``."""
        mem = ConversationMemory(conversation_id=conversation_id)
        for i, h in enumerate(history):
            role = str(h.get("role") or h.get("speaker") or default_speaker).lower()
            speaker = {"human": "user", "ai": "assistant"}.get(role, role)
            text = str(h.get("content") or h.get("text") or "")
            turn = Turn(
                turn_id=int(h.get("turn_id", i + 1)),
                speaker=speaker,
                text=text,
                timestamp=float(h.get("timestamp", mem.created_at + i)),
                metadata=dict(h.get("metadata") or {}),
            )
            mem.add_turn(
                turn,
                source=h.get("source"),
                retrieved=bool(h.get("retrieved", False)),
                verified=bool(h.get("verified", False)),
            )
        return mem

    # ------------------------------------------------------------------ query
    @property
    def all_claims(self) -> List[Claim]:
        return [c for t in self.turns for c in t.claims]

    @property
    def model_claims(self) -> List[Claim]:
        return [c for c in self.all_claims if c.is_model_claim]

    @property
    def unverified_model_claims(self) -> List[Claim]:
        return [c for c in self.model_claims if c.is_unverified]

    def claims_by_speaker(self, speaker: str) -> List[Claim]:
        return [c for t in self.turns if t.speaker == speaker for c in t.claims]

    def relevant_claims(self, prompt: str, top_k: int = 8, min_sim: float = 0.05) -> List[Tuple[Claim, float]]:
        scored = []
        for c in self.all_claims:
            s = similarity(prompt, c.text)
            # provenance-neutral ranking: trust never *raises* relevance,
            # it is applied by downstream risk logic.
            if s >= min_sim:
                scored.append((c, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def recent_turns(self, n: int) -> List[Turn]:
        return self.turns[-n:] if n > 0 else list(self.turns)

    def recent(self, n: int) -> "ConversationMemory":
        """A read-only view over the most recent ``n`` turns.

        Claims are shared with the parent memory (they are not re-extracted),
        so the view is cheap and provenance annotations stay identical.  The
        view carries no edges: it is used for feature computation only.
        """
        if n <= 0 or n >= len(self.turns):
            return self
        view = ConversationMemory(conversation_id=self.conversation_id)
        view.turns = list(self.turns[-n:])
        return view

    # ---------------------------------------------------------- contamination
    def contamination(self, prompt: str, min_sim: float = 0.22) -> ContaminationResult:
        """MODEL_MEMORY_CONTAMINATION_SCORE (spec section 5).

        High when the current prompt semantically depends on a previous
        *model-generated, unverified* claim.
        """
        res = ContaminationResult()
        if not prompt or not self.unverified_model_claims:
            return res
        ptoks = set(content_tokens(prompt))
        best = 0.0
        for c in self.unverified_model_claims:
            # A model paraphrase of an earlier user-provided claim is still
            # not independently verified, but reusing it is not model-memory
            # contamination. Preserve the provenance distinction without
            # punishing a directly grounded follow-up.
            if self._supported_by_prior_user_claim(c):
                continue
            s = similarity(prompt, c.text)
            ctoks = set(content_tokens(c.text))
            entity_overlap = len(ptoks & ctoks) / max(1, len(ctoks))
            # dependence: prompt shares substantial content with the claim
            dependence = max(s, 0.75 * entity_overlap if entity_overlap >= 0.5 else s)
            if dependence < min_sim:
                continue
            unverified_boost = 1.0 if c.provenance_type == ProvenanceType.MODEL_GENERATED else 0.85
            partial = dependence * unverified_boost * (1.0 - c.trust)
            if partial > best:
                best = partial
            res.dependent_claims.append(c)
            res.reasons.append(f"prompt depends on {c.provenance_type.value} claim (turn {c.turn_id})")
            res.explanations.append(
                f"Turn {c.turn_id} [{c.provenance_type.value}/{c.verification_status.value}] "
                f"\"{c.text[:110]}\" has trust={c.trust:.2f}; current prompt reuses it."
            )
        # also count how many distinct unverified model claims are relied upon
        res.dependent_claims.sort(key=lambda c: -similarity(prompt, c.text))
        res.dependent_claims = res.dependent_claims[:5]
        score = best + 0.1 * max(0, len(res.dependent_claims) - 1)
        res.score = float(max(0.0, min(1.0, score)))
        if res.dependent_claims and res.score >= 0.5:
            res.reasons.append("MODEL_GENERATED + UNVERIFIED + PROMPT_DEPENDS")
        return res

    def _supported_by_prior_user_claim(self, model_claim: Claim) -> bool:
        model_tokens = set(content_tokens(model_claim.text))
        if len(model_tokens) < 2:
            return False
        model_numbers = _significant_numbers(model_claim.text)
        for source_turn in self.turns:
            if source_turn.speaker != "user" or source_turn.turn_id >= model_claim.turn_id:
                continue
            # Compare to the complete source turn rather than its sentence-level
            # extracted claims: sentence splitting can strip a heading/entity
            # that the assistant includes in an otherwise faithful paraphrase.
            source_tokens = set(content_tokens(source_turn.text))
            shared = model_tokens & source_tokens
            # Numeric facts must match exactly. This deliberately does not
            # turn user assertions into independently verified evidence.
            if model_numbers and not model_numbers.issubset(_significant_numbers(source_turn.text)):
                continue
            overlap = len(shared) / max(1, len(model_tokens))
            if overlap >= 0.55 and len(shared) >= 2:
                return True
        return False

    # ------------------------------------------------------------------ graph
    def _link_new_turn(self, turn: Turn) -> None:
        prior = [c for t in self.turns[:-1] for c in t.claims]
        for c in turn.claims:
            for p in prior:
                s = similarity(c.text, p.text)
                if s < 0.45:
                    continue
                self.edges.append(
                    MemoryEdge(src=c.claim_id, dst=p.claim_id, edge_type=EdgeType.REFERENCES, weight=s)
                )
            if turn.speaker == "assistant":
                self.edges.append(
                    MemoryEdge(
                        src=c.claim_id,
                        dst=f"turn:{turn.turn_id}",
                        edge_type=EdgeType.GENERATED_BY,
                        weight=1.0,
                    )
                )
        # derived-from: a later claim that reuses an earlier *numeric fact*.
        # The rule used to compare every token (stopwords included), so any two
        # sentences sharing two common words produced an edge.
        for c in turn.claims:
            nums = _significant_numbers(c.text)
            if not nums:
                continue
            for p in prior:
                if nums & _significant_numbers(p.text):
                    self.edges.append(
                        MemoryEdge(src=c.claim_id, dst=p.claim_id, edge_type=EdgeType.DERIVED_FROM, weight=1.0)
                    )

    def support_contradiction_edges(self, pairs: Iterable[Tuple[str, str, EdgeType, float]]) -> None:
        for src, dst, et, w in pairs:
            self.edges.append(MemoryEdge(src=src, dst=dst, edge_type=et, weight=w))

    def graph_payload(self) -> Dict[str, Any]:
        """Node/edge payload for the conversation graph visualization."""
        nodes = []
        for t in self.turns:
            nodes.append(
                {
                    "id": f"turn:{t.turn_id}",
                    "type": "turn",
                    "label": f"T{t.turn_id} ({t.speaker})",
                    "state": "neutral",
                    "turn": t.turn_id,
                }
            )
            for e in _naive_entities(t.text)[:4]:
                nodes.append(
                    {"id": f"ent:{e}", "type": "entity", "label": e, "state": "entity", "turn": t.turn_id}
                )
        for c in self.all_claims:
            if c.provenance_type == ProvenanceType.SOURCE_VERIFIED:
                state = "verified"
            elif c.verification_status == VerificationStatus.CONTRADICTED:
                state = "contradicted"
            elif c.is_model_claim:
                state = "model-generated"
            elif c.is_unverified:
                state = "unverified"
            else:
                state = "supported"
            nodes.append(
                {
                    "id": c.claim_id,
                    "type": "claim",
                    "label": c.text[:90],
                    "state": state,
                    "turn": c.turn_id,
                    "provenance": c.provenance_type.value,
                    "verification": c.verification_status.value,
                    "trust": round(c.trust, 3),
                }
            )
            if c.source:
                nodes.append({"id": f"src:{c.source}", "type": "source", "label": c.source, "state": "verified"})
        # dedupe node ids
        seen, uniq = set(), []
        for n in nodes:
            if n["id"] not in seen:
                seen.add(n["id"])
                uniq.append(n)
        return {
            "conversation_id": self.conversation_id,
            "nodes": uniq,
            "edges": [e.to_dict() for e in self.edges],
            "summary": self.summary(),
        }

    def summary(self) -> Dict[str, Any]:
        from ..provenance.classifier import provenance_summary

        s = provenance_summary(self.all_claims)
        s.update(
            {
                "turns": len(self.turns),
                "conversation_id": self.conversation_id,
                "model_claims": len(self.model_claims),
                "unverified_model_claims": len(self.unverified_model_claims),
            }
        )
        return s

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "turns": [t.to_dict() for t in self.turns],
            "edges": [e.to_dict() for e in self.edges],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ConversationMemory":
        mem = ConversationMemory(
            conversation_id=d.get("conversation_id", "conv"),
            turns=[Turn.from_dict(t) for t in d.get("turns", [])],
        )
        mem.created_at = float(d.get("created_at", mem.created_at))
        mem.edges = [
            MemoryEdge(
                src=e["src"],
                dst=e["dst"],
                edge_type=EdgeType(e["edge_type"]),
                weight=float(e.get("weight", 1.0)),
                metadata=e.get("metadata", {}) or {},
            )
            for e in d.get("edges", [])
        ]
        return mem
