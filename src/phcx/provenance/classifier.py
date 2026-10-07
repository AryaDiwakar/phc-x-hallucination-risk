"""Provenance classification.

Central rule of PHC-X: a previous LLM statement is MODEL_GENERATED and
UNVERIFIED unless an independent source/verification is attached.  It must
never be silently promoted to ground truth.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

from ..memory.types import Claim, ProvenanceType, Speaker, Turn, VerificationStatus

_VERIFIED_SOURCES = ("wikipedia", "fever", "snli", "doc:", "http", "doi:", "retrieved:")


def classify_provenance(
    turn: Turn,
    claim: Claim,
    source: Optional[str] = None,
    retrieved: bool = False,
    verified: bool = False,
) -> Claim:
    """Assign provenance + verification status to a claim (in place)."""
    src = source or claim.source
    speaker = (turn.speaker or claim.speaker or "").lower()

    if verified:
        ptype = ProvenanceType.SOURCE_VERIFIED
        status = VerificationStatus.VERIFIED
    elif retrieved or speaker in ("tool", "retriever", "search"):
        ptype = ProvenanceType.RETRIEVED
        status = VerificationStatus.SUPPORTED if src else VerificationStatus.UNVERIFIED
    elif speaker in ("assistant", "model", "ai", "bot"):
        # Model claims are split into (a) claims attributed to an external
        # source and (b) free-standing model assertions/inferences.
        if src and any(tok in src.lower() for tok in _VERIFIED_SOURCES):
            ptype = ProvenanceType.SOURCE_VERIFIED if verified else ProvenanceType.MODEL_GENERATED
            status = VerificationStatus.SUPPORTED if verified else VerificationStatus.UNVERIFIED
        elif _looks_inferential(claim.text):
            ptype = ProvenanceType.MODEL_INFERRED
            status = VerificationStatus.UNVERIFIED
        else:
            ptype = ProvenanceType.MODEL_GENERATED
            status = VerificationStatus.UNVERIFIED
    elif speaker in ("user", "human"):
        ptype = ProvenanceType.USER_ASSERTED
        status = VerificationStatus.UNVERIFIED
    elif speaker in ("system", ""):
        ptype = ProvenanceType.UNKNOWN
        status = VerificationStatus.UNKNOWN
    else:
        ptype = ProvenanceType.UNKNOWN
        status = VerificationStatus.UNKNOWN

    if src:
        # keep the attribution visible for every provenance type: SOURCE_VERIFIED
        # claims must still show WHICH source verified them.
        claim.source = src
    claim.provenance_type = ptype
    claim.verification_status = status
    return claim


def _looks_inferential(text: str) -> bool:
    low = text.lower()
    markers = (
        "therefore", "probably", "likely", "i think", "it seems", "appears to",
        "must be", "might be", "could be", "based on this", "so we can conclude",
        "this suggests", "in other words",
    )
    return any(m in low for m in markers)


def annotate_turn(
    turn: Turn,
    source: Optional[str] = None,
    retrieved: bool = False,
    verified: bool = False,
) -> List[Claim]:
    for c in turn.claims:
        classify_provenance(turn, c, source=source, retrieved=retrieved, verified=verified)
    return turn.claims


def provenance_summary(claims: Iterable[Claim]) -> dict:
    claims_l: List[Claim] = list(claims)
    counts = {p.value: 0 for p in ProvenanceType}
    vcounts = {v.value: 0 for v in VerificationStatus}
    for c in claims_l:
        counts[c.provenance_type.value] += 1
        vcounts[c.verification_status.value] += 1
    n = len(claims_l)
    unverified_model = sum(1 for c in claims_l if c.is_model_claim and c.is_unverified)
    return {
        "total_claims": n,
        "by_provenance": counts,
        "by_verification": vcounts,
        "unverified_model_claims": unverified_model,
        "mean_trust": (sum(c.trust for c in claims_l) / n) if n else 0.0,
    }
