"""Core conversation representation: turns, claims, provenance, memory graph."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class ProvenanceType(str, Enum):
    USER_ASSERTED = "USER_ASSERTED"
    SOURCE_VERIFIED = "SOURCE_VERIFIED"
    MODEL_GENERATED = "MODEL_GENERATED"
    MODEL_INFERRED = "MODEL_INFERRED"
    RETRIEVED = "RETRIEVED"
    UNKNOWN = "UNKNOWN"


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"
    UNKNOWN = "UNKNOWN"


class Speaker(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


# Relative unreliability weight used by the contamination scorer.
# A model-generated, unverified claim must NEVER be treated as ground truth.
PROVENANCE_TRUST: Dict[ProvenanceType, float] = {
    ProvenanceType.SOURCE_VERIFIED: 1.0,
    ProvenanceType.RETRIEVED: 0.75,
    ProvenanceType.USER_ASSERTED: 0.6,
    ProvenanceType.UNKNOWN: 0.3,
    ProvenanceType.MODEL_INFERRED: 0.15,
    ProvenanceType.MODEL_GENERATED: 0.1,
}

VERIFICATION_TRUST: Dict[VerificationStatus, float] = {
    VerificationStatus.VERIFIED: 1.0,
    VerificationStatus.SUPPORTED: 0.8,
    VerificationStatus.UNVERIFIED: 0.2,
    VerificationStatus.UNKNOWN: 0.2,
    VerificationStatus.CONTRADICTED: 0.0,
}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass
class Claim:
    claim_id: str
    text: str
    turn_id: int
    speaker: str
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    provenance_type: ProvenanceType = ProvenanceType.UNKNOWN
    confidence: float = 0.5
    verification_status: VerificationStatus = VerificationStatus.UNKNOWN
    entities: List[str] = field(default_factory=list)
    temporal_scope: Optional[str] = None
    embeddings: Optional[List[float]] = None
    supporting_claims: List[str] = field(default_factory=list)
    contradicting_claims: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(text: str, turn_id: int, speaker: str, **kw: Any) -> "Claim":
        return Claim(claim_id=_new_id("clm"), text=text.strip(), turn_id=turn_id, speaker=speaker, **kw)

    @property
    def is_model_claim(self) -> bool:
        return self.provenance_type in (ProvenanceType.MODEL_GENERATED, ProvenanceType.MODEL_INFERRED)

    @property
    def is_unverified(self) -> bool:
        return self.verification_status in (VerificationStatus.UNVERIFIED, VerificationStatus.UNKNOWN)

    @property
    def trust(self) -> float:
        """Combined trust in [0, 1]; used everywhere instead of blind reuse."""
        return PROVENANCE_TRUST.get(self.provenance_type, 0.3) * VERIFICATION_TRUST.get(
            self.verification_status, 0.2
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provenance_type"] = self.provenance_type.value
        d["verification_status"] = self.verification_status.value
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Claim":
        d = dict(d)
        d["provenance_type"] = ProvenanceType(d.get("provenance_type", "UNKNOWN"))
        d["verification_status"] = VerificationStatus(d.get("verification_status", "UNKNOWN"))
        d.setdefault("claim_id", _new_id("clm"))
        return Claim(**d)


@dataclass
class Turn:
    turn_id: int
    speaker: str
    text: str
    timestamp: float = field(default_factory=time.time)
    claims: List[Claim] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp,
            "claims": [c.to_dict() for c in self.claims],
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Turn":
        return Turn(
            turn_id=int(d["turn_id"]),
            speaker=str(d.get("speaker", "user")),
            text=str(d.get("text", "")),
            timestamp=float(d.get("timestamp", 0.0)),
            claims=[Claim.from_dict(c) for c in d.get("claims", [])],
            metadata=d.get("metadata", {}) or {},
        )


class EdgeType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REFERENCES = "references"
    DERIVED_FROM = "derived-from"
    GENERATED_BY = "generated-by"


@dataclass
class MemoryEdge:
    src: str
    dst: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }
