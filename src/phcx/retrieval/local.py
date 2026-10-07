"""Local, deterministic retrieval (no network required).

An optional dense backend can be plugged in later; the default scorer is a
hashed-n-gram TF similarity so experiments are fully reproducible offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..utils.text import content_tokens, similarity, sent_split


@dataclass
class Document:
    doc_id: str
    text: str
    title: str = ""
    source: str = ""
    timestamp: Optional[str] = None
    reliability: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedPassage:
    doc_id: str
    text: str
    score: float
    title: str = ""
    source: str = ""
    timestamp: Optional[str] = None
    reliability: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "score": round(self.score, 4),
            "title": self.title,
            "source": self.source,
            "timestamp": self.timestamp,
            "reliability": self.reliability,
        }


class LocalRetriever:
    def __init__(self, documents: Optional[Sequence[Document]] = None):
        self.documents: List[Document] = list(documents or [])

    def add(self, doc: Document) -> None:
        self.documents.append(doc)

    def load_jsonl(self, path: str) -> int:
        from ..utils.io import read_jsonl

        n = 0
        for row in read_jsonl(path):
            self.documents.append(
                Document(
                    doc_id=str(row.get("doc_id", row.get("id", f"doc{n}"))),
                    text=str(row.get("text", row.get("content", ""))),
                    title=str(row.get("title", "")),
                    source=str(row.get("source", "")),
                    timestamp=row.get("timestamp"),
                    reliability=float(row.get("reliability", 0.7)),
                )
            )
            n += 1
        return n

    def _passages(self) -> List[RetrievedPassage]:
        out: List[RetrievedPassage] = []
        for d in self.documents:
            for s in sent_split(d.text):
                if len(content_tokens(s)) < 3:
                    continue
                out.append(
                    RetrievedPassage(
                        doc_id=d.doc_id, text=s, score=0.0, title=d.title,
                        source=d.source, timestamp=d.timestamp, reliability=d.reliability,
                    )
                )
        return out

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.08) -> List[RetrievedPassage]:
        if not self.documents:
            return []
        passages = self._passages()
        for p in passages:
            p.score = similarity(query, p.text)
        passages = [p for p in passages if p.score >= min_score]
        passages.sort(key=lambda p: -p.score)
        return passages[:top_k]
