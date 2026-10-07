"""Deterministic, dependency-light text utilities.

PHC-X deliberately avoids heavyweight NLP downloads for its core feature
engine so that experiments are reproducible offline.  All similarity
functions here are deterministic and pure-python/numpy.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List, Sequence

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_STOP = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in",
    "on", "at", "by", "for", "with", "about", "as", "is", "are", "was", "were",
    "be", "been", "being", "it", "its", "this", "that", "these", "those", "i",
    "you", "he", "she", "we", "they", "them", "his", "her", "our", "your", "my",
    "me", "him", "do", "does", "did", "not", "no", "yes", "from", "into", "than",
    "so", "such", "can", "could", "will", "would", "should", "shall", "may",
    "might", "must", "have", "has", "had", "what", "which", "who", "whom", "how",
    "when", "where", "why", "isnt", "wasnt", "there", "here", "also", "very",
}

_NEGATIONS = {
    "not", "no", "never", "none", "nothing", "nobody", "neither", "nor",
    "without", "cannot", "cant", "couldnt", "wouldnt", "shouldnt", "wont",
    "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "hadnt", "doesnt",
    "dont", "didnt", "aint", "hardly", "scarcely", "barely", "unless",
}


def tokenize(text: str, lowercase: bool = True, drop_stop: bool = False) -> List[str]:
    text = text or ""
    if lowercase:
        text = text.lower()
    toks = _WORD_RE.findall(text)
    if drop_stop:
        toks = [t for t in toks if t not in _STOP]
    return toks


def content_tokens(text: str) -> List[str]:
    return tokenize(text, drop_stop=True)


def sent_split(text: str) -> List[str]:
    """Rule-based sentence splitter (deterministic, no external models)."""
    text = (text or "").strip()
    if not text:
        return []
    # protect common abbreviations
    text = re.sub(r"\b(e\.g\.|i\.e\.|etc\.|vs\.|cf\.|Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)", lambda m: m.group(0).replace(".", "\u0001"), text)
    parts = re.split(r"(?<=[.!?:;])\s+|\n+", text)
    out = []
    for p in parts:
        p = p.replace("\u0001", ".").strip()
        if p:
            out.append(p)
    return out


def _hashed_vec(tokens: Sequence[str], dim: int = 512) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float64)
    for t in tokens:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        v[h % dim] += 1.0
        # bigram channel
    for a, b in zip(tokens, tokens[1:]):
        h = int(hashlib.md5((a + "\x00" + b).encode("utf-8")).hexdigest(), 16)
        v[h % dim] += 0.5
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def ngram_overlap(a: Sequence[str], b: Sequence[str], n: int = 2) -> float:
    ga = {tuple(a[i:i + n]) for i in range(max(0, len(a) - n + 1))}
    gb = {tuple(b[i:i + n]) for i in range(max(0, len(b) - n + 1))}
    if not ga and not gb:
        return 0.0
    return len(ga & gb) / max(1, len(ga | gb))


def similarity(a: str, b: str) -> float:
    """Deterministic hybrid similarity in [0, 1]."""
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return 0.0
    cos = cosine(_hashed_vec(ta), _hashed_vec(tb))
    jac = jaccard(ta, tb)
    big = ngram_overlap(ta, tb)
    return float(max(0.0, min(1.0, 0.55 * cos + 0.25 * jac + 0.20 * big)))


def contains_any(text: str, words: Sequence[str]) -> bool:
    low = (text or "").lower()
    return any(w.lower() in low for w in words)


def negation_present(text: str) -> bool:
    return any(t in _NEGATIONS for t in tokenize(text))


def ratio(a: float, b: float) -> float:
    return 0.0 if b == 0 else float(a) / float(b)


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def entropy(probs: Sequence[float]) -> float:
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p + 1e-12, 2)
    return h
