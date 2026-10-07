"""Data splits and data-leakage audit (spec sections 33-34).

Splitting is performed by *conversation family* (source question family) so
that transformed variants of the same source never cross splits.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..utils.io import save_json, write_jsonl
from ..utils.text import similarity

SPLIT_NAMES = ("train", "val", "test")


def family_split(
    rows: Sequence[Dict[str, Any]],
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    family_key: str = "family_id",
) -> Dict[str, List[Dict[str, Any]]]:
    """Deterministic split by family id (no family crosses splits)."""
    import random

    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    families: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        families[str(r.get(family_key, r.get("sample_id", "unknown")))].append(r)

    fam_ids = sorted(families.keys())
    rng = random.Random(seed)
    rng.shuffle(fam_ids)

    n = len(fam_ids)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    train_ids = set(fam_ids[:n_train])
    val_ids = set(fam_ids[n_train:n_train + n_val])
    test_ids = set(fam_ids[n_train + n_val:])

    out: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for fid, rs in families.items():
        if fid in train_ids:
            out["train"].extend(rs)
        elif fid in val_ids:
            out["val"].extend(rs)
        else:
            out["test"].extend(rs)
    for k in out:
        out[k].sort(key=lambda r: str(r.get("sample_id", "")))
    return out


def stratified_family_split(
    rows: Sequence[Dict[str, Any]],
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    family_key: str = "family_id",
    category_key: str = "category",
) -> Dict[str, List[Dict[str, Any]]]:
    """Family split that also keeps category proportions approximately stable."""
    import random

    by_cat: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cat[str(r.get(category_key, "unknown"))][str(r.get(family_key, "?"))].append(r)

    out: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for cat in sorted(by_cat):
        fam_ids = sorted(by_cat[cat])
        rng = random.Random(f"{seed}:{cat}")
        rng.shuffle(fam_ids)
        n = len(fam_ids)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        for i, fid in enumerate(fam_ids):
            split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            out[split].extend(by_cat[cat][fid])
    for k in out:
        out[k].sort(key=lambda r: str(r.get("sample_id", "")))
    return out


def _p(text: str) -> str:
    return hashlib.sha1((text or "").strip().lower().encode()).hexdigest()


def leakage_report(
    rows: Sequence[Dict[str, Any]],
    splits: Dict[str, List[Dict[str, Any]]],
    similarity_threshold: float = 0.90,
    semantic_sample_cap: int = 400,
) -> Dict[str, Any]:
    """Audit for train/test leakage.  Writes findings as a JSON-able dict."""
    report: Dict[str, Any] = {
        "duplicate_prompts": [],
        "semantic_duplicates": [],
        "family_overlap": [],
        "evidence_overlap": [],
        "variant_crossing": [],
        "passed": True,
        "checks": {},
    }

    def _split_of(row: Dict[str, Any]) -> Optional[str]:
        for name, rs in splits.items():
            if any(r is row or r.get("sample_id") == row.get("sample_id") for r in rs):
                return name
        return None

    index = {r.get("sample_id"): _split_of(r) for r in rows}

    # 1) exact duplicate prompts
    seen: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        seen[_p(str(r.get("prompt", "")))].append(str(r.get("sample_id")))
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    cross = []
    for ids in dup.values():
        splits_touched = {index.get(i) for i in ids}
        if len(splits_touched - {None}) > 1:
            cross.append(ids)
    report["duplicate_prompts"] = cross[:50]
    report["checks"]["duplicate_prompts"] = len(cross)

    # 2) semantic duplicates across splits
    all_rows = list(rows)[:semantic_sample_cap]
    sem_cross = []
    for i in range(len(all_rows)):
        for j in range(i + 1, len(all_rows)):
            a, b = all_rows[i], all_rows[j]
            sa, sb = index.get(a.get("sample_id")), index.get(b.get("sample_id"))
            if sa is None or sb is None or sa == sb:
                continue
            if similarity(str(a.get("prompt", "")), str(b.get("prompt", ""))) >= similarity_threshold:
                sem_cross.append([a.get("sample_id"), b.get("sample_id"), sa, sb])
    report["semantic_duplicates"] = sem_cross[:50]
    report["checks"]["semantic_duplicates"] = len(sem_cross)

    # 3) family overlap across splits
    fam_split: Dict[str, set] = defaultdict(set)
    for split, rs in splits.items():
        for r in rs:
            fam_split[str(r.get("family_id", "?"))].add(split)
    fam_cross = sorted([f for f, ss in fam_split.items() if len(ss) > 1])
    report["family_overlap"] = fam_cross
    report["checks"]["family_overlap"] = len(fam_cross)

    # 4) identical evidence documents across splits
    ev_split: Dict[str, set] = defaultdict(set)
    for split, rs in splits.items():
        for r in rs:
            for e in r.get("evidence", []) or []:
                ev_split[_p(str(e.get("text", "")))].add(split)
    ev_cross = sorted([k for k, ss in ev_split.items() if len(ss) > 1])
    report["evidence_overlap"] = ev_cross[:50]
    report["checks"]["evidence_overlap"] = len(ev_cross)

    # 5) transformed variants (same family) crossing splits -> equals (3)
    report["variant_crossing"] = fam_cross
    report["checks"]["variant_crossing"] = len(fam_cross)

    report["passed"] = all(v == 0 for k, v in report["checks"].items())
    report["n_rows"] = len(rows)
    report["split_sizes"] = {k: len(v) for k, v in splits.items()}
    return report


def write_splits(
    splits: Dict[str, List[Dict[str, Any]]],
    out_dir: str | Path,
    report: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    out_dir = Path(out_dir)
    paths = {}
    for name, rows in splits.items():
        p = out_dir / f"{name}.jsonl"
        write_jsonl(p, rows)
        paths[name] = str(p)
    save_json(out_dir / "dataset_stats.json", {
        "sizes": {k: len(v) for k, v in splits.items()},
        "label_rate": {
            k: round(sum(int(r.get("label", 0)) for r in v) / max(1, len(v)), 4)
            for k, v in splits.items()
        },
        "categories": {
            k: sorted({str(r.get("category")) for r in v}) for k, v in splits.items()
        },
    })
    if report is not None:
        save_json(out_dir / "leakage_report.json", report)
    return paths
