"""Feature-matrix construction from conversation-level dataset rows."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..features import FEATURE_GROUPS, ABLATION_GROUPS, FeatureEngine
from ..pipeline import PHCPipeline
from ..pipeline.risk_pipeline import heuristic_risk

NON_MODEL_GROUPS = [g for g in FEATURE_GROUPS if g != "model"]


@dataclass
class DatasetMatrix:
    """Aligned feature matrix + labels + row metadata."""

    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    sample_ids: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    family_ids: List[str] = field(default_factory=list)
    heuristic: Optional[np.ndarray] = None

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def label_rate(self) -> float:
        return float(self.y.mean()) if self.n_rows else 0.0

    def subset(self, names: Sequence[str]) -> "DatasetMatrix":
        idx = [self.feature_names.index(n) for n in names if n in self.feature_names]
        return DatasetMatrix(
            X=self.X[:, idx],
            y=self.y,
            feature_names=[self.feature_names[i] for i in idx],
            sample_ids=list(self.sample_ids),
            categories=list(self.categories),
            family_ids=list(self.family_ids),
            heuristic=self.heuristic,
        )

    def groups(self, group_names: Sequence[str]) -> "DatasetMatrix":
        names: List[str] = []
        for g in group_names:
            names.extend(FEATURE_GROUPS.get(g, []))
        return self.subset(names)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_splits(data_dir: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    data_dir = Path(data_dir)
    out = {}
    for split in ("train", "val", "test"):
        p = data_dir / f"{split}.jsonl"
        if p.exists():
            out[split] = load_jsonl(p)
    if not out:
        raise FileNotFoundError(f"no splits found under {data_dir}")
    return out


def featurize_rows(
    rows: Sequence[Dict[str, Any]],
    groups: Optional[Sequence[str]] = None,
    history_turns: int = 10,
    use_retriever: bool = False,
    show_progress: bool = False,
) -> DatasetMatrix:
    """Run the deterministic feature pipeline over dataset rows.

    ``use_retriever`` is off by default: dataset rows carry their own
    evidence passages, so retrieval would only add nondeterminism.
    """
    pipeline = PHCPipeline(groups=list(groups) if groups else None,
                           history_turns=history_turns)
    names = list(pipeline.features.feature_names)
    X = np.zeros((len(rows), len(names)), dtype=float)
    heuristic = np.zeros(len(rows), dtype=float)
    y = np.zeros(len(rows), dtype=int)
    sample_ids, categories, family_ids = [], [], []

    for i, row in enumerate(rows):
        pred = pipeline.analyze(
            prompt=row.get("prompt", ""),
            conversation_history=row.get("conversation_history") or [],
            evidence=row.get("evidence") or [],
            conversation_id=row.get("family_id", "conv"),
            use_retriever=use_retriever,
        )
        feats = pred.features
        X[i] = [feats.get(n, 0.0) for n in names]
        heuristic[i] = heuristic_risk(feats)
        # Unlabeled external stress rows use a placeholder target; callers
        # must inspect the row labels before computing supervised metrics.
        y[i] = int(row.get("label", 0) or 0)
        sample_ids.append(row.get("sample_id", f"row_{i}"))
        categories.append(row.get("category", "unknown"))
        family_ids.append(row.get("family_id", ""))
        if show_progress and (i + 1) % 100 == 0:
            print(f"  featurized {i + 1}/{len(rows)}", flush=True)

    return DatasetMatrix(
        X=X, y=y, feature_names=names, sample_ids=sample_ids,
        categories=categories, family_ids=family_ids, heuristic=heuristic,
    )


def featurize_splits(
    splits: Dict[str, List[Dict[str, Any]]],
    groups: Optional[Sequence[str]] = None,
    history_turns: int = 10,
    show_progress: bool = False,
) -> Dict[str, DatasetMatrix]:
    t0 = time.perf_counter()
    out = {
        name: featurize_rows(rows, groups=groups, history_turns=history_turns,
                             show_progress=show_progress)
        for name, rows in splits.items()
    }
    elapsed = time.perf_counter() - t0
    if show_progress:
        print(f"  featurized {sum(m.n_rows for m in out.values())} rows "
              f"in {elapsed:.1f}s", flush=True)
    return out
