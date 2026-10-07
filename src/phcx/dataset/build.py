"""One-call builder: generate PHC-Conversation, split by family, audit leakage."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from ..utils.io import save_json
from .generator import PHCConversationGenerator
from .splits import family_split, leakage_report, write_splits


def build_dataset(
    out_dir: str | Path = "data/phc_conversation",
    samples_per_category: int = 100,
    n_families: int = 240,
    seed: int = 42,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    long_context_extra_turns: Sequence[int] = (30, 45, 60),
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = PHCConversationGenerator(
        samples_per_category=samples_per_category,
        seed=seed,
        n_families=n_families,
        long_context_extra_turns=tuple(long_context_extra_turns),
    )
    rows = gen.generate()
    splits = family_split(rows, ratios=ratios, seed=seed)
    report = leakage_report(rows, splits)
    paths = write_splits(splits, out_dir, report=report)

    summary = {
        "n_rows": len(rows),
        "n_families": len({r["family_id"] for r in rows}),
        "seed": seed,
        "ratios": list(ratios),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "label_rate": {
            k: round(sum(int(r["label"]) for r in v) / max(1, len(v)), 4)
            for k, v in splits.items()
        },
        "label_source": "union(label_gold, label_verified)",
        "gold_verified_agreement": round(
            sum(1 for r in rows if r["label_gold"] == r["label_verified"]) / max(1, len(rows)), 4
        ),
        "leakage_passed": report["passed"],
        "leakage_checks": report["checks"],
        "paths": paths,
    }
    save_json(out_dir / "build_summary.json", summary)
    return summary
