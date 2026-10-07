"""Create a blinded human-validation packet without embedding gold labels."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from phcx.experiments.external import load_diahalu
from phcx.experiments.matrix import load_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/annotations/human_validation_sample.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=240)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    phc = load_jsonl("data/phc_conversation/test.jsonl")
    dia = load_diahalu("data/raw/diahalu_bench.jsonl")
    pool = [("PHC-Conversation", r) for r in phc] + [("DiaHalu", r) for r in dia]
    rng.shuffle(pool)
    selected = pool[:args.n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for i, (source, row) in enumerate(selected):
            fh.write(json.dumps({
                "annotation_id": f"hv_{i:04d}",
                "source_dataset": source,
                "prompt": row.get("prompt", ""),
                "conversation_history": row.get("conversation_history", []),
                "evidence": row.get("evidence", []),
                "proposed_answer": row.get("answer") or row.get("generated_answer", ""),
                "annotator_label": None,
                "rationale": None,
            }) + "\n")
    print(f"wrote {len(selected)} blinded items to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
