"""Run a *synthetic* annotation round over the human-validation packet.

This is a rehearsal, not a study.  The "annotators" are a simulator with
per-annotator noise; the output is used to prove that the analysis pipeline
works and to measure how much label noise the design tolerates before a
single real person has looked at a single item.

The artifacts it writes are marked ``synthetic: true`` and carry a warning
at the top of every file.  They must never be reported as human validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from phcx.experiments.human_validation import (
    _SIMULATED_WARNING,
    analyze_annotations,
    simulate_annotations,
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", default="data/annotations/human_validation_sample.jsonl")
    parser.add_argument("--key", default="data/annotations/human_validation_key.jsonl")
    parser.add_argument("--out", default="results/raw/human_validation_simulated.jsonl")
    parser.add_argument("--report", default="results/raw/human_validation_simulated_report.json")
    parser.add_argument("--annotators", type=int, default=100)
    parser.add_argument("--n-annotations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    packet = _read_jsonl(Path(args.packet))
    key = _read_jsonl(Path(args.key))
    # the key file starts with a header line
    if key and "annotation_id" not in key[0]:
        key = key[1:]

    annotations = simulate_annotations(
        packet, key,
        n_annotators=args.annotators,
        n_annotations=args.n_annotations,
        seed=args.seed,
    )
    if not annotations:
        print("no annotations produced -- check the packet and key paths")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "warning": _SIMULATED_WARNING,
            "n_annotations": len(annotations),
            "n_annotators": args.annotators,
            "seed": args.seed,
            "synthetic": True,
        }) + "\n")
        for a in annotations:
            fh.write(json.dumps(a) + "\n")

    report = analyze_annotations(annotations, key, seed=args.seed)
    report["warning"] = _SIMULATED_WARNING
    report["synthetic"] = True
    report["parameters"] = {
        "n_annotators": args.annotators,
        "n_annotations": args.n_annotations,
        "seed": args.seed,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print(f"[simulate] {len(annotations)} synthetic annotations by "
          f"{args.annotators} simulated annotators -> {out}")
    print(f"[simulate] analysis report -> {report_path}")
    print("          NOTE: these are simulated, not human, annotations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
