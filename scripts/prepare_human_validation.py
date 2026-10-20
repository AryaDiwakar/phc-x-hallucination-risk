"""Create the stratified, blinded human-validation packet.

Implements the sampling plan in ``docs/human_validation_protocol.md``:

* 240 unique items plus 40 duplicates for intra-rater consistency
* stratified across the PHC-Conversation risk categories and the DiaHalu
  domains, and across the PHC-X action (GENERATE / RETRIEVE / CLARIFY-or-
  ABSTAIN) and the risk score band
* the annotator sees no label, no score and no stratum

Two files are written:

``human_validation_sample.jsonl``
    the blind packet handed to the annotator
``human_validation_key.jsonl``
    the unblinded key (model score, action, band, stratum, gold label) used
    only for the analysis step -- never shown to the annotator
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from phcx.experiments.external import load_diahalu
from phcx.experiments.matrix import load_jsonl
from phcx.intervention import BLOCKING_ACTIONS

RUBRIC_VERSION = "hv-rubric-v1"
RUBRIC = [
    "Is the answer factually supported by the supplied context? (yes/no/uncertain)",
    "Does it contradict the conversation history? (yes/no/uncertain)",
    "Does it contain an unsupported presupposition or entity/time drift? (yes/no/uncertain)",
    "Would you allow generation without retrieval or clarification? (allow/retrieve/clarify/abstain)",
    "Confidence (1-5) and a short rationale.",
]

#: the three action groups the protocol stratifies over
ACTION_GROUPS = ("GENERATE", "RETRIEVE", "CLARIFY_OR_ABSTAIN")
BANDS = ("LOW", "MODERATE", "HIGH", "CRITICAL")


def _action_group(action: str) -> str:
    if action in BLOCKING_ACTIONS:
        return "CLARIFY_OR_ABSTAIN"
    if action == "RETRIEVE":
        return "RETRIEVE"
    return "GENERATE"


def _band(level: str) -> str:
    return level if level in BANDS else "UNKNOWN"


def _score_rows(rows: List[Dict[str, Any]], checkpoint: Optional[str],
                model_name: str, groups: Optional[List[str]]) -> List[Dict[str, Any]]:
    """Attach the model's risk score, level and action to every candidate."""
    from phcx.pipeline import PHCPipeline
    from phcx.risk import RiskModel

    model = RiskModel.load(checkpoint) if checkpoint else None
    pipe = PHCPipeline(model=model, groups=groups)
    out: List[Dict[str, Any]] = []
    for row in rows:
        pred = pipe.analyze(
            row.get("prompt", ""),
            conversation_history=row.get("conversation_history", []) or [],
            evidence=row.get("evidence", []) or [],
            use_retriever=False,
        )
        enriched = dict(row)
        enriched["_risk"] = round(float(pred.risk_score), 6)
        enriched["_level"] = pred.risk_level
        enriched["_action"] = pred.recommended_action
        enriched["_action_group"] = _action_group(pred.recommended_action)
        enriched["_band"] = _band(pred.risk_level)
        out.append(enriched)
    return out


def _stratum_key(row: Dict[str, Any]) -> str:
    """The category/domain stratum of a candidate.

    For DiaHalu the annotated error type is included: the dialogues are
    mostly clean turns, so stratifying on the domain alone leaves the
    annotated error turns -- the only ones a human can say something useful
    about -- as a rounding error in the sample.
    """
    base = str(row.get("category") or row.get("domain") or "unknown")
    meta = row.get("metadata") or {}
    errors = row.get("error_types") or meta.get("error_types") or []
    if errors:
        base = f"{base}:{','.join(sorted(errors))}"
    return base


def _stratify(candidates: List[Dict[str, Any]], n: int,
              rng: random.Random) -> List[Dict[str, Any]]:
    """Round-robin over the (category/domain, action, band) cells.

    Taking one item per cell in turn keeps every populated stratum
    represented, which a plain shuffle does not: a uniform draw would leave
    whole action/band combinations empty whenever they are rare in the pool.
    """
    cells: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for c in candidates:
        key = (_stratum_key(c), c["_action_group"], c["_band"])
        cells.setdefault(key, []).append(c)
    for bucket in cells.values():
        rng.shuffle(bucket)

    order = sorted(cells)
    rng.shuffle(order)
    picked: List[Dict[str, Any]] = []
    round_idx = 0
    while len(picked) < n:
        progressed = False
        for key in order:
            bucket = cells[key]
            if round_idx < len(bucket):
                picked.append(bucket[round_idx])
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
        round_idx += 1
    return picked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/annotations/human_validation_sample.jsonl")
    parser.add_argument("--key-out", default="data/annotations/human_validation_key.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--duplicates", type=int, default=40)
    parser.add_argument("--checkpoint", default="checkpoints/phcx_train_baseline_seed42/model")
    parser.add_argument("--phc", default="data/phc_conversation/test.jsonl")
    parser.add_argument("--diahalu", default="data/raw/diahalu_bench.jsonl")
    parser.add_argument("--pool-limit", type=int, default=None,
                        help="cap the number of candidate rows taken from each source "
                             "(the pool is scored with the model, which is the slow part)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    phc = load_jsonl(args.phc)
    dia = load_diahalu(args.diahalu)
    if args.pool_limit:
        phc = phc[: args.pool_limit]
        dia = dia[: args.pool_limit]

    pool: List[Dict[str, Any]] = []
    for r in phc:
        pool.append({**r, "source_dataset": "PHC-Conversation",
                     "stratum_dimension": "risk_category"})
    for r in dia:
        pool.append({**r, "source_dataset": "DiaHalu",
                     "stratum_dimension": "domain"})

    checkpoint = args.checkpoint if Path(str(args.checkpoint) + ".json").exists() else None
    scored = _score_rows(pool, checkpoint, "rf", None)

    selected = _stratify(scored, args.n, rng)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    key_path = Path(args.key_out)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    keys: List[Dict[str, Any]] = []
    for i, row in enumerate(selected):
        aid = f"hv_{i:04d}"
        records.append({
            "annotation_id": aid,
            "rubric_version": RUBRIC_VERSION,
            "source_dataset": row["source_dataset"],
            "prompt": row.get("prompt", ""),
            "conversation_history": row.get("conversation_history", []),
            "evidence": row.get("evidence", []),
            "proposed_answer": row.get("answer") or row.get("generated_answer", ""),
            "rubric": RUBRIC,
            "annotator_label": None,
            "rationale": None,
        })
        meta = row.get("metadata") or {}
        keys.append({
            "annotation_id": aid,
            "source_dataset": row["source_dataset"],
            "stratum_dimension": row["stratum_dimension"],
            "category_or_domain": row.get("category") or row.get("domain"),
            "stratum": _stratum_key(row),
            "error_types": row.get("error_types") or meta.get("error_types") or [],
            "risk_score": row["_risk"],
            "risk_level": row["_level"],
            "action": row["_action"],
            "action_group": row["_action_group"],
            "score_band": row["_band"],
            "gold_label": row.get("label"),
            "duplicate_of": None,
        })

    # intra-rater duplicates: the same item is presented again, under a new
    # id, with no marker that it is a repeat
    dups = min(args.duplicates, len(records))
    for j in range(dups):
        src_idx = (j * max(1, len(records) // max(dups, 1))) % len(records)
        src = records[src_idx]
        aid = f"hv_dup_{j:04d}"
        records.append({**src, "annotation_id": aid})
        keys.append({**keys[src_idx], "annotation_id": aid,
                     "duplicate_of": src["annotation_id"]})

    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    with key_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "rubric_version": RUBRIC_VERSION,
            "analysis_seed": args.seed,
            "checkpoint": checkpoint,
            "n_unique": len(selected),
            "n_duplicates": dups,
            "note": "UNBLINDED KEY -- for the analysis step only, never for the annotator.",
        }) + "\n")
        for k in keys:
            fh.write(json.dumps(k) + "\n")

    print(f"wrote {len(records)} blinded items "
          f"({len(selected)} unique + {dups} duplicates) to {out}")
    print(f"wrote the unblinded key to {key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
