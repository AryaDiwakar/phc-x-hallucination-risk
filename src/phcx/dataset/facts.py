"""Deterministic fact-bank used to build PHC-Conversation families.

Everything is generated offline from a seed: no external downloads, no private
conversations, no copyrighted content.  Names are synthetic but realistic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import random

_TOPICS = [
    "Orion", "Nimbus", "Vega", "Kestrel", "Aurora", "Basalt", "Cobalt", "Delta",
    "Ember", "Fjord", "Granite", "Harbor", "Indigo", "Juniper", "Krypton",
]
_DOMAINS = [
    "medical imaging", "credit-risk modelling", "wildlife monitoring", "route planning",
    "document classification", "speech recognition", "energy forecasting", "fraud detection",
    "network intrusion detection", "quality inspection", "customer churn prediction",
    "air-quality estimation", "satellite crop mapping", "clinical triage", "chat intent detection",
]
_DATASET_HEADS = ["Corpus", "Atlas", "Bench", "Set", "CorpusX", "Archive", "Pool"]
_MODEL_NAMES = [
    "BERT", "RoBERTa", "DistilBERT", "XLNet", "ELECTRA", "DeBERTa", "ALBERT",
    "Longformer", "BigBird", "ConvBERT",
]
_METRICS = ["accuracy", "F1", "precision", "recall", "AUROC", "BLEU"]
_TOOLS = ["PyTorch", "TensorFlow", "JAX", "scikit-learn", "XGBoost", "LightGBM", "ONNX"]
_FOUNDED = list(range(2015, 2027))
_TEAMS = ["the retrieval group", "the vision team", "the core ML team", "the platform group"]


@dataclass
class FactFamily:
    family_id: str
    topic: str
    domain: str
    dataset: str
    alt_dataset: str
    model: str
    alt_model: str
    tool: str
    metric: str
    value_a: float
    value_b: float
    value_c: float
    year: int
    alt_year: int
    team: str
    facts: List[str] = field(default_factory=list)
    false_facts: List[str] = field(default_factory=list)

    @property
    def true_facts(self) -> List[str]:
        return list(self.facts)


def build_family(rng: random.Random, idx: int) -> FactFamily:
    # unique synthetic project name: guarantees no two families share a topic
    topic = f"{_TOPICS[idx % len(_TOPICS)]}-{idx // len(_TOPICS)}{rng.choice(['A', 'B', 'X', 'P', 'Q'])}"
    domain = rng.choice(_DOMAINS)
    dataset = f"{rng.choice(_DATASET_HEADS)} {rng.choice(['A', 'B', 'C', 'N', 'R'])}"
    alt_dataset = f"{rng.choice(_DATASET_HEADS)} {rng.choice(['D', 'E', 'M', 'S'])}"
    # alternatives sometimes collide on purpose (alias ambiguity)
    if rng.random() < 0.25:
        alt_dataset = dataset
    model = rng.choice(_MODEL_NAMES)
    alt_model = rng.choice([m for m in _MODEL_NAMES if m != model])
    tool = rng.choice(_TOOLS)
    metric = rng.choice(_METRICS)
    base = round(rng.uniform(60, 92), 1)
    delta = round(rng.uniform(2.0, 9.5), 1)
    after = round(base + delta, 1)
    other = round(rng.uniform(55, 89), 1)
    year = rng.choice(_FOUNDED)
    alt_year = year + rng.randint(1, 3)
    team = rng.choice(_TEAMS)

    fam = FactFamily(
        family_id=f"fam_{idx:04d}",
        topic=topic, domain=domain, dataset=dataset, alt_dataset=alt_dataset,
        model=model, alt_model=alt_model, tool=tool, metric=metric,
        value_a=base, value_b=after, value_c=other, year=year, alt_year=alt_year, team=team,
    )
    fam.facts = [
        f"The {topic} project works on {domain}.",
        f"The project uses {dataset} for evaluation.",
        f"The baseline model is {model}.",
        f"The training pipeline is built with {tool}.",
        f"The {metric} on {dataset} is {base}%.",
        f"{team} maintains the {topic} repository.",
        f"The project started in {year}.",
    ]
    fam.false_facts = [
        f"The {dataset} contains 50,000 samples.",
        f"The {metric} of {model} is 99.1%.",
        f"The project uses {alt_dataset} for evaluation.",
        f"The baseline model is {alt_model}.",
        f"The project was started in {alt_year}.",
        f"{dataset} was retired in 2019.",
    ]
    return fam


def build_families(n: int, seed: int = 42) -> List[FactFamily]:
    rng = random.Random(seed)
    return [build_family(rng, i) for i in range(n)]
