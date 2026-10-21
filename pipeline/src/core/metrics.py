"""Document recall and query-view cosine diagnostics."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def gold_metrics(gold_doc_ids: Sequence[str], final_doc_ids: Sequence[str]) -> dict[str, float | bool]:
    gold = set(gold_doc_ids)
    if not gold:
        raise ValueError("Each target must contain at least one supporting document")
    hits = len(gold.intersection(final_doc_ids))
    return {"benign_recall_at_5": hits / len(gold), "hit_any_gold_at_5": bool(hits)}


def query_diversity(vectors: np.ndarray) -> dict[str, float | str]:
    if vectors.ndim != 2 or len(vectors) < 2:
        raise ValueError("Diversity needs at least two query views")
    cosine = np.clip(vectors @ vectors.T, -1.0, 1.0)
    pairwise = cosine[np.triu_indices(len(vectors), k=1)]
    mean = float(pairwise.mean())
    return {
        "pairwise_cosines": [float(value) for value in pairwise],
        "mean_pairwise_cosine": mean,
        "min_pairwise_cosine": float(pairwise.min()),
        "max_pairwise_cosine": float(pairwise.max()),
        "rewrite_diversity": 1.0 - mean,
        "diversity_scope": "original_plus_four_rewrites",
    }
