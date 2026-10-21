"""Fusion rules operate on 1-based ranked lists and never use raw-score sums."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Sequence

RankedList = Sequence[dict[str, Any]]


def union_statistics(ranked_lists: Sequence[RankedList]) -> dict[str, dict[str, Any]]:
    ranks: dict[str, list[int]] = defaultdict(list)
    for ranked in ranked_lists:
        seen = set()
        for expected_rank, row in enumerate(ranked, start=1):
            doc_id = row["doc_id"]
            if doc_id in seen:
                raise ValueError(f"Duplicate doc_id in a query view: {doc_id}")
            if row["rank"] != expected_rank:
                raise ValueError("Ranked lists must use contiguous 1-based ranks")
            seen.add(doc_id)
            ranks[doc_id].append(expected_rank)
    return {
        doc_id: {
            "num_supporting_views": len(values),
            "best_rank": min(values),
            "mean_rank": sum(values) / len(values),
        }
        for doc_id, values in sorted(ranks.items())
    }


def reciprocal_rank_fusion(
    ranked_lists: Sequence[RankedList], *, k_rrf: float = 60, top_k: int = 5
) -> list[dict[str, Any]]:
    """RRF(d) = sum(1 / (k_rrf + rank)); ranks begin at one."""
    if k_rrf < 0 or top_k < 1:
        raise ValueError("k_rrf must be nonnegative and top_k must be positive")
    statistics = union_statistics(ranked_lists)
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for row in ranked:
            scores[row["doc_id"]] += 1.0 / (k_rrf + row["rank"])
    ordering = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:top_k]
    return [
        {"doc_id": doc_id, "rank": rank, "rrf_score": scores[doc_id], **statistics[doc_id]}
        for rank, doc_id in enumerate(ordering, start=1)
    ]


def make_fusion(config: dict[str, Any]) -> Callable[..., list[dict[str, Any]]]:
    """One extension seam for future CombSUM/weighted RRF rules."""
    rule = config.get("rule", "rrf")
    if rule != "rrf":
        raise ValueError(f"Fusion rule {rule!r} is not implemented in this core")

    def fuse(ranked_lists: Sequence[RankedList], *, top_k: int) -> list[dict[str, Any]]:
        return reciprocal_rank_fusion(ranked_lists, k_rrf=float(config["k_rrf"]), top_k=top_k)

    return fuse
