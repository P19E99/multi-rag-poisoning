"""Single-query versus multi-query RRF with identical final document budgets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .common import file_sha256, fingerprint, read_jsonl, seed_everything, write_jsonl
from .encoder import DenseEncoder, validate_embeddings
from .fusion import make_fusion, union_statistics
from .index import cached_vectors, ensure_corpus_index, read_metadata, save_numpy, validate_corpus
from .metrics import gold_metrics, query_diversity

LOGGER = logging.getLogger(__name__)


def validate_inputs(
    corpus: Sequence[dict[str, Any]], targets: Sequence[dict[str, Any]],
    expansions: Sequence[dict[str, Any]], num_rewrites: int = 4,
) -> dict[str, dict[str, Any]]:
    from .expansion import validate_expansions

    corpus_ids = set(validate_corpus(corpus))
    if not targets or num_rewrites != 4:
        raise ValueError("This pipeline requires nonempty targets and exactly four rewrites")
    for target in targets:
        sample_id = target["sample_id"]
        gold = target.get("supporting_doc_ids", [])
        if not gold or len(set(gold)) != len(gold) or not set(gold).issubset(corpus_ids):
            raise ValueError(f"Target {sample_id} has empty, duplicate, or absent supporting_doc_ids")
    # Both generated and imported caches obey the same validation contract.
    # This module imports no Qwen/model-loading dependency at this boundary.
    return validate_expansions(list(targets), list(expansions), num_rewrites)


def apply_reranker(
    question: str, candidates: list[dict[str, Any]],
    reranker: Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Identity by default; a future reranker must preserve the candidate set."""
    if reranker is None:
        return candidates
    reordered = reranker(question, list(candidates))
    if len(reordered) != len(candidates) or sorted(row["doc_id"] for row in reordered) != sorted(
        row["doc_id"] for row in candidates
    ):
        raise ValueError("A reranker must preserve the complete candidate document set")
    return reordered


def sample_results(
    target: dict[str, Any], expansion: dict[str, Any], ranked_lists: Sequence[list[dict[str, Any]]],
    query_vectors: np.ndarray, *, corpus_size: int, retrieval_config: dict[str, Any],
    fusion_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if int(retrieval_config["top_k_per_view"]) != 10 or int(retrieval_config["final_top_k"]) != 5:
        raise ValueError("This pipeline requires each view Top-10 and final Top-5")
    if len(ranked_lists) != 5 or len(expansion["rewrites"]) != 4 or len(query_vectors) != 5:
        raise ValueError("Expected original plus four rewrite query views")
    expected_view_count = min(10, corpus_size)
    final_count = min(5, corpus_size)
    if final_count < 1:
        raise ValueError("Cannot retrieve from an empty corpus")
    for ranked in ranked_lists:
        if len(ranked) != expected_view_count:
            raise AssertionError("Each retrieval list must have min(10, corpus_size) documents")
        if any(not np.isfinite(row["raw_cosine_score"]) for row in ranked):
            raise ValueError("Raw cosine scores must be finite")
    statistics = union_statistics(ranked_lists)
    queries = [target["question"], *expansion["rewrites"]]
    common = {
        "sample_id": target["sample_id"],
        "question": target["question"],
        "original_query": target["question"],
        "rewrites": list(expansion["rewrites"]),
        "expansion_generation_fingerprint": expansion.get("generation", {}).get("fingerprint"),
        "semantic_review_guidance_applied": bool(expansion.get("generation", {}).get("semantic_review_guidance")),
        "gold_doc_ids": list(target["supporting_doc_ids"]),
        "final_context_budget_documents": final_count,
        "requested_final_top_k": 5,
        "top_k_per_view": 10,
        **query_diversity(query_vectors),
    }
    per_view = [
        {"view_id": index, "query": query, "ranked_list": ranked}
        for index, (query, ranked) in enumerate(zip(queries, ranked_lists))
    ]
    single_documents = ranked_lists[0][:final_count]
    single_ids = [row["doc_id"] for row in single_documents]
    single = {
        **common,
        "condition": "single",
        "num_query_views": 1,
        "final_doc_ids": single_ids,
        "final_scores": [row["raw_cosine_score"] for row in single_documents],
        "final_score_type": "cosine",
        "union_size": len(ranked_lists[0]),
        "union_scope": "per_view_top_10_before_final_truncation",
        "per_view_ranked_lists": per_view[:1],
        **gold_metrics(target["supporting_doc_ids"], single_ids),
    }
    fused = make_fusion(fusion_config)(ranked_lists, top_k=final_count)
    fused = apply_reranker(target["question"], fused)
    fused_ids = [row["doc_id"] for row in fused]
    multi = {
        **common,
        "condition": "multi_rrf",
        "num_query_views": 5,
        "final_doc_ids": fused_ids,
        "final_scores": [row["rrf_score"] for row in fused],
        "final_score_type": "rrf",
        "k_rrf": float(fusion_config["k_rrf"]),
        "union_size": len(statistics),
        "union_scope": "per_view_top_10_before_final_truncation",
        "union_doc_statistics": statistics,
        "per_view_ranked_lists": per_view,
        "final_doc_num_supporting_views": [row["num_supporting_views"] for row in fused],
        **gold_metrics(target["supporting_doc_ids"], fused_ids),
    }
    for row in (single, multi):
        final_ids = row["final_doc_ids"]
        if len(final_ids) != final_count or len(final_ids) > 5 or len(set(final_ids)) != len(final_ids):
            raise AssertionError("Final context budget or document deduplication failed")
    if len(single_ids) != len(fused_ids):
        raise AssertionError("Single and multi RRF final context budgets differ")
    return [single, multi]


def query_embeddings(
    queries: Sequence[str], encoder: DenseEncoder, cache_dir: str | Path, *, force: bool = False, strict_cache: bool = False
) -> tuple[np.ndarray, str]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "query_embeddings.npy"
    metadata_path = cache / "query_embeddings.meta.json"
    key = fingerprint({"ordered_queries": queries, "encoder": encoder.identity, "cache_version": 1})
    metadata = read_metadata(metadata_path)
    if strict_cache and not force and path.exists():
        if metadata.get("fingerprint") != key or metadata.get("embeddings_sha256") != file_sha256(path):
            raise ValueError("Query embedding artifact/config mismatch; rebuild with src/run_retrieval.py --force and the intended --config")
    vectors = None if force else cached_vectors(path, metadata, key, len(queries))
    if vectors is None:
        vectors = encoder.encode_queries(queries)
        validate_embeddings(vectors, len(queries))
        save_numpy(path, vectors)
        write_jsonl(metadata_path, [{
            "fingerprint": key,
            "n_queries": len(queries),
            "dimension": vectors.shape[1],
            "encoder": encoder.identity,
            "embeddings_sha256": file_sha256(path),
        }])
    else:
        LOGGER.info("Reusing %d cached query embeddings", len(queries))
    return vectors, key


def run_retrieval(config: dict[str, Any], *, force: bool = False) -> list[dict[str, Any]]:
    from .contracts import guard_writable, require_stage, record_stage, strict
    guard_writable(config)
    if strict(config):
        for stage in ("corpus", "expansion", "index"):
            require_stage(config, stage)
    if strict(config) and Path(config["paths"]["results"]).exists() and not force:
        require_stage(config, "retrieval")
    seed_everything(config["seed"])
    paths = config["paths"]
    corpus = read_jsonl(paths["corpus"])
    targets = read_jsonl(paths["targets"])
    expansions = read_jsonl(paths["expansions"])
    if "dataset" in config:
        from .corpus import validate_corpus_targets
        validate_corpus_targets(corpus, targets, config["dataset"]["corpus_size"], config["dataset"]["n_targets"])
    expansion_map = validate_inputs(corpus, targets, expansions, config["expansion"]["num_rewrites"])
    encoder = DenseEncoder({**config["retrieval"], "seed": config["seed"]})
    # Corpus cache has its own --force on build_index.py. Re-running retrieval
    # never needlessly recomputes a valid corpus embedding cache.
    index = ensure_corpus_index(corpus, encoder, paths["cache_dir"])
    queries = [query for target in targets for query in [target["question"], *expansion_map[target["sample_id"]]["rewrites"]]]
    vectors, query_key = query_embeddings(queries, encoder, paths["cache_dir"], force=force, strict_cache=strict(config))
    result_key = fingerprint({
        "implementation": "single-multi-rrf-v1",
        "corpus_index": index.fingerprint,
        "queries": query_key,
        "targets": targets,
        "expansion_records": expansions,
        "retrieval": config["retrieval"],
        "fusion": config["fusion"],
        "seed": config["seed"],
    })
    results_path = Path(paths["results"])
    metadata_path = results_path.with_suffix(".meta.json")
    metadata = read_metadata(metadata_path)
    if not force and results_path.exists() and metadata.get("fingerprint") == result_key:
        if file_sha256(results_path) == metadata.get("results_sha256"):
            rows = read_jsonl(results_path)
            if len(rows) == 2 * len(targets):
                LOGGER.info("Reusing completed retrieval results %s (%d rows)", results_path, len(rows))
                record_stage(config, "retrieval")
                return rows
    ranked_lists = index.search(vectors, int(config["retrieval"]["top_k_per_view"]))
    rows = []
    for offset, target in enumerate(targets):
        start = offset * 5
        sample_rows = sample_results(
            target, expansion_map[target["sample_id"]], ranked_lists[start : start + 5], vectors[start : start + 5],
            corpus_size=len(corpus), retrieval_config=config["retrieval"], fusion_config=config["fusion"],
        )
        for row in sample_rows:
            row["seed"] = config["seed"]
            row["corpus_fingerprint"] = index.fingerprint
            row["query_embeddings_fingerprint"] = query_key
            row["run_fingerprint"] = result_key
        rows.extend(sample_rows)
        LOGGER.info("Retrieved sample %d/%d %s: single recall=%.3f, multi recall=%.3f, union=%d",
                    offset + 1, len(targets), target["sample_id"], sample_rows[0]["benign_recall_at_5"],
                    sample_rows[1]["benign_recall_at_5"], sample_rows[1]["union_size"])
    write_jsonl(results_path, rows)
    write_jsonl(metadata_path, [{
        "fingerprint": result_key,
        "n_samples": len(targets),
        "n_rows": len(rows),
        "corpus_path": paths["corpus"],
        "corpus_sha256": file_sha256(paths["corpus"]),
        "results_sha256": file_sha256(results_path),
    }])
    LOGGER.info("Wrote %d condition rows to %s", len(rows), results_path)
    record_stage(config, "retrieval")
    return rows
