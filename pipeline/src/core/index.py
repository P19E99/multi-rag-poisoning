"""Cached normalized corpus embeddings and an exact FAISS inner-product index."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np

from .common import file_sha256, fingerprint, read_jsonl, seed_everything, write_jsonl
from .encoder import DenseEncoder, validate_embeddings

LOGGER = logging.getLogger(__name__)


def read_metadata(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_corpus(corpus: Sequence[dict[str, Any]]) -> list[str]:
    if not corpus:
        raise ValueError("Corpus cannot be empty")
    doc_ids = []
    for doc in corpus:
        if not isinstance(doc.get("doc_id"), str) or not doc["doc_id"]:
            raise ValueError("Each corpus document needs a nonempty string doc_id")
        if not isinstance(doc.get("title"), str) or not isinstance(doc.get("text"), str):
            raise ValueError("Each corpus document needs string title and text fields")
        doc_ids.append(doc["doc_id"])
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("Corpus doc_ids must be deduplicated")
    return doc_ids


def cached_vectors(path: Path, metadata: dict[str, Any], key: str, n_rows: int) -> np.ndarray | None:
    if metadata.get("fingerprint") != key or not path.exists():
        return None
    try:
        if file_sha256(path) != metadata.get("embeddings_sha256"):
            raise ValueError("Embedding file checksum mismatch")
        vectors = np.load(path, mmap_mode="r", allow_pickle=False)
        validate_embeddings(vectors, n_rows)
        return vectors
    except (OSError, ValueError, EOFError) as exc:
        LOGGER.warning("Ignoring invalid embedding cache %s: %s", path, exc)
        return None


class ExactIndex:
    def __init__(self, index: Any, doc_ids: Sequence[str], cache_fingerprint: str = ""):
        self.index = index
        self.doc_ids = list(doc_ids)
        self.fingerprint = cache_fingerprint
        if index.ntotal != len(self.doc_ids) or len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError("Index document mapping is inconsistent")

    def search(self, vectors: np.ndarray, top_k: int) -> list[list[dict[str, Any]]]:
        if top_k < 1:
            raise ValueError("Search top_k must be positive")
        validate_embeddings(vectors, len(vectors))
        if vectors.shape[1] != self.index.d:
            raise ValueError("Query/index embedding dimensions differ")
        count = min(top_k, len(self.doc_ids))
        if count == 0:
            return [[] for _ in vectors]
        # One extra neighbor detects ties at the cutoff. Extend only tied rows,
        # so lexicographic doc_id tiebreaking includes every boundary candidate.
        search_count = min(count + 1, len(self.doc_ids))
        scores, positions = self.index.search(np.ascontiguousarray(vectors, dtype=np.float32), search_count)
        output = []
        for row_index in range(len(vectors)):
            row_scores, row_positions = scores[row_index], positions[row_index]
            fetched = search_count
            while fetched < len(self.doc_ids) and row_scores[count - 1] == row_scores[-1]:
                fetched = min(len(self.doc_ids), fetched * 2)
                extended_scores, extended_positions = self.index.search(vectors[row_index : row_index + 1], fetched)
                row_scores, row_positions = extended_scores[0], extended_positions[0]
            neighbors = sorted(
                ((self.doc_ids[int(position)], float(score)) for score, position in zip(row_scores, row_positions)),
                key=lambda pair: (-pair[1], pair[0]),
            )[:count]
            ranked = [
                {"doc_id": doc_id, "rank": rank, "raw_cosine_score": score}
                for rank, (doc_id, score) in enumerate(neighbors, start=1)
            ]
            if len(ranked) != count or len({row["doc_id"] for row in ranked}) != count:
                raise AssertionError("Exact retrieval returned the wrong number of distinct documents")
            output.append(ranked)
        return output


def ensure_corpus_index(
    corpus: Sequence[dict[str, Any]], encoder: DenseEncoder, cache_dir: str | Path, *, force: bool = False
) -> ExactIndex:
    import faiss

    doc_ids = validate_corpus(corpus)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    embeddings_path = cache / "corpus_embeddings.npy"
    index_path = cache / "corpus_index.faiss"
    metadata_path = cache / "corpus_index.meta.json"
    key = fingerprint({"ordered_corpus": corpus, "encoder": encoder.identity, "cache_version": 1})
    metadata = read_metadata(metadata_path)
    vectors = None if force else cached_vectors(embeddings_path, metadata, key, len(corpus))
    if vectors is None:
        LOGGER.info("Encoding %d shared corpus documents", len(corpus))
        vectors = encoder.encode_documents(corpus)
        validate_embeddings(vectors, len(corpus))
        save_numpy(embeddings_path, vectors)
        metadata = {
            "fingerprint": key,
            "encoder": encoder.identity,
            "n_documents": len(corpus),
            "dimension": vectors.shape[1],
            "embeddings_sha256": file_sha256(embeddings_path),
            "index_type": "IndexFlatIP",
        }
        # Saving before index construction allows interrupted builds to resume.
        write_jsonl(metadata_path, [metadata])
    else:
        LOGGER.info("Reusing %d cached corpus embeddings", len(corpus))
    if not force and index_path.exists() and metadata.get("index_sha256"):
        try:
            if file_sha256(index_path) != metadata["index_sha256"]:
                raise ValueError("Index file checksum mismatch")
            index = faiss.read_index(str(index_path))
            if not isinstance(index, faiss.IndexFlatIP) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise ValueError("Cached index is not an exact IndexFlatIP")
            if index.d != vectors.shape[1] or index.ntotal != len(corpus):
                raise ValueError("Cached index dimensions differ")
            LOGGER.info("Reusing exact FAISS index %s", index_path)
            return ExactIndex(index, doc_ids, key)
        except (OSError, ValueError, RuntimeError) as exc:
            LOGGER.warning("Rebuilding invalid FAISS index: %s", exc)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, prefix=".corpus_index.", delete=False) as handle:
            temporary = Path(handle.name)
        faiss.write_index(index, str(temporary))
        os.replace(temporary, index_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    metadata["index_sha256"] = file_sha256(index_path)
    write_jsonl(metadata_path, [metadata])
    LOGGER.info("Built exact IndexFlatIP: %d documents, %d dimensions", index.ntotal, index.d)
    return ExactIndex(index, doc_ids, key)


def build_index(config: dict[str, Any], *, force: bool = False) -> ExactIndex:
    from .contracts import guard_writable, require_stage, record_stage, strict
    guard_writable(config)
    if strict(config):
        require_stage(config, "corpus")
    seed_everything(config["seed"])
    corpus = read_jsonl(config["paths"]["corpus"])
    encoder = DenseEncoder({**config["retrieval"], "seed": config["seed"]})
    result = ensure_corpus_index(corpus, encoder, config["paths"]["cache_dir"], force=force)
    record_stage(config, "index")
    return result
