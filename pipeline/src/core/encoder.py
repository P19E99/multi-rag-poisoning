"""The shared BGE encoder: CLS pooling and unit-length float32 vectors."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .common import file_sha256, fingerprint, seed_everything

LOGGER = logging.getLogger(__name__)


def encoder_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Identify encoding semantics and local model artifacts without loading a GPU."""
    location = config.get("model_path") or config["model_name"]
    model_dir = Path(location)
    manifest = []
    if model_dir.is_dir():
        for path in sorted(model_dir.rglob("*")):
            if not path.is_file() or ".cache" in path.parts:
                continue
            if path.suffix not in {".json", ".txt", ".model", ".safetensors", ".bin"}:
                continue
            stat = path.stat()
            item = {"name": str(path.relative_to(model_dir)), "size": stat.st_size}
            if path.suffix in {".safetensors", ".bin"}:
                # Weight manifests avoid rereading nearly 1 GB on every stage.
                item["mtime_ns"] = stat.st_mtime_ns
            else:
                item["sha256"] = file_sha256(path)
            manifest.append(item)
    return {
        "implementation": "transformers-cls-l2-float32-v1",
        "source_sha256": file_sha256(__file__),
        "execution": {key: config.get(key) for key in ("seed", "device", "batch_size")},
        "packages": __import__("core.contracts", fromlist=["package_versions"]).package_versions(),
        "model_name": config["model_name"],
        "model_location": str(location),
        "revision": config.get("revision"),
        "local_model_manifest": manifest,
        "max_length": int(config["max_length"]),
        "query_instruction": config.get("query_instruction", ""),
        "document_format": "title + newline + text",
    }


class DenseEncoder:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.identity = encoder_identity(config)
        self.fingerprint = fingerprint(self.identity)
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        seed_everything(int(self.config.get("seed", 42)))

        device = str(self.config.get("device", "cpu"))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; set retrieval.device=cpu for CPU embedding.")
        self.device = device
        location = self.config.get("model_path") or self.config["model_name"]
        options = {
            "local_files_only": self.config.get("local_files_only", False),
            "trust_remote_code": False,
        }
        if self.config.get("revision"):
            options["revision"] = self.config["revision"]
        LOGGER.info("Loading BGE encoder from %s on %s", location, device)
        self._tokenizer = AutoTokenizer.from_pretrained(location, **options)
        self._model = AutoModel.from_pretrained(location, **options).to(device).eval()

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("Cannot encode an empty sequence")
        import torch
        import torch.nn.functional as functional

        self._load()
        batch_size = int(self.config["batch_size"])
        if batch_size < 1:
            raise ValueError("retrieval.batch_size must be positive")
        output = []
        with torch.inference_mode():
            for offset in range(0, len(texts), batch_size):
                encoded = self._tokenizer(
                    list(texts[offset : offset + batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=int(self.config["max_length"]),
                    return_tensors="pt",
                ).to(self.device)
                vectors = self._model(**encoded).last_hidden_state[:, 0].float()
                vectors = functional.normalize(vectors, p=2, dim=1)
                output.append(vectors.cpu().numpy())
                LOGGER.info("Embedded %d/%d texts", min(offset + batch_size, len(texts)), len(texts))
        result = np.ascontiguousarray(np.concatenate(output), dtype=np.float32)
        validate_embeddings(result, len(texts))
        return result

    def encode_documents(self, documents: Sequence[dict[str, Any]]) -> np.ndarray:
        return self._encode([f"{doc['title']}\n{doc['text']}" for doc in documents])

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        instruction = self.config.get("query_instruction", "")
        return self._encode([instruction + query for query in queries])


def validate_embeddings(vectors: np.ndarray, n_rows: int) -> None:
    if vectors.ndim != 2 or vectors.shape[0] != n_rows or vectors.shape[1] < 1:
        raise ValueError(f"Invalid embedding shape {vectors.shape}; expected {n_rows} rows")
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be finite float32 values")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
        raise ValueError("All embeddings must have unit L2 norm")
