"""prior phase single-poison exact retrieval overlays (construction is frozen)."""
from __future__ import annotations
import json
from pathlib import Path
import faiss
import numpy as np
from core.common import file_sha256, fingerprint, read_jsonl
from core.encoder import DenseEncoder
from core.index import ExactIndex
from adversarial.mechanism import mechanism_record

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/baseline"

def main():
    targets = {x["sample_id"]: x for x in read_jsonl(ROOT / "data/target_selection/targets.jsonl")}
    views = {x["sample_id"]: x for x in read_jsonl(ROOT / "data/frozen_retrieval/main/query_views.jsonl")}
    poisons = {x["sample_id"]: x for x in read_jsonl(OUT / "construction/poisons.jsonl")}
    corpus = read_jsonl(ROOT / "data/corpus_base/corpus.jsonl")
    lookup = {d["doc_id"]: d for d in corpus}
    base_path = ROOT / "cache/corpus_base/corpus_index.faiss"
    base = faiss.read_index(str(base_path))
    base_ids = [d["doc_id"] for d in corpus]
    enc = DenseEncoder({"model_name": "BAAI/bge-base-en-v1.5", "model_path": str(ROOT.parent / "models/bge-base-en-v1.5"), "local_files_only": True, "device": "cuda", "batch_size": 32, "max_length": 512, "query_instruction": "Represent this sentence for searching relevant passages: ", "seed": 42})
    ordered = sorted(poisons)
    docs = [poisons[s]["poison_document"] for s in ordered]
    vectors = enc.encode_documents(docs)
    OUT.joinpath("retrieval").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "retrieval/consensus_poison_embeddings.npy", vectors)
    embedding_hash = file_sha256(OUT / "retrieval/consensus_poison_embeddings.npy")
    exposure_failures = []
    all_counts = {str(q): 0 for q in (1, 2, 4, 8)}
    for i, sid in enumerate(ordered):
        t, attack, vr = targets[sid], poisons[sid], views[sid]
        idx = faiss.clone_index(base)
        vec = np.ascontiguousarray(vectors[i:i+1], dtype=np.float32)
        idx.add(vec)
        poison_id = attack["poison_document"]["doc_id"]
        ex = ExactIndex(idx, base_ids + [poison_id], fingerprint({"base_index": file_sha256(base_path), "target_id": sid, "poison_id": poison_id, "poison_vector_sha256": fingerprint(vec.tolist())}))
        ranked = ex.search(np.load(ROOT / "cache/frozen_retrieval/main" / f"{sid}.queries.npy", allow_pickle=False), 10)
        docs_for_target = dict(lookup)
        docs_for_target[poison_id] = attack["poison_document"]
        records = []
        for q in (1, 2, 4, 8):
            rec = mechanism_record(t, {"attack_id": "PoisonedRAG-LM_targeted-consensus-breaking", "target_answer": attack["target_answer"], "poison_document": attack["poison_document"]}, vr["query_views"][:q], ranked[:q], docs_for_target, q, "poisoned", seed=42, final_k=5, k_rrf=60)
            rec.update({"variant": attack["variant"], "construction_record_sha256": attack["record_sha256"], "poison_embedding_sha256": embedding_hash, "base_index_sha256": file_sha256(base_path), "index_fingerprint": ex.fingerprint, "retrieval_protocol": "exact IndexFlatIP; per-view Top-10; q=1 cosine; q>1 RRF k=60; final Top-5"})
            records.append(rec)
            all_counts[str(q)] += int(rec["fusion_survival"])
            if not rec["fusion_survival"]:
                exposure_failures.append({"sample_id": sid, "q": q, "view_coverage": rec["view_coverage"], "poison_fused_rank": rec["poison_fused_rank"], "stage": "initial_construct"})
        (OUT / "retrieval" / f"{sid}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
        if (i + 1) % 10 == 0:
            print(f"retrieved {i+1}/100", flush=True)
    (OUT / "retrieval" / "exposure_failures.json").write_text(json.dumps({"n_failures": len(exposure_failures), "failures": exposure_failures, "tier_ladder_invoked": False if not exposure_failures else True}, indent=2) + "\n")
    (OUT / "retrieval" / "manifest.json").write_text(json.dumps({"n_targets": len(ordered), "n_records": len(ordered) * 4, "poison_survival_counts": all_counts, "embedding_sha256": embedding_hash, "base_index_sha256": file_sha256(base_path), "construction_sha256": file_sha256(OUT / "construction/poisons.jsonl"), "exposure_failures": len(exposure_failures)}, indent=2) + "\n")
    print(json.dumps({"n_targets": len(ordered), "n_records": len(ordered) * 4, "poison_survival_counts": all_counts, "exposure_failures": len(exposure_failures)}, indent=2))
    if exposure_failures:
        raise SystemExit("Exposure failure requires the preregistered Tier A->B->C ladder; stopping before generation.")

if __name__ == "__main__":
    main()
