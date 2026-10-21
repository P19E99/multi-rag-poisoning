"""Build a deterministic paragraph corpus shared by all target questions."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import random
import re
import unicodedata

from .common import file_sha256, fingerprint, read_jsonl, write_jsonl

LOGGER = logging.getLogger(__name__)
CORPUS_VERSION = 2


def normalize_text(value):
    if not isinstance(value, str):
        raise ValueError(f"Expected text, received {type(value).__name__}")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def make_document(title, sentences):
    title = normalize_text(title)
    if isinstance(sentences, str):
        text = normalize_text(sentences)
    elif isinstance(sentences, (list, tuple)):
        text = normalize_text(" ".join(sentences))
    else:
        raise ValueError("Paragraph sentences must be text or a list of strings")
    if not title or not text:
        raise ValueError("Corpus paragraphs must have nonempty title and text")
    return {"doc_id": "hp_" + fingerprint([title, text]), "title": title, "text": text}


def context_paragraphs(row):
    """Support HF struct-of-lists and the original HotpotQA pair lists."""
    context = row.get("context")
    if isinstance(context, dict):
        titles, sentences = context.get("title", []), context.get("sentences", [])
        if len(titles) != len(sentences):
            raise ValueError("Mismatched context title/sentences lengths")
        return list(zip(titles, sentences))
    if isinstance(context, (list, tuple)):
        paragraphs = []
        for item in context:
            if isinstance(item, dict):
                paragraphs.append((item["title"], item["sentences"]))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                paragraphs.append((item[0], item[1]))
            else:
                raise ValueError("Invalid HotpotQA context paragraph")
        return paragraphs
    raise ValueError("Missing or invalid HotpotQA context")


def supporting_facts(row):
    facts = row.get("supporting_facts")
    if isinstance(facts, dict):
        titles, ids = facts.get("title", []), facts.get("sent_id", [])
        if len(titles) != len(ids):
            raise ValueError("Mismatched supporting fact title/sent_id lengths")
        facts = list(zip(titles, ids))
    if not isinstance(facts, (list, tuple)) or not facts:
        raise ValueError("Target must have nonempty supporting_facts")
    result = []
    for fact in facts:
        if isinstance(fact, dict):
            title, sent_id = fact["title"], fact["sent_id"]
        elif isinstance(fact, (list, tuple)) and len(fact) == 2:
            title, sent_id = fact
        else:
            raise ValueError("Invalid supporting fact")
        if isinstance(sent_id, bool) or not isinstance(sent_id, int) or sent_id < 0:
            raise ValueError("Supporting sentence index must be a nonnegative integer")
        result.append((normalize_text(title), sent_id))
    return result


def target_and_gold(row, split, row_index):
    question, answer = row.get("question"), row.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Target {row_index} has no question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Target {row_index} has no answer; choose a labeled split")
    by_title = {}
    for title, sentences in context_paragraphs(row):
        document = make_document(title, sentences)
        normalized_title = document["title"]
        if normalized_title in by_title and by_title[normalized_title][0] != document:
            raise ValueError(f"Ambiguous context title: {normalized_title!r}")
        by_title[normalized_title] = (document, sentences)
    gold = {}
    for title, sent_id in supporting_facts(row):
        if title not in by_title:
            raise ValueError(f"Supporting title absent from target context: {title!r}")
        document, sentences = by_title[title]
        if not isinstance(sentences, str) and sent_id >= len(sentences):
            raise ValueError(f"Supporting sentence {sent_id} outside paragraph {title!r}")
        gold[document["doc_id"]] = document
    sample_id = str(row.get("id") or row.get("_id") or f"{split}:{row_index}")
    return {
        "sample_id": sample_id,
        "question": question.strip(),
        "answer": answer.strip(),
        "supporting_doc_ids": sorted(gold),
        "source_split": split,
        "source_row_index": row_index,
    }, list(gold.values())


def validate_corpus_targets(corpus, targets, expected_corpus_size=None, expected_targets=None):
    if not corpus or not targets:
        raise ValueError("Corpus and targets must both be nonempty")
    ids = set()
    for doc in corpus:
        expected = make_document(doc["title"], doc["text"])
        if expected != doc:
            raise ValueError(f"Invalid or noncanonical corpus document: {doc.get('doc_id')}")
        if doc["doc_id"] in ids:
            raise ValueError(f"Duplicate corpus doc_id: {doc['doc_id']}")
        ids.add(doc["doc_id"])
    sample_ids = set()
    for target in targets:
        if target["sample_id"] in sample_ids:
            raise ValueError(f"Duplicate sample_id: {target['sample_id']}")
        sample_ids.add(target["sample_id"])
        gold = target.get("supporting_doc_ids", [])
        if not gold or len(gold) != len(set(gold)):
            raise ValueError("Target gold IDs must be nonempty and unique")
        if not set(gold).issubset(ids):
            raise ValueError(f"Missing supporting documents for {target['sample_id']}")
        if not target.get("question") or not target.get("answer"):
            raise ValueError("Target missing question or answer")
    if expected_corpus_size is not None and len(corpus) != expected_corpus_size:
        raise ValueError(f"Expected {expected_corpus_size} corpus documents, got {len(corpus)}")
    if expected_targets is not None and len(targets) != expected_targets:
        raise ValueError(f"Expected {expected_targets} targets, got {len(targets)}")


def build_shared_corpus(target_rows, distractor_rows, *, n_targets, corpus_size, seed=42,
                        target_split="validation"):
    """Random access Arrow datasets stay memory mapped; only selected rows materialize."""
    if n_targets < 1 or n_targets > len(target_rows):
        raise ValueError(f"n_targets must be between 1 and {len(target_rows)}")
    if corpus_size < 1:
        raise ValueError("corpus_size must be positive")
    target_rng = random.Random(seed)
    rng = random.Random(fingerprint([seed, "distractor-v1"]))
    indices = target_rng.sample(range(len(target_rows)), n_targets)
    targets, documents = [], {}
    for index in indices:
        target, gold = target_and_gold(target_rows[index], target_split, index)
        targets.append(target)
        documents.update((doc["doc_id"], doc) for doc in gold)
    if len(documents) > corpus_size:
        raise ValueError(f"corpus_size={corpus_size} cannot fit {len(documents)} gold paragraphs")
    row_indices = list(range(len(distractor_rows)))
    rng.shuffle(row_indices)
    for index in row_indices:
        if len(documents) >= corpus_size:
            break
        paragraphs = context_paragraphs(distractor_rows[index])
        rng.shuffle(paragraphs)
        for title, sentences in paragraphs:
            doc = make_document(title, sentences)
            documents.setdefault(doc["doc_id"], doc)
            if len(documents) >= corpus_size:
                break
    if len(documents) < corpus_size:
        raise ValueError(f"Sources exhausted: only {len(documents)} unique paragraphs, "
                         f"requested {corpus_size}")
    corpus = sorted(documents.values(), key=lambda row: row["doc_id"])
    validate_corpus_targets(corpus, targets, corpus_size, n_targets)
    return corpus, targets


def _local_files(path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and
                  p.suffix in {".arrow", ".json", ".jsonl", ".parquet"})


def _source_signature(config):
    settings = config["dataset"]
    source = {key: settings.get(key) for key in
              ("name", "subset", "revision", "target_split", "distractor_split", "local_path")}
    if settings.get("local_path"):
        path = Path(settings["local_path"])
        if not path.exists():
            raise FileNotFoundError(f"dataset.local_path does not exist: {path}")
        source["files"] = [{"path": str(p.relative_to(path) if path.is_dir() else p.name),
                            "bytes": p.stat().st_size, "sha256": file_sha256(p)}
                           for p in _local_files(path)]
        if not source["files"]:
            raise ValueError(f"No supported dataset files found under {path}")
    return source


def load_dataset_splits(settings):
    """Read prepared HF Arrow directories, save_to_disk datasets, JSON, or the Hub."""
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

    splits = list(dict.fromkeys([settings["target_split"], settings["distractor_split"]]))
    local = settings.get("local_path")
    if local:
        path = Path(local)
        if (path / "dataset_dict.json").exists() or (path / "state.json").exists():
            loaded = load_from_disk(str(path))
            if isinstance(loaded, DatasetDict):
                result = {split: loaded[split] for split in splits}
            elif len(splits) == 1:
                result = {splits[0]: loaded}
            else:
                raise ValueError("Single saved Dataset requires identical target/distractor splits")
        elif path.is_file():
            if len(splits) != 1:
                raise ValueError("Single dataset file requires identical target/distractor splits")
            kind = "parquet" if path.suffix == ".parquet" else "json"
            result = {splits[0]: load_dataset(kind, data_files=str(path), split="train",
                                            cache_dir=settings.get("cache_dir"))}
        else:
            result = {}
            for split in splits:
                # Match exact split boundaries so 'train' cannot match a directory name.
                files = sorted(p for p in path.glob("*.arrow") if
                               re.search(rf"(?:^|[-_]){re.escape(split)}(?:[-_.]|$)", p.name))
                if files:
                    parts = [Dataset.from_file(str(p)) for p in files]
                    result[split] = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
                    continue
                candidates = [path / f"{split}{suffix}" for suffix in (".json", ".jsonl", ".parquet")]
                candidates = [p for p in candidates if p.exists()]
                if len(candidates) != 1:
                    raise ValueError(f"Expected Arrow shards or one {split}.json/jsonl/parquet in {path}")
                kind = "parquet" if candidates[0].suffix == ".parquet" else "json"
                result[split] = load_dataset(kind, data_files=str(candidates[0]), split="train",
                                             cache_dir=settings.get("cache_dir"))
    else:
        result = {split: load_dataset(settings["name"], settings.get("subset"), split=split,
                                      revision=settings.get("revision"),
                                      cache_dir=settings.get("cache_dir")) for split in splits}
    for split, rows in result.items():
        LOGGER.info("Loaded %s: %d rows, columns=%s", split, len(rows), rows.column_names)
    return result


def run_build_corpus(config, force=False):
    from .contracts import guard_writable, record_stage
    guard_writable(config)
    settings, paths = config["dataset"], config["paths"]
    corpus_path, targets_path = Path(paths["corpus"]), Path(paths["targets"])
    manifest_path = Path(paths["cache_dir"]) / "corpus_manifest.jsonl"
    source = _source_signature(config)
    datasets = None
    # A Hub dataset fingerprint detects changes even for an unpinned revision.
    if not settings.get("local_path"):
        datasets = load_dataset_splits(settings)
        source["dataset_fingerprints"] = {split: rows._fingerprint for split, rows in datasets.items()}
    identity = {"version": CORPUS_VERSION, "seed": config["seed"], "source": source,
                "n_targets": settings["n_targets"], "corpus_size": settings["corpus_size"]}
    expected_fingerprint = fingerprint(identity)
    if not force and manifest_path.exists() and corpus_path.exists() and targets_path.exists():
        try:
            cached = read_jsonl(manifest_path)[0]
            if (cached["fingerprint"] == expected_fingerprint and
                    cached["corpus_sha256"] == file_sha256(corpus_path) and
                    cached["targets_sha256"] == file_sha256(targets_path)):
                corpus, targets = read_jsonl(corpus_path), read_jsonl(targets_path)
                validate_corpus_targets(corpus, targets, settings["corpus_size"], settings["n_targets"])
                LOGGER.info("Reusing corpus cache: %d paragraphs, %d targets", len(corpus), len(targets))
                record_stage(config, "corpus")
                return corpus, targets
            LOGGER.info("Corpus cache identity or file hashes changed; rebuilding")
        except (ValueError, KeyError, IndexError, OSError) as exc:
            LOGGER.warning("Ignoring invalid corpus cache: %s", exc)
    if datasets is None:
        datasets = load_dataset_splits(settings)
    corpus, targets = build_shared_corpus(
        datasets[settings["target_split"]], datasets[settings["distractor_split"]],
        n_targets=settings["n_targets"], corpus_size=settings["corpus_size"],
        seed=config["seed"], target_split=settings["target_split"])
    write_jsonl(corpus_path, corpus)
    write_jsonl(targets_path, targets)
    write_jsonl(manifest_path, [{"fingerprint": expected_fingerprint, **identity,
                                "corpus_sha256": file_sha256(corpus_path),
                                "targets_sha256": file_sha256(targets_path),
                                "source_rows": {split: len(rows) for split, rows in datasets.items()}}])
    LOGGER.info("Saved shared corpus: %d paragraphs, %d targets, complete gold coverage",
                len(corpus), len(targets))
    record_stage(config, "corpus")
    return corpus, targets
