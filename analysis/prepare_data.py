"""One-time, read-only conversion of frozen experiment records.

Usage: python analysis/prepare_data.py --source /path/to/original/fp
Writes only beside this script, under artifact/data.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
QS = (1, 2, 4, 8)


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    if HERE == source or HERE in source.parents:
        raise ValueError("Source must be the original experiment directory")
    with (source / "query_diversity_analysis/gold_document_inventory.csv").open(encoding="utf-8-sig", newline="") as stream:
        gold_rows = list(csv.DictReader(stream))
    answer_src = source / "exports/multiquery_rag/target_questions_with_answers.csv"
    with answer_src.open(encoding="utf-8", newline="") as stream:
        answers = list(csv.DictReader(stream))
    ids = {r["target_id"] for r in answers}
    matches = []
    for directory in (source / "cache").glob("*/main"):
        paths = sorted(directory.glob("*.retrieval.json"))
        rewrite_path = source / "data" / directory.parent.name / "main/query_views.jsonl"
        if len(paths) != 100 or not rewrite_path.is_file():
            continue
        if {path.name.removesuffix(".retrieval.json") for path in paths} != ids:
            continue
        rewrites = jsonl(rewrite_path)
        if len(rewrites) != 100 or {r["sample_id"] for r in rewrites} != ids:
            continue
        query_views = {r["sample_id"]: r["query_views"] for r in rewrites}
        if all(next(r for r in json.loads(path.read_text(encoding="utf-8"))["records"]
                    if r["query_count"] == 8 and r["treatment"] == "poisoned")["query_views"]
               == query_views[path.name.removesuffix(".retrieval.json")] for path in paths):
            matches.append((paths, rewrites))
    if len(matches) != 1:
        raise ValueError(f"Expected one matching frozen retrieval and rewrite pair, got {len(matches)}")
    paths, rewrites = matches[0]
    if not (len(paths) == len(rewrites) == len(gold_rows) == len(ids) == 100):
        raise ValueError("Expected exactly 100 aligned targets in all four inputs")
    if {r["sample_id"] for r in rewrites} != ids or {r["target_id"] for r in gold_rows} != ids:
        raise ValueError("Target IDs disagree between intermediate results")
    compact_rewrites = []
    for row in rewrites:
        if len(row["rewrites"]) != 7 or row["query_views"] != [row["original_query"], *row["rewrites"]]:
            raise ValueError(f"Incomplete seven-rewrite pool: {row['sample_id']}")
        compact_rewrites.append({"target_id": row["sample_id"], "original": row["original_query"], "rewrites": row["rewrites"]})
    compact_gold = []
    for row in gold_rows:
        known = json.loads(row["known_supporting_doc_ids"].replace("'", '"'))
        if len(known) != 2 or int(row["missing_supporting_doc_ids"]) != 0:
            raise ValueError(f"Incomplete gold mapping: {row['target_id']}")
        compact_gold.append({"target_id": row["target_id"], "gold_doc_ids": known})
    records = []
    for path in paths:
        sid = path.name.removesuffix(".retrieval.json")
        if sid not in ids:
            raise ValueError(f"Unknown retrieval target: {sid}")
        raw = json.loads(path.read_text(encoding="utf-8"))["records"]
        by = {(r["query_count"], r["treatment"]): r for r in raw}
        if len(raw) != 8 or set(by) != {(q, t) for q in QS for t in ("clean", "poisoned")}:
            raise ValueError(f"Incomplete retrieval records: {sid}")
        for treatment in ("clean", "poisoned"):
            full = by[(8, treatment)]
            lists = []
            for view in full["per_view_retrieval"]:
                ranked = view["ranked_list"]
                if len(ranked) != 10 or [hit["rank"] for hit in ranked] != list(range(1, 11)):
                    raise ValueError(f"Malformed top-10: {sid}/{treatment}")
                lists.append([[hit["doc_id"], hit["raw_cosine_score"]] for hit in ranked])
            if len(lists) != 8:
                raise ValueError(f"Missing retrieval views: {sid}/{treatment}")
            for q in QS:
                short = by[(q, treatment)]
                if [[(h["doc_id"], h["raw_cosine_score"]) for h in v["ranked_list"]] for v in short["per_view_retrieval"]] != [list(map(tuple, view)) for view in lists[:q]]:
                    raise ValueError(f"Non-nested retrieval lists: {sid}/{treatment}/q{q}")
            records.append({"target_id": sid, "treatment": treatment, "poison_doc_id": full["poison_document_id"], "views": lists})
    DATA.mkdir(parents=True, exist_ok=True)
    for name, rows in (("retrieval_views.jsonl", records), ("rewrites.jsonl", compact_rewrites), ("gold_documents.jsonl", compact_gold)):
        with (DATA / name).open("w", encoding="utf-8") as stream:
            for row in sorted(rows, key=lambda x: (x["target_id"], x.get("treatment", ""))):
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    shutil.copyfile(answer_src, DATA / "target_questions_with_answers.csv")
    shutil.copyfile(source / "exports/multiquery_rag/multiquery_rag_target_results.csv", DATA / "multiquery_rag_target_results.csv")
    print("Converted 100 targets, 200 treatment records, 1,600 top-10 view lists, 700 rewrites, 100 gold mappings, 3,200 outcomes")


if __name__ == "__main__":
    main()
