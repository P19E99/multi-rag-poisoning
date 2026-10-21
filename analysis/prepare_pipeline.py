"""Copy reviewable pipeline modules from a read-only source tree.

Usage: python analysis/prepare_pipeline.py --source /path/to/original/fp
Writes only under artifact/pipeline. External paths are replaced with placeholders.
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    "src/adversarial/prepare.py", "src/adversarial/mechanism.py",
    "src/adversarial/validation.py", "src/adversarial/lm.py",
    "src/core/corpus.py", "src/core/encoder.py", "src/core/index.py",
    "src/core/retrieval.py", "src/core/fusion.py", "src/core/expansion.py",
    "src/core/metrics.py", "src/core/common.py",
    "src/runtime/runtime.py", "src/runtime/contract.py",
    "src/attack/study.py", "src/attack/backend.py", "src/attack/serialization.py",
    "experiments/baseline/construct.py", "experiments/baseline/retrieve.py",
    "experiments/baseline/generate.py", "experiments/baseline/score.py",
    "experiments/consensus/generate.py", "experiments/consensus/score.py",
    "experiments/consensus/c4_rerun/generate_c4.py",
    "experiments/consensus/c4_rerun/score_repair.py",
    "export_multiquery_rag.py",
]


def only(paths, label):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f"Expected one {label} source, found {len(paths)}")
    return paths[0]


def discover(source):
    src = source / "src"
    core = only((p for p in src.iterdir() if p.is_dir() and
                 all((p / n).is_file() for n in ("corpus.py", "retrieval.py", "fusion.py"))), "core")
    runtime = only((p for p in src.iterdir() if p.is_dir() and
                    all((p / n).is_file() for n in ("runtime.py", "contract.py"))), "runtime")
    attack = max((p for p in src.iterdir() if p.is_dir() and
                  all((p / n).is_file() for n in ("study.py", "backend.py", "serialization.py"))),
                 key=lambda p: p.name)
    baseline = only((p for p in source.iterdir() if p.is_dir() and
                     all((p / n).is_file() for n in ("construct.py", "retrieve.py", "generate.py", "score.py"))),
                    "baseline experiment")
    consensus = only((p for p in source.iterdir() if p.is_dir() and
                      all((p / n).is_file() for n in ("generate.py", "score.py")) and
                      any(q.is_dir() and (q / "generate_c4.py").is_file() for q in p.iterdir())),
                     "consensus experiment")
    rerun = only((p for p in consensus.iterdir() if p.is_dir() and
                  (p / "generate_c4.py").is_file() and (p / "score_repair.py").is_file()), "C4 rerun")
    frozen = only((p for p in (source / "cache").glob("*/main")
                   if len(list(p.glob("*.retrieval.json"))) == 100), "frozen retrieval")
    target_data = only((p for p in (source / "data").iterdir() if p.is_dir() and
                        (p / "targets.jsonl").is_file() and
                        sum(1 for line in (p / "targets.jsonl").open(encoding="utf-8") if line.strip()) == 100),
                       "target data")
    corpus_data = only((p for p in (source / "data").iterdir() if p.is_dir() and
                        (p / "corpus.jsonl").is_file()), "corpus data")
    query_dirs = sorted((p for p in (source / "data").iterdir() if p.is_dir() and
                         (p / "main/query_views.jsonl").is_file()), key=lambda p: p.name)
    earlier_queries = query_dirs[query_dirs.index(source / "data" / frozen.parent.name) - 1]
    return {"core": core, "runtime": runtime, "attack": attack,
            "baseline": baseline, "consensus": consensus, "c4_rerun": rerun,
            "frozen": frozen.parent, "target_data": target_data,
            "corpus_data": corpus_data, "earlier_queries": earlier_queries,
            "probes": only(source.glob("*probes"), "probes"),
            "multiple": only(source.glob("*multipoison"), "multiple-document experiment"),
            "summary": only(source.glob("*x_analysis"), "summary analysis")}


def source_path(source, found, name):
    parts = Path(name).parts
    if parts[:2] == ("src", "adversarial"):
        return source / name
    if len(parts) == 3 and parts[:2] in (("src", "core"), ("src", "runtime"), ("src", "attack")):
        return found[parts[1]] / parts[2]
    if parts[:2] == ("experiments", "baseline"):
        return found["baseline"] / parts[-1]
    if parts[:2] == ("experiments", "consensus"):
        return (found["c4_rerun"] if len(parts) == 4 else found["consensus"]) / parts[-1]
    return source / name


def rename_references(body, found):
    body = re.sub(r"fresh_step\d+_main", "fresh_frozen_main", body)
    body = re.sub(r"step\d+_5b", "consensus_single", body)
    body = re.sub(r"step\d+_6_n", "multi_document_n", body)
    replacements = [
        (found["consensus"].name + "/" + found["c4_rerun"].name,
         "experiments/consensus/c4_rerun"),
        (found["baseline"].name, "experiments/baseline"),
        (found["consensus"].name, "experiments/consensus"),
        (found["c4_rerun"].name, "c4_rerun"),
        (found["multiple"].name, "multi_document"),
        (found["probes"].name, "consensus_probes"),
        (found["summary"].name, "summary_analysis"),
        (found["frozen"].name, "frozen_retrieval"),
        (found["earlier_queries"].name, "frozen_queries"),
        (found["target_data"].name, "target_selection"),
        (found["runtime"].name, "runtime"),
        (found["attack"].name, "attack"),
        (found["corpus_data"].name, "corpus_base"),
        ("This " + found["core"].name, "This pipeline"),
        (found["core"].name, "core"),
    ]
    for old, new in replacements:
        body = body.replace(old, new)
    return re.sub(r"Step\s+\d+(?:\.[0-9a-z]+)?", "prior phase", body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    source = parser.parse_args().source.resolve()
    if ROOT == source or ROOT in source.parents:
        raise ValueError("Source must be the original experiment directory")
    found = discover(source)
    for name in MODULES:
        body = source_path(source, found, name).read_text(encoding="utf-8")
        body = body.replace(str(source), "<experiment-root>")
        body = body.replace(str(source.parent), "<workspace-root>")
        root_prefix = "/" + "root" + "/"
        body = re.sub(re.escape(root_prefix) + r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+", "<local-path>", body)
        body = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)['\"][^'\"]+['\"]", r"\1'<redacted>'", body)
        body = rename_references(body, found)
        if root_prefix in body or re.search(r"(?i)" + "sk" + r"-[A-Za-z0-9]{12,}", body):
            raise ValueError(f"Unsanitized path or key in {name}")
        target = ROOT / "pipeline" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    print(f"Copied and scanned {len(MODULES)} pipeline modules")


if __name__ == "__main__":
    main()
