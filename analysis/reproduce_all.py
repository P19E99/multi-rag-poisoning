#!/usr/bin/env python3
"""Recompute paper metrics from the small, frozen data directory.

Run from any directory: python analysis/reproduce_all.py --check
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import statistics
import string
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
QS = (1, 2, 4, 8)
MODELS = ("qwen3-8b", "llama-3.1-8b-instruct")
CONDITIONS = ("C1", "C2", "C3", "C4")
TOL_PP = 0.05
REPS = 10_000
SEED = 42

# Frozen, rounded manuscript reference values. All percentages are in pp.
EXPECTED = {
    "rank1": {1: 85, 2: 82, 4: 83, 8: 83},
    "corrected_qwen_c1": {1: 20.8, 2: 20.0, 4: 17.7, 8: 17.7},
    "lexical": {"rewrite_pair": .6161, "original_rewrite": .6434},
    "retrieval_overlap": {"poisoned": .7266, "clean": .7032},
    "gold_candidate": {1: 1.78, 8: 1.81},
    "gold_top5": {1: 1.70, 2: 1.66, 4: 1.70, 8: 1.69},
    "subset_changes": {2: 11, 4: 12},
    "cochran": {"qwen3-8b": (1.60, .659, 95), "llama-3.1-8b-instruct": (3.00, .392, 95)},
    "bootstrap_upper": {("qwen3-8b", "C1"): 7.4, ("qwen3-8b", "C3"): 4.2,
                        ("llama-3.1-8b-instruct", "C1"): 2.1, ("llama-3.1-8b-instruct", "C3"): 8.4},
}


def read_jsonl(name):
    with (DATA / name).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_csv(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Checker:
    def __init__(self, enabled):
        self.enabled = enabled
        self.failures = []

    def eq(self, label, got, expected, tolerance=0, quiet=False):
        okay = abs(got - expected) <= tolerance + 1e-10
        if self.enabled and not okay:
            self.failures.append(label)
        if not quiet or not okay:
            print(f"{'PASS' if okay else 'FAIL'} {label}: {got:.4f} (paper {expected:.4f})")
        return okay

    def truth(self, label, okay, detail=""):
        if self.enabled and not okay:
            self.failures.append(label)
        print(f"{'PASS' if okay else 'FAIL'} {label}{': ' + detail if detail else ''}")


def fuse(lists, method="RRF"):
    scores = defaultdict(float)
    for view in lists:
        if len(view) != 10 or len({hit[0] for hit in view}) != 10:
            raise ValueError("Each view needs ten unique ranked hits")
        for rank, (doc, score) in enumerate(view, 1):
            if method == "RRF":
                scores[doc] += 1 / (60 + rank)
            elif method == "MaxScore":
                scores[doc] = max(scores[doc], score) if doc in scores else score
            elif method == "AvgScore":
                if score < 0:
                    raise ValueError("AvgScore zero imputation needs nonnegative scores")
                scores[doc] += score
            else:
                raise ValueError(method)
    if method == "RRF" and len(lists) == 1:
        return [hit[0] for hit in lists[0]]
    # AvgScore divides by q for every document. The common divisor preserves order.
    return sorted(scores, key=lambda doc: (-scores[doc], doc))


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


def lexical_tokens(value):
    return set(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


def retrieval_and_rewrites(check):
    rows = read_jsonl("retrieval_views.jsonl")
    rewrites = read_jsonl("rewrites.jsonl")
    gold = read_jsonl("gold_documents.jsonl")
    by = {(r["target_id"], r["treatment"]): r for r in rows}
    gold_by = {r["target_id"]: set(r["gold_doc_ids"]) for r in gold}
    ids = set(gold_by)
    check.truth("data coverage: 100 targets, 200 treatments, 8 views, 7 rewrites, 2 gold IDs",
                len(ids) == len(rewrites) == 100 and len(by) == 200 and
                all(len(r["views"]) == 8 for r in rows) and
                all(len(r["rewrites"]) == 7 and r["target_id"] in ids for r in rewrites) and
                all(len(g) == 2 for g in gold_by.values()))
    token_orig, token_pairs = [], []
    for r in rewrites:
        tokens = [lexical_tokens(s) for s in [r["original"], *r["rewrites"]]]
        token_orig.append(statistics.mean(jaccard(tokens[0], tokens[i]) for i in range(1, 8)))
        token_pairs.append(statistics.mean(jaccard(tokens[i], tokens[j]) for i, j in itertools.combinations(range(1, 8), 2)))
    for label, values in (("original_rewrite", token_orig), ("rewrite_pair", token_pairs)):
        check.eq(f"lexical Jaccard {label}", statistics.mean(values), EXPECTED["lexical"][label], .00005)
    for treatment in ("poisoned", "clean"):
        per_target = []
        for sid in sorted(ids):
            views = by[(sid, treatment)]["views"]
            sets = [set(hit[0] for hit in view) for view in views]
            per_target.append(statistics.mean(jaccard(sets[i], sets[j]) for i, j in itertools.combinations(range(8), 2)))
        check.eq(f"retrieval top10 Jaccard {treatment}", statistics.mean(per_target), EXPECTED["retrieval_overlap"][treatment], .00005)
    print("Table I: q | view exposure | median rank | rank1 | top5 | gold candidates | gold top5")
    for q in QS:
        ranks, view_hits, gold_candidates, gold_top5 = [], [], [], []
        for sid in sorted(ids):
            row = by[(sid, "poisoned")]
            views = row["views"][:q]
            pid = row["poison_doc_id"]
            order = fuse(views)
            ranks.append(order.index(pid) + 1 if pid in order else len(order) + 1)
            view_hits.append(any(pid in {hit[0] for hit in v} for v in views))
            gold_candidates.append(len(set(order) & gold_by[sid]))
            gold_top5.append(len(set(order[:5]) & gold_by[sid]))
        exposure = 100 * statistics.mean(view_hits)
        rank1 = 100 * statistics.mean(rank == 1 for rank in ranks)
        top5 = 100 * statistics.mean(rank <= 5 for rank in ranks)
        med = statistics.median(ranks)
        cand, top_gold = statistics.mean(gold_candidates), statistics.mean(gold_top5)
        print(f"  {q} | {exposure:.1f}% | {med:g} | {rank1:.1f}% | {top5:.1f}% | {cand:.2f} | {top_gold:.2f}")
        for label, got, expected in (("view exposure", exposure, 100), ("median rank", med, 1),
                                     ("rank1", rank1, EXPECTED["rank1"][q]), ("top5", top5, 100)):
            check.eq(f"Table I q={q} {label}", got, expected, TOL_PP if label != "median rank" else 0)
        if q in EXPECTED["gold_candidate"]:
            check.eq(f"gold candidate q={q}", cand, EXPECTED["gold_candidate"][q], .005)
        check.eq(f"gold top5 q={q}", top_gold, EXPECTED["gold_top5"][q], .005)
    for q in (2, 4):
        variants = [(0, *combo) for combo in itertools.combinations(range(1, 8), q - 1)]
        changed = 0
        all_top5 = []
        for sid in sorted(ids):
            row = by[(sid, "poisoned")]
            pid = row["poison_doc_id"]
            ranks = []
            for choice in variants:
                order = fuse([row["views"][i] for i in choice])
                ranks.append(order.index(pid) + 1 if pid in order else len(order) + 1)
                all_top5.append(pid in order[:5])
            changed += len(set(ranks)) > 1
        check.eq(f"subset q={q} targets with varying poison rank", changed, EXPECTED["subset_changes"][q])
        check.eq(f"subset q={q} all-combination top5 rate", 100 * statistics.mean(all_top5), 100, TOL_PP)
    for method in ("MaxScore", "AvgScore"):
        for q in (2, 4, 8):
            ranks = []
            for sid in sorted(ids):
                row = by[(sid, "poisoned")]
                order = fuse(row["views"][:q], method)
                pid = row["poison_doc_id"]
                ranks.append(order.index(pid) + 1 if pid in order else len(order) + 1)
            check.eq(f"{method} q={q} median rank", statistics.median(ranks), 1)
            check.eq(f"{method} q={q} top5 rate", 100 * statistics.mean(r <= 5 for r in ranks), 100, TOL_PP)


def answer_after_final(raw):
    found = list(re.finditer("final answer", raw, flags=re.IGNORECASE))
    return raw[found[-1].end():] if found else raw


def answer_tokens(raw):
    raw = answer_after_final(raw).lower()
    return "".join(ch for ch in raw if ch not in string.punctuation).split()


def em_f1(raw, reference, abstained):
    if abstained:
        return 0.0, 0.0
    a, b = answer_tokens(raw), answer_tokens(reference)
    em = float(a == b)
    if not a or not b:
        return em, float(a == b)
    overlap = sum((Counter(a) & Counter(b)).values())
    if not overlap:
        return em, 0.0
    return em, 2 * overlap / (len(a) + len(b))


def jeffreys(successes, n):
    return (100 * beta.ppf(.025, successes + .5, n - successes + .5),
            100 * beta.ppf(.975, successes + .5, n - successes + .5))


def cochran(matrix):
    x = np.asarray(matrix, dtype=int)
    n, k = x.shape
    columns = x.sum(axis=0)
    total = int(columns.sum())
    denom = k * total - int(np.square(x.sum(axis=1)).sum())
    value = (k - 1) * (k * int(np.square(columns).sum()) - total**2) / denom if denom else 0.0
    z = math.sqrt(value / 2)
    p = math.erfc(z) + 2 * z * math.exp(-z*z) / math.sqrt(math.pi)
    return value, p, n


def percentile_bootstrap_upper(values):
    x = np.asarray(values, dtype=float)
    n = len(x)
    # Legacy MT19937 RandomState is required for the frozen 10k-draw manuscript CI.
    rng = np.random.RandomState(SEED)
    draws = rng.randint(0, n, size=(REPS, n))
    means = x[draws].mean(axis=1)
    return 100 * float(np.quantile(means, .975))


def exact_p(gain, loss):
    return float(binomtest(gain, gain + loss, .5, alternative="two-sided").pvalue) if gain + loss else 1.0


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    running = 0.0
    for j, i in enumerate(order):
        running = max(running, min(1.0, (len(values) - j) * values[i]))
        out[i] = running
    return out


def outcomes(check):
    rows = read_csv("multiquery_rag_target_results.csv")
    refs = {r["target_id"]: r["reference_answer"] for r in read_csv("target_questions_with_answers.csv")}
    index = {}
    for row in rows:
        key = (row["generator"], row["condition"], int(row["q"]), row["target_id"])
        if key in index:
            raise ValueError(f"Duplicate outcome: {key}")
        index[key] = row
    targets = sorted(refs)
    check.truth("outcome coverage: 3,200 unique rows / 100 targets / 16 cells per model",
                len(targets) == 100 and len(index) == 3200 and
                all((m, c, q, sid) in index for m in MODELS for c in CONDITIONS for q in QS for sid in targets))
    print("Table II: model condition q | ASR [Jeffreys 95% CI] | abstain | EM | F1 (percent)")
    table = {}
    for model in MODELS:
        for condition in CONDITIONS:
            for q in QS:
                group = [index[(model, condition, q, sid)] for sid in targets]
                eligible = [r for r in group if r["clean_hit"] == "0"]
                successes = sum(int(r["attack_success"]) for r in eligible)
                asr = 100 * successes / len(eligible)
                low, high = jeffreys(successes, len(eligible))
                abstain = 100 * sum(int(r["abstained"]) for r in group) / 100
                metrics = [em_f1(r["raw_answer"], refs[r["target_id"]], int(r["abstained"])) for r in group]
                em, f1 = (100 * statistics.mean(m[i] for m in metrics) for i in (0, 1))
                table[(model, condition, q)] = (asr, low, high, abstain, em, f1)
                print(f"  {model} {condition} q={q}: {asr:.2f} [{low:.2f},{high:.2f}] | {abstain:.1f} | {em:.2f} | {f1:.2f}")
                if model == "qwen3-8b" and condition == "C1":
                    check.eq(f"Table I corrected ASR q={q}", asr, EXPECTED["corrected_qwen_c1"][q], TOL_PP)
                if model == "qwen3-8b" and condition in ("C2", "C4"):
                    lo, hi = (26, 28) if condition == "C2" else (20, 28)
                    check.truth(f"Qwen {condition} q={q} abstention range", lo <= abstain <= hi)
                if model == "llama-3.1-8b-instruct" and condition in ("C2", "C4"):
                    lo, hi = (9.5, 11.5) if condition == "C2" else (7, 12)
                    check.truth(f"Llama {condition} q={q} abstention range", lo <= abstain <= hi)
    # The manuscript's Table II cells are frozen in this separate reference file.
    expected_table = json.loads((DATA / "table_ii_reference.json").read_text(encoding="utf-8"))
    for (model, condition, q), values in table.items():
        paper = expected_table[f"{model}/{condition}/q{q}"]
        ok = []
        for name, got, target in zip(("ASR", "CI low", "CI high", "abstain", "EM", "F1"), values, paper):
            ok.append(check.eq(f"Table II {model} {condition} q={q} {name}", got, target, TOL_PP, quiet=True))
        print(f"{'PASS' if all(ok) else 'FAIL'} Table II {model} {condition} q={q}: all 6 metrics")
    for model in MODELS:
        for condition in ("C1", "C3"):
            common = [sid for sid in targets if all(index[(model, condition, q, sid)]["clean_hit"] == "0" for q in QS)]
            matrix = [[int(index[(model, condition, q, sid)]["attack_success"]) for q in QS] for sid in common]
            if condition == "C1":
                value, p, n = cochran(matrix)
                eq, ep, en = EXPECTED["cochran"][model]
                check.eq(f"Cochran Q {model} C1 (df=3,n={n})", value, eq, .005)
                check.eq(f"Cochran p {model} C1", p, ep, .0005)
                check.eq(f"Cochran n {model} C1", n, en)
            paired = [sid for sid in targets if index[(model, condition, 1, sid)]["clean_hit"] == "0"
                      and index[(model, condition, 8, sid)]["clean_hit"] == "0"]
            diffs = [int(index[(model, condition, 1, sid)]["attack_success"]) -
                     int(index[(model, condition, 8, sid)]["attack_success"]) for sid in paired]
            upper = percentile_bootstrap_upper(diffs)
            check.eq(f"percentile bootstrap 95% upper q1-q8 {model} {condition} (n={len(paired)})", upper,
                     EXPECTED["bootstrap_upper"][(model, condition)], TOL_PP)
    print("McNemar exact two-sided + Holm within each model/comparison across four q")
    for model in MODELS:
        for first, second in (("C2", "C1"), ("C4", "C3")):
            raw = []
            for q in QS:
                paired = [sid for sid in targets if index[(model, first, q, sid)]["clean_hit"] == "0"
                          and index[(model, second, q, sid)]["clean_hit"] == "0"]
                a = [int(index[(model, first, q, sid)]["attack_success"]) for sid in paired]
                b = [int(index[(model, second, q, sid)]["attack_success"]) for sid in paired]
                gain = sum(x and not y for x, y in zip(a, b))
                loss = sum(y and not x for x, y in zip(a, b))
                raw.append(exact_p(gain, loss))
            adjusted = holm(raw)
            print(f"  {model} {first}vs{second}: exact={','.join(f'{p:.6g}' for p in raw)} Holm={','.join(f'{p:.6g}' for p in adjusted)}")
            if model == "qwen3-8b":
                check.truth(f"Qwen {first}vs{second} all q significant", all(p < .05 for p in adjusted))
            else:
                check.truth(f"Llama {first}vs{second} all q nonsignificant", all(p >= .05 for p in adjusted))
                floor = .18 if first == "C2" else .65
                check.truth(f"Llama {first}vs{second} p floor at paper precision (2 decimals)",
                            min(round(p, 2) for p in raw) >= floor)
    for condition in ("C2", "C4"):
        rates = []
        for q in QS:
            group = [index[("llama-3.1-8b-instruct", condition, q, sid)] for sid in targets]
            rates.append(100 * statistics.mean(bool(re.search("final answer", r["raw_answer"], re.I)) for r in group))
        overall = statistics.mean(rates)
        print(f"Llama {condition} Final answer marker rate: q={rates}, overall={overall:.2f}%")
        check.truth(f"Llama {condition} format adherence 96.5-98.5%", 96.5 <= overall <= 98.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="compare all results to frozen paper references")
    parser.add_argument("--section", choices=("all", "retrieval", "outcomes"), default="all")
    args = parser.parse_args()
    check = Checker(args.check)
    if args.section in ("all", "retrieval"):
        retrieval_and_rewrites(check)
    if args.section in ("all", "outcomes"):
        outcomes(check)
    print(f"CHECK RESULT: {'PASS' if not check.failures else 'FAIL (' + ', '.join(check.failures) + ')'}")
    if check.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
