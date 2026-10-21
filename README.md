# Multi-query RAG poisoning reproduction artifact

This double-blind artifact reproduces the reported aggregate numbers offline from 100 HotpotQA targets. It contains target-level outputs, the seven-rewrite pool, clean and poisoned per-view top-10 document IDs with cosine scores, and gold supporting-document IDs. No language model, encoder, corpus, or FAISS index is needed for the numerical checks.

## Run

```bash
python -m pip install -r requirements.txt
python analysis/reproduce_all.py --check
```

The command prints every numerical comparison and exits 0 only when all checks pass. It accepts `--section retrieval` and `--section outcomes` for focused reruns. Percentage comparisons use a maximum deviation of 0.05 percentage points; four-decimal Jaccard checks use 0.00005. Table II reference cells are rounded to two decimal places. The complete successful output is saved in `analysis/check_output.txt`.

## Paper value to command map

| Paper values | Source data | Reproduction command | Output label |
|---|---|---|---|
| Table I: exposure 100% at all q, median rank 1, rank-1 85/82/83/83%, top-5 100% | `data/retrieval_views.jsonl` | `python analysis/reproduce_all.py --check --section retrieval` | `Table I` |
| Table I corrected ASR 20.8/20.0/17.7/17.7% | `data/multiquery_rag_target_results.csv` | `python analysis/reproduce_all.py --check --section outcomes` | `Table I corrected ASR` |
| Rewrite Jaccard 0.6161 between rewrites, 0.6434 against original | `data/rewrites.jsonl` | `python analysis/reproduce_all.py --check --section retrieval` | `lexical Jaccard` |
| Top-10 overlap Jaccard 0.7266 poisoned, 0.7032 clean | `data/retrieval_views.jsonl` | `python analysis/reproduce_all.py --check --section retrieval` | `retrieval top10 Jaccard` |
| Gold candidate union 1.78 to 1.81; gold in top-5 1.70/1.66/1.70/1.69 | `data/retrieval_views.jsonl`, `data/gold_documents.jsonl` | `python analysis/reproduce_all.py --check --section retrieval` | `gold candidate`, `gold top5` |
| q=2 and q=4 subset rank changes in 11 and 12 targets; top-5 exposure 100% | `data/retrieval_views.jsonl` | `python analysis/reproduce_all.py --check --section retrieval` | `subset` |
| MaxScore and AvgScore: median rank 1, top-5 exposure 100% | `data/retrieval_views.jsonl` | `python analysis/reproduce_all.py --check --section retrieval` | `MaxScore`, `AvgScore` |
| Cochran Q, df=3: Qwen C1 1.60, p=.659, n=95; Llama C1 3.00, p=.392, n=95 | `data/multiquery_rag_target_results.csv` | `python analysis/reproduce_all.py --check --section outcomes` | `Cochran` |
| Paired q=1 minus q=8 bootstrap upper bounds: Qwen C1/C3 7.4/4.2 pp, Llama C1/C3 2.1/8.4 pp | `data/multiquery_rag_target_results.csv` | `python analysis/reproduce_all.py --check --section outcomes` | `percentile bootstrap` |
| McNemar exact and Holm: Qwen C2 vs C1 and C4 vs C3 significant; Llama comparisons nonsignificant | `data/multiquery_rag_target_results.csv` | `python analysis/reproduce_all.py --check --section outcomes` | `McNemar` |
| Table II: all 16 q-by-condition cells per generator, each with ASR, Jeffreys interval, abstention, EM and F1 | `data/multiquery_rag_target_results.csv`, `data/target_questions_with_answers.csv`, `data/table_ii_reference.json` | `python analysis/reproduce_all.py --check --section outcomes` | `Table II` |
| Qwen C2/C4, Llama C2/C4 abstention bands; Llama consensus format 96.5–98.5% | `data/multiquery_rag_target_results.csv` | `python analysis/reproduce_all.py --check --section outcomes` | `abstention range`, `format adherence` |

## Data and statistical definitions

`retrieval_views.jsonl` has one row per target and retrieval treatment. Each row holds eight views in fixed order; each view contains ten `[document_id, raw_cosine_score]` pairs. q=1,2,4,8 are prefixes of the same eight views. The compact file was extracted from 100 frozen retrieval records and checked against all 800 original target×q×treatment records. `rewrites.jsonl` holds the original question and seven rewrite texts. `gold_documents.jsonl` contains the two identifiable supporting passage IDs for each target. The 3,200-row result CSV and the question/answer CSV are direct copies of the frozen exports. `analysis/prepare_data.py` documents the one-time, source-read-only conversion and can be rerun with a source directory argument; it writes only inside this artifact.

For q=1, retrieval keeps the first view's order. For q>1, reciprocal rank fusion sums `1/(60+rank)` over views, with ascending document ID as the tie break. Subset sensitivity fixes the original view and enumerates all seven single rewrites at q=2 and all 35 triples at q=4. MaxScore takes the maximum observed cosine score. AvgScore sums view scores, assigns zero for an absent document, then divides by q. All three fusion methods rank the same candidate union.

Lexical overlap uses case-folded Unicode word sets and Jaccard intersection/union. Retrieval overlap uses unordered top-10 document ID sets across all 28 pairs of eight views; means are first taken per target, then over 100 targets. Gold counts use document IDs rather than answer strings.

Corrected ASR excludes a target when its corresponding clean control hits the attacker target; the `clean_hit` flag is in the exported row. Jeffreys 95% intervals use `Beta(successes+0.5, eligible-successes+0.5)`. Cochran's Q uses only targets eligible at all four q values. McNemar tests use targets eligible in both compared conditions at each q, with exact two-sided binomial p values and Holm correction within each model/comparison family across four q values. Paper floors 0.18 and 0.65 refer to p values rounded to two decimals; the exact minima are 0.1796875 and 0.647606.

The paired q=1 minus q=8 95% intervals use 10,000 target-level percentile bootstrap draws, NumPy legacy `RandomState(42)`, separately restarted for each contrast. This detail matters at a discrete quantile: `default_rng(42)` gives a different Qwen C3 upper endpoint. The printed bound is rounded to one decimal place in the paper.

EM/F1 use all 100 targets, count abstentions as zero, and score only text after the last case-insensitive `Final answer` marker. If absent, the full response is used. Lowercase normalization removes ASCII punctuation and retains articles. Token F1 uses multiset overlap. Llama format adherence is the fraction of consensus responses containing a `Final answer` marker.

`data/table_ii_reference.json` freezes two-decimal values calculated from the supplied target-level export under the stated Table II rules. A manuscript Table II file was not supplied, so this file is a regression reference for the reproducible table, rather than an independently checked copy of manuscript cells. The separately specified paper figures above are checked against their literal constants in `reproduce_all.py`.

## Corpus and pipeline provenance

The frozen retrieval base is a shared 100,000-paragraph corpus built from the HotpotQA `distractor` configuration: 100 selected validation questions contribute their supporting passages, then train split context paragraphs fill the remaining unique document budget. The dataset revision was `1908d6afbbead072334abe2965f91bd2709910ab`; the source is [HotpotQA on Hugging Face](https://huggingface.co/datasets/hotpotqa/hotpot_qa). The corpus builder and index code are under `pipeline/src/core/`. The original build sequence is corpus construction, query expansion, embedding/index construction, and retrieval. The encoder is `BAAI/bge-base-en-v1.5` with the query instruction `Represent this sentence for searching relevant passages: `, CLS pooling, L2 normalization, and exact FAISS `IndexFlatIP` search. The original generation models were Qwen3-8B and Llama-3.1-8B-Instruct. The manuscript analysis runs entirely from the compact data supplied here; the full corpus, index, and weights are intentionally omitted.

`pipeline/` contains sanitized copies of construction, retrieval, generation, and scoring code for inspection. It is not a runnable end-to-end package without the original external data, weights, configs, and full experiment environment. `analysis/prepare_pipeline.py` documents the read-only source copy and path redaction.

## License

The repository code is provided with the MIT license placeholder in `LICENSE`. HotpotQA data and model licenses remain governed by their respective sources; this artifact includes only the selected questions, answer labels, IDs, retrieval scores, rewrites, and model outputs needed for reproduction.
