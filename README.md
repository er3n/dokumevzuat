# KDV RAG Benchmark

A retrieval benchmark for Turkish VAT (KDV, Katma Değer Vergisi) law, built by adding
retrieval layers one at a time — chunking, model choice, hybrid search, reranking, query
rewriting, historical/date filtering — and statistically validating each layer on its own.

The differentiator: **point-in-time retrieval**. "Was this transaction VAT-exempt in
2019?" should return the article version that was actually in force on the event date,
not today's text.

| | recall@10 |
|---|---|
| Best general retrieval (BM25∪Nomic pool + query rewriting + Qwen3-Reranker-8B rerank) | **0.738** (hard/clean question subset) |
| Point-in-time, date-blind (current-text approach) | 0.089 |
| Point-in-time, historical data + date filter | **0.768** |

Full writeup, including the layers that didn't help: *(blog link to be added)*.
Dataset: [huggingface.co/datasets/er3nhf/kdv-rag-benchmark](https://huggingface.co/datasets/er3nhf/kdv-rag-benchmark).

## What's here, and what isn't

This repo ships the retrieval/chunking/evaluation code and pulls its data from the
already-published HF dataset above. It does **not** ship a scraper. The source rulings
and law text come from `gib.gov.tr` and `mevzuat.gov.tr`, and scraping either directly is
a legal gray area this repo intentionally stays out of — the dataset is published as
structured, labeled records instead of raw scraped text.

Practically: `hatch run fetch-data` gets you everything the benchmark scripts need.
Nothing here requires re-scraping anything.

## Quickstart

```bash
pip install hatch
hatch run fetch-data       # downloads the dataset into data/
hatch run bm25-chunk       # BM25 baseline, recall@k on the chunked article corpus
hatch run dense-nomic-chunk  # best single dense model
hatch run rerank-qwen8b-pool20-chunk  # cross-encoder rerank
```

Every hatch command in `pyproject.toml` runs one specific experiment from the ledger
above — article vs. chunked corpus, raw vs. boilerplate-excluded scoring, single-model
vs. hybrid vs. reranked, with/without query rewriting, with/without the point-in-time
date filter. `hatch run pipeline` runs a small end-to-end smoke test (fetch data, rebuild
the chunked corpus, run BM25).

## Repo layout

```
src/parsers/
  article_chunker.py       — splits long articles at clause/sentence boundaries
  build_versioned_corpus.py — adds historical article versions to the chunked corpus
  chunk_text.py             — shared text-splitting logic used by the above

src/benchmark/
  bm25_baseline.py, dense_baseline.py, hybrid_rerank.py, cross_encoder_rerank.py
                             — the retrieval methods themselves
  generate_rewrites.py      — LLM query rewriting (daily language -> legal register)
  point_in_time_ground_truth.py, point_in_time_score.py
                             — resolves/scores the version of an article in force on a
                               given date
  bootstrap_compare.py      — paired bootstrap significance testing between two result files

scripts/
  fetch_data.py             — downloads the published dataset into data/
  publish_to_hf.py          — republishes data/ + results to HF (maintainer use)

docker/                     — vLLM serving scripts for the reranker/embedding/rewrite
                               models (see below)

data/                       — not checked in; populated by fetch_data.py
```

## Reproducing a result

Every experiment command writes a `data/benchmark/*_results_*.json` file recording
`per_query` recall/precision at k=1,3,5,10, so any two runs can be compared for
statistical significance with `bootstrap_compare.py`:

```bash
hatch run bm25-chunk-traintest
hatch run dense-nomic-chunk-traintest
hatch run python src/benchmark/bootstrap_compare.py \
  --a data/benchmark/bm25_results_chunk_traintest.json \
  --b data/benchmark/dense_results_nomic-embed-text-v2-moe_chunk_traintest.json \
  --metric recall@10
```

A subset of these result files for the target protocol (VUK-citation-cleaned, train+test
combined) is also published on HF under `data/benchmark_results/`, as the raw records
behind the numbers in the writeup.

## vLLM backend note

On an RTX 5090 (Blackwell/sm_120), no flash-attention wheel is available, so the local
`sentence-transformers` reranker falls back to a slower SDPA/eager path. `docker/`
contains scripts to serve the reranker, first-stage embedding model, and query-rewriting
model via `vllm/vllm-openai:nightly` in Docker instead — roughly 2x faster for the
reranker. `docker/start_all.sh` brings up the reranker + embedding servers;
`cross_encoder_rerank.py --backend vllm` then routes through them.

The first-stage embedding model has its own vLLM server script
(`docker/serve_nomic_embed_vllm.sh`) but it is **not recommended**: in testing, the
embeddings vLLM produces for this model don't match local `sentence-transformers` output
closely enough (cosine similarity ~0.94 instead of ~0.999+), and recall drops
accordingly. It's kept for reference; the default (local) backend should be used for the
first stage. See the comments in that script for the full investigation.

These hardware-specific workarounds were only validated on that one machine — YMMV on
other GPUs, and flash-attention support for sm_120 may have improved since.

## License

Code: [MIT](LICENSE). Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
on HF, sourced from public data of Turkish government institutions (GİB, mevzuat.gov.tr).
