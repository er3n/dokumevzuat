---
license: cc-by-4.0
language:
- tr
tags:
- rag
- retrieval
- turkish
- tax-law
- legal
- benchmark
- point-in-time
size_categories:
- 1K<n<10K
task_categories:
- question-answering
- text-retrieval
configs:
- config_name: benchmark
  data_files:
  - split: train
    path: data/benchmark/train.jsonl
  - split: test
    path: data/benchmark/test.jsonl
- config_name: corpus
  data_files:
  - split: train
    path: data/corpus/kdv_maddeler.jsonl
- config_name: corpus_chunked
  data_files:
  - split: train
    path: data/corpus_chunked/kdv_maddeler_chunks.jsonl
- config_name: corpus_versioned
  data_files:
  - split: train
    path: data/corpus_versioned/kdv_maddeler_versiyonlu.jsonl
- config_name: corpus_versioned_chunked
  data_files:
  - split: train
    path: data/corpus_versioned_chunked/kdv_maddeler_versioned_chunks.jsonl
---

# KDV RAG Benchmark

A retrieval benchmark dataset for Turkish VAT (KDV, Katma Değer Vergisi) law — built by adding retrieval layers one at a time (chunking, model choice, hybrid search, reranking, query rewriting, historical/date filtering) and statistically validating each one individually (see Results).

## Dataset structure

### Splits

| Split | Records | Period |
|-------|-------|--------|
| `train` | 728 | 2018-2023 |
| `test` | 154 | 2024-2026 |

Split strategy: **temporal** — train and test come from different time windows, minimizing data leakage risk.

> ⚠️ In some evaluation runs (see scores below), the train split is folded into eval as well to increase statistical power (154→813 queries) — this is valid **only for zero-shot/pretrained model comparisons**, since no model here was fine-tuned on train. If you fine-tune your own model on train, evaluate only on the original `test` split.

### Corpus

Provided at four granularities:

| Config | Records | Content |
|--------|-------|---------|
| `corpus` | 110 | Articles of the VAT Law (Law No. 3065), unsplit |
| `corpus_chunked` | 174 | Same articles, long ones split into sub-sections — noticeably better retrieval results |
| `corpus_versioned` | 88 | Real multi-version history for 8 articles, article-level |
| `corpus_versioned_chunked` | 456 | `corpus_chunked` + the same 8 articles' historical versions also chunked — the corpus used for point-in-time eval |

### Features

**train / test:**
```
id             : int      — GIB ruling ID
siteLink       : string   — source URL (gib.gov.tr)
ozelgeNo       : string   — official ruling number
ozelgeTarih    : string   — publication date (YYYY-MM-DD)
baslik         : string   — ruling title
kanunNo        : string   — 3065
soru           : string   — the taxpayer's question
cevap          : string   — GIB's answer
madde_atiflar  : list     — cited articles [{id, title}]
```

**corpus:**
```
id             : int      — article ID
title          : string   — "Madde 17 Sosyal ve Askeri Amaçlı İstisnalarla..."
metin          : string   — full article text
siteLink       : string   — source URL
priority       : int      — article order
bolum          : string   — section title
```

**corpus_chunked:**
```
chunk_id       : string   — "MADDE_<article_id>_<part>"
article_id     : int      — the article id in corpus (ground truth matches on this)
title          : string   — article title
section        : string?  — sub-section title (if any)
part           : int      — order within the article
metin          : string   — chunk text
metin_len      : int      — character length
```

**corpus_versioned:**
```
madde_id       : int      — the article id in corpus (same space as article_id)
madde_no       : string   — the law's article number ("13")
version        : int      — version order (1 = oldest)
metin          : string   — the full article text for that version
valid_from     : string?  — date this version took effect (null = in force since the law's start)
valid_until    : string?  — date this version ended (null = still in force)
kaynak_edit_key: string?  — the edit record that triggered the transition to this version
```

**corpus_versioned_chunked:** `corpus_chunked` fields + `version`, `valid_from`, `valid_until` (as above).

## Results

| | recall@10 |
|---|---|
| Best general retrieval (BM25∪Nomic pool + query rewriting + Qwen3-Reranker-8B rerank) | **0.737** (the 270 real-subject questions) |
| Point-in-time¹, date-blind (current-text approach) | 0.089 |
| Point-in-time¹, historical data + date filter | **0.768** |

¹ Point-in-time: the ability to answer a question like "was this transaction VAT-exempt in 2019?" with the article version that was actually in force on the event date, not today's text — `corpus_versioned`/`corpus_versioned_chunked` exist for this.

The path to these numbers — chunking mattering more than model choice, the MTEB leaderboard not transferring to this task, the risks of drawing conclusions from a small sample, and how much the date filter's position in the pipeline (before vs. after ranking) changes the result — is written up in a separate post: [dokukoza.com/blog/measured-legal-rag](https://dokukoza.com/blog/measured-legal-rag/). Raw experiment records and code: [github.com/er3n/dokumevzuat](https://github.com/er3n/dokumevzuat).

## Usage

```python
from datasets import load_dataset

# Benchmark
train = load_dataset("dokukoza/kdv-rag-benchmark", "benchmark", split="train")
test  = load_dataset("dokukoza/kdv-rag-benchmark", "benchmark", split="test")

# Corpus (unsplit, 110 articles)
corpus = load_dataset("dokukoza/kdv-rag-benchmark", "corpus", split="train")

# Corpus (chunked, 174 records — better retrieval results)
corpus_chunked = load_dataset("dokukoza/kdv-rag-benchmark", "corpus_chunked", split="train")

# Point-in-time article versions (8 multi-version articles)
corpus_versioned = load_dataset("dokukoza/kdv-rag-benchmark", "corpus_versioned", split="train")

# Point-in-time article versions, chunked — the corpus actually used for eval
corpus_versioned_chunked = load_dataset("dokukoza/kdv-rag-benchmark", "corpus_versioned_chunked", split="train")
```

### Evaluation example

```python
# For each test record:
# query  = record["soru"]
# ground_truth = {a["id"] for a in record["madde_atiflar"]}
# retrieved = retrieval_system(query, corpus, top_k=10)
# recall@10 = len(ground_truth & set(retrieved[:10])) / len(ground_truth)
```

## Data source

- **Rulings**: [Revenue Administration (GİB)](https://gib.gov.tr) — VAT Law ruling database
- **Articles**: [mevzuat.gov.tr](https://www.mevzuat.gov.tr) — Law No. 3065 (VAT Law)

Public institutions' open data — full text included, not just references and labels.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — usable with attribution.

## Citation

```bibtex
@misc{kdv_rag_benchmark_2026,
  title        = {KDV RAG Benchmark: A Point-in-Time Turkish VAT Law Retrieval Benchmark},
  author       = {Öztürk, Eren},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/dokukoza/kdv-rag-benchmark}}
}
```
