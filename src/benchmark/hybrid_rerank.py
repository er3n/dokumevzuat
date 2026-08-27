"""
BM25 + dense hybrid rerank (Reciprocal Rank Fusion).
Corpus  : chunk corpus (default, article also supported)
Query   : the question from a test ruling (ozelge)
Relevant: madde_atiflar (matched by id)
Metric  : recall@k, precision@k (k=1,3,5,10) — directly comparable to bm25_baseline.py /
          dense_baseline.py
Method  : BM25 and the dense model each independently produce their own ranking,
          deduped to article level; the RRF score
              score(article) = sum_system 1 / (rrf_k + rank_system(article))
          combines them (rank starts at 1). rrf_k=60 is the standard value
          (Cormack, Clarke & Buettcher 2009).
Usage:
    python src/benchmark/hybrid_rerank.py --dense-model nomic-ai/nomic-embed-text-v2-moe --corpus chunk
    python src/benchmark/hybrid_rerank.py --dense-model Qwen/Qwen3-Embedding-8B --corpus chunk --exclude-boilerplate
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

from rank_bm25 import BM25Okapi

from bm25_baseline import (
    load_corpus as bm25_load_corpus,
    load_test,
    tokenize,
    dedupe_ranked,
    recall_at_k,
    precision_at_k,
    compute_boilerplate_ids,
    compute_out_of_corpus_ids,
    filter_valid_queries,
)
import dense_baseline as db

DATA_DIR = Path(__file__).parent.parent.parent / "data"
KS = [1, 3, 5, 10]
RRF_K = 60


def rrf_fuse(rankings: list[list[int]], rrf_k: int) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (rrf_k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-model", default="nomic-ai/nomic-embed-text-v2-moe")
    parser.add_argument("--corpus", choices=["article", "chunk", "versioned"], default="chunk")
    parser.add_argument("--exclude-boilerplate", action="store_true")
    parser.add_argument("--include-train", action="store_true")
    parser.add_argument("--no-quantize-4bit", action="store_true")
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name, corpus, rrf_k = args.dense_model, args.corpus, args.rrf_k
    quantized = db.is_quantized(model_name, args.no_quantize_4bit)
    print(f"BM25 + {model_name}  Corpus: {corpus}  RRF k={rrf_k}" + ("  [4-bit quantized]" if quantized else ""))

    corpus_docs, corpus_tokens = bm25_load_corpus(corpus)
    bm25 = BM25Okapi(corpus_tokens)
    corpus_article_ids = [doc["article_id"] for doc in corpus_docs]

    model = db.load_dense_model(model_name, args.no_quantize_4bit)
    corpus_embeddings = db.get_corpus_embeddings(model, corpus_docs, model_name, corpus, quantized)

    test = load_test(include_train=args.include_train)
    boilerplate_ids = compute_boilerplate_ids() if args.exclude_boilerplate else set()
    out_of_corpus_ids = compute_out_of_corpus_ids()
    valid, relevant_ids_list, n_skipped_out_of_corpus = filter_valid_queries(
        test, boilerplate_ids | out_of_corpus_ids
    )

    query_texts = [db.query_text(r["soru"], model_name) for r in valid]
    query_embeddings = model.encode(
        query_texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True
    ).astype("float32")

    metrics: dict[str, list[float]] = defaultdict(list)
    missed: list[dict] = []  # recall@10 = 0
    per_query_keys: list[str] = []
    per_query_recall_10: list[float] = []

    for record, relevant_ids, q_emb in zip(valid, relevant_ids_list, query_embeddings):
        q_tokens = tokenize(record["soru"])
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_ranked = sorted(zip(corpus_article_ids, bm25_scores), key=lambda x: x[1], reverse=True)
        bm25_ids = dedupe_ranked([aid for aid, _ in bm25_ranked])

        dense_sims = corpus_embeddings @ q_emb
        dense_ranked = sorted(zip(corpus_article_ids, dense_sims), key=lambda x: x[1], reverse=True)
        dense_ids = dedupe_ranked([aid for aid, _ in dense_ranked])

        retrieved_ids = rrf_fuse([bm25_ids, dense_ids], rrf_k)

        for k in KS:
            metrics[f"recall@{k}"].append(recall_at_k(retrieved_ids, relevant_ids, k))
            metrics[f"precision@{k}"].append(precision_at_k(retrieved_ids, relevant_ids, k))

        per_query_keys.append(f"{record['ozelgeTarih']}|{record['baslik']}")
        per_query_recall_10.append(recall_at_k(retrieved_ids, relevant_ids, 10))

        if recall_at_k(retrieved_ids, relevant_ids, 10) == 0.0:
            missed.append({
                "ozelgeTarih": record["ozelgeTarih"],
                "baslik": record["baslik"],
                "relevant": list(relevant_ids),
                "top5": retrieved_ids[:5],
            })

    n = len(metrics[f"recall@{KS[0]}"])
    results: dict = {
        "method": "bm25+dense_rrf",
        "dense_model": model_name,
        "corpus": corpus,
        "rrf_k": rrf_k,
        "n_queries": n,
        "scores": {},
        "quantized_4bit": quantized,
    }

    print(f"\nQuery count: {n}  (skipped for out-of-corpus citations: {n_skipped_out_of_corpus})", end="")
    if args.exclude_boilerplate:
        print(f"  (boilerplate articles: {sorted(boilerplate_ids)})")
    else:
        print()
    print()
    print(f"{'Metric':<15} {'Mean':>10}")
    print("-" * 27)
    for k in KS:
        for metric in ("recall", "precision"):
            key = f"{metric}@{k}"
            avg = sum(metrics[key]) / len(metrics[key])
            results["scores"][key] = round(avg, 4)
            print(f"{key:<15} {avg:>10.4f}")
        print()

    results["missed_top10"] = missed[:10]
    results["exclude_boilerplate"] = args.exclude_boilerplate
    results["include_train"] = args.include_train
    results["out_of_corpus_ids"] = sorted(out_of_corpus_ids)
    results["n_skipped_out_of_corpus"] = n_skipped_out_of_corpus
    if args.exclude_boilerplate:
        results["boilerplate_ids"] = sorted(boilerplate_ids)
    results["per_query"] = {"keys": per_query_keys, "recall@10": per_query_recall_10}

    suffix = "_noboilerplate" if args.exclude_boilerplate else ""
    suffix += "_traintest" if args.include_train else ""
    out = DATA_DIR / "benchmark" / f"hybrid_results_{db.cache_slug(model_name, quantized)}_{corpus}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Queries with recall@10 = 0: {len(missed)}/{n}")
    if missed:
        print("\nExample failed queries:")
        for m in missed[:3]:
            print(f"  {m['ozelgeTarih']} | {m['baslik'][:60]}")
            print(f"    relevant: {m['relevant']}")

    print(f"\nOutput: {out}")


if __name__ == "__main__":
    main()
