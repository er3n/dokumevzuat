"""
Dense retrieval baseline.
    hatch run dense                          -> e5-large, article corpus (default)
    hatch run dense-bge                      -> BGE-M3, article corpus
    hatch run dense --corpus chunk           -> e5-large, chunk corpus
    hatch run dense-bge --corpus chunk       -> BGE-M3, chunk corpus
    python src/benchmark/dense_baseline.py --model <hf-model-id> [--corpus article|chunk]
Corpus  : VAT Law articles — article (110 articles) or chunk (174 chunks,
          long articles split — article_chunker.py)
Query   : the question from a test ruling (ozelge)
Relevant: madde_atiflar (matched by id)
Metric  : recall@k, precision@k  (k = 1, 3, 5, 10) — directly comparable to bm25_baseline.py
Note    : E5 models want a "query: "/"passage: " prefix; Qwen3-Embedding wants an
          instruction prefix only on the query side (nothing on the passage side);
          Nomic v2 wants "search_query: "/"search_document: " and requires
          trust_remote_code; BGE-M3 needs none of these. Embeddings are L2-normalized
          so cosine similarity = dot product.
          The chunk corpus can have multiple chunks per article; the ranked result
          list is deduped by article_id (best rank kept) before scoring.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from bm25_baseline import (
    load_test,
    recall_at_k,
    precision_at_k,
    dedupe_ranked,
    compute_boilerplate_ids,
    compute_out_of_corpus_ids,
    filter_valid_queries,
    load_rewrite_cache,
    resolve_query,
)

DATA_DIR             = Path(__file__).parent.parent.parent / "data"
ARTICLE_CORPUS_PATH  = DATA_DIR / "processed" / "kdv_maddeler_parsed.jsonl"
CHUNK_CORPUS_PATH    = DATA_DIR / "chunks" / "kdv_maddeler_chunks.jsonl"
VERSIONED_CORPUS_PATH = DATA_DIR / "chunks" / "kdv_maddeler_versioned_chunks.jsonl"
CACHE_DIR            = DATA_DIR / "benchmark" / "embeddings_cache"

CORPUS_PATHS = {
    "article": ARTICLE_CORPUS_PATH,
    "chunk": CHUNK_CORPUS_PATH,
    "versioned": VERSIONED_CORPUS_PATH,
}

KS = [1, 3, 5, 10]

# We explicitly raise max_seq_length for models that support long context; otherwise
# sentence-transformers can silently fall back to a shorter default (usually 512).
MAX_SEQ_LENGTH_OVERRIDES = {
    "bge-m3": 8192,
    "qwen3-embedding-0.6b": 32768,
    "qwen3-embedding-4b": 32768,
    "qwen3-embedding-8b": 32768,
    # gte-multilingual-base's config claims 8192 but its RoPE cache throws an
    # index-out-of-bounds CUDA assert at that length (a known library-version issue);
    # our corpus is already short (<4000 chars per chunk), so 512 is plenty.
    "gte-multilingual-base": 512,
}

# Loading the 8B model in fp32 needs ~32GB VRAM (risky on a 32GB card); force bfloat16.
BF16_MODELS = {"qwen3-embedding-8b"}

# Required for models with custom modeling code, like Nomic/GTE.
TRUST_REMOTE_CODE_MODELS = {"nomic-embed-text-v2-moe", "gte-multilingual-base"}

# flash-attn isn't installed on the RTX 5090, so 8B bf16 is both VRAM-heavy and slow
# (same issue as in cross_encoder_rerank.py). 4-bit quantization (bitsandbytes) shrinks
# weight size by ~4x and gives a real speedup — on by default, disable with
# --no-quantize-4bit.
QUANTIZE_4BIT_MODELS = {"qwen3-embedding-8b"}

QWEN_INSTRUCTION = "Given a Turkish tax law question, retrieve the relevant statute article that answers it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--corpus", choices=["article", "chunk", "versioned"], default="article")
    parser.add_argument("--exclude-boilerplate", action="store_true")
    parser.add_argument("--include-train", action="store_true")
    parser.add_argument("--no-quantize-4bit", action="store_true",
                         help="Forces bf16 even for models in QUANTIZE_4BIT_MODELS")
    parser.add_argument("--query-mode", choices=["original", "rewrite"], default="original")
    parser.add_argument("--rewrite-cache", type=Path, default=None)
    parser.add_argument("--dump-ranked", type=Path, default=None,
                         help="Write the final ranked (deduped, article-level) id list "
                              "for each query to a separate JSON file -- lets the "
                              "protocol be rescored later without rerunning from scratch")
    args = parser.parse_args()
    if args.query_mode == "rewrite" and args.rewrite_cache is None:
        parser.error("--query-mode rewrite requires --rewrite-cache")
    return args


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def is_quantized(model_name: str, no_quantize_4bit: bool) -> bool:
    return model_slug(model_name).lower() in QUANTIZE_4BIT_MODELS and not no_quantize_4bit


def load_dense_model(model_name: str, no_quantize_4bit: bool = False) -> SentenceTransformer:
    slug_lower = model_slug(model_name).lower()
    st_kwargs: dict = {}
    if is_quantized(model_name, no_quantize_4bit):
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        st_kwargs["model_kwargs"] = {"quantization_config": bnb_config}
    elif slug_lower in BF16_MODELS:
        st_kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
    if slug_lower in TRUST_REMOTE_CODE_MODELS:
        st_kwargs["trust_remote_code"] = True

    model = SentenceTransformer(model_name, **st_kwargs)  # auto-selects GPU if available
    print(f"device = {model.device}" + ("  [4-bit quantized]" if is_quantized(model_name, no_quantize_4bit) else ""))
    max_seq_length = MAX_SEQ_LENGTH_OVERRIDES.get(slug_lower)
    if max_seq_length:
        model.max_seq_length = max_seq_length
    print(f"max_seq_length = {model.max_seq_length}")
    return model


def model_family(model_name: str) -> str:
    name = model_name.lower()
    if "e5" in name:
        return "e5"
    if "qwen" in name:
        return "qwen"
    if "nomic" in name:
        return "nomic"
    return "none"  # BGE-M3 etc.: needs no prefix


def cache_slug(model_name: str, quantized: bool = False) -> str:
    return model_slug(model_name) + ("-4bit" if quantized else "")


def out_path(model_name: str, corpus: str, quantized: bool = False) -> Path:
    return DATA_DIR / "benchmark" / f"dense_results_{cache_slug(model_name, quantized)}_{corpus}.json"


def corpus_cache_path(model_name: str, corpus: str, quantized: bool = False) -> Path:
    # 4-bit weights produce different embeddings -> must never mix with the bf16 cache.
    return CACHE_DIR / f"{cache_slug(model_name, quantized)}_kdv_maddeler_{corpus}.npz"


def load_corpus(corpus: str) -> list[dict]:
    path = CORPUS_PATHS[corpus]
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc["article_id"] = doc["id"] if corpus == "article" else doc["article_id"]
            docs.append(doc)
    return docs


def passage_text(doc: dict, model_name: str) -> str:
    text = f"{(doc.get('title') or '')} {(doc.get('metin') or '')}"
    family = model_family(model_name)
    if family == "e5":
        return f"passage: {text}"
    if family == "nomic":
        return f"search_document: {text}"
    return text  # Qwen3-Embedding, BGE-M3: no prefix on the passage side


def query_text(question: str, model_name: str) -> str:
    family = model_family(model_name)
    if family == "e5":
        return f"query: {question}"
    if family == "qwen":
        return f"Instruct: {QWEN_INSTRUCTION}\nQuery: {question}"
    if family == "nomic":
        return f"search_query: {question}"
    return question


def get_corpus_embeddings(
    model: SentenceTransformer, docs: list[dict], model_name: str, corpus: str, quantized: bool = False
) -> np.ndarray:
    ids = [doc["id"] if corpus == "article" else doc["chunk_id"] for doc in docs]
    path = corpus_cache_path(model_name, corpus, quantized)

    if path.exists():
        cached = np.load(path, allow_pickle=True)
        if list(cached["ids"]) == ids:
            print(f"Corpus embeddings loaded from cache: {path}")
            return cached["embeddings"]

    print("Computing corpus embeddings (model downloads on first run)...")
    texts = [passage_text(doc, model_name) for doc in docs]
    embeddings = model.encode(
        texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True
    ).astype("float32")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), embeddings=embeddings)
    print(f"Written to cache: {path}")
    return embeddings


def main() -> None:
    args = parse_args()
    model_name, corpus = args.model, args.corpus
    quantized = is_quantized(model_name, args.no_quantize_4bit)
    print(f"Model: {model_name}  Corpus: {corpus}" + ("  [4-bit quantized]" if quantized else ""))

    corpus_docs = load_corpus(corpus)
    corpus_article_ids = [doc["article_id"] for doc in corpus_docs]

    model = load_dense_model(model_name, args.no_quantize_4bit)
    corpus_embeddings = get_corpus_embeddings(model, corpus_docs, model_name, corpus, quantized)

    test = load_test(include_train=args.include_train)

    boilerplate_ids = compute_boilerplate_ids() if args.exclude_boilerplate else set()
    out_of_corpus_ids = compute_out_of_corpus_ids()
    valid, relevant_ids_list, n_skipped_out_of_corpus = filter_valid_queries(
        test, boilerplate_ids | out_of_corpus_ids
    )

    rewrite_cache = load_rewrite_cache(args.rewrite_cache) if args.query_mode == "rewrite" else {}
    query_texts = [query_text(resolve_query(r, rewrite_cache), model_name) for r in valid]
    query_embeddings = model.encode(
        query_texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True
    ).astype("float32")

    metrics: dict[str, list[float]] = defaultdict(list)
    missed: list[dict] = []  # recall@10 = 0
    per_query_keys: list[str] = []
    per_query_recall_10: list[float] = []
    ranked_dump: dict[str, list[int]] = {}

    for record, relevant_ids, q_emb in zip(valid, relevant_ids_list, query_embeddings):
        sims = corpus_embeddings @ q_emb  # normalized -> cosine similarity
        ranked = sorted(zip(corpus_article_ids, sims), key=lambda x: x[1], reverse=True)
        retrieved_ids = dedupe_ranked([article_id for article_id, _ in ranked])

        for k in KS:
            metrics[f"recall@{k}"].append(recall_at_k(retrieved_ids, relevant_ids, k))
            metrics[f"precision@{k}"].append(precision_at_k(retrieved_ids, relevant_ids, k))

        key = f"{record['ozelgeTarih']}|{record['baslik']}"
        per_query_keys.append(key)
        per_query_recall_10.append(recall_at_k(retrieved_ids, relevant_ids, 10))
        if args.dump_ranked:
            ranked_dump[key] = retrieved_ids

        if recall_at_k(retrieved_ids, relevant_ids, 10) == 0.0:
            missed.append({
                "ozelgeTarih": record["ozelgeTarih"],
                "baslik": record["baslik"],
                "relevant": list(relevant_ids),
                "top5": retrieved_ids[:5],
            })

    n = len(metrics[f"recall@{KS[0]}"])
    results: dict = {
        "model": model_name, "corpus": corpus, "n_queries": n, "scores": {},
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
    results["per_query"] = {"keys": per_query_keys, "recall@10": per_query_recall_10}
    results["query_mode"] = args.query_mode
    if args.exclude_boilerplate:
        results["boilerplate_ids"] = sorted(boilerplate_ids)

    suffix = "_noboilerplate" if args.exclude_boilerplate else ""
    suffix += "_traintest" if args.include_train else ""
    suffix += "_rewrite" if args.query_mode == "rewrite" else ""
    out = out_path(model_name, corpus + suffix, quantized)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if args.dump_ranked:
        args.dump_ranked.parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_ranked, "w", encoding="utf-8") as f:
            json.dump(ranked_dump, f, ensure_ascii=False)
        print(f"Ranked id lists (deduped, article-level): {args.dump_ranked}")

    print(f"Queries with recall@10 = 0: {len(missed)}/{n}")
    if missed:
        print("\nExample failed queries:")
        for m in missed[:3]:
            print(f"  {m['ozelgeTarih']} | {m['baslik'][:60]}")
            print(f"    relevant: {m['relevant']}")

    print(f"\nOutput: {out}")


if __name__ == "__main__":
    main()
