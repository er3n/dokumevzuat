"""
Pointwise cross-encoder rerank.
Corpus  : chunk corpus (174 chunks, produced by the article chunking script)
Query   : the ruling's question text
Relevant: madde_atiflar (matched by id)
Metric  : recall@k, precision@k (k=1,3,5,10) — directly comparable to bm25_baseline.py /
          dense_baseline.py / hybrid_rerank.py
Method  : Pull the first-stage (Nomic v2-moe standalone dense) chunk-level top-k candidate
          pool (no article-level dedupe at first-stage — multiple chunks from the same
          article can both be candidates). The cross-encoder scores each query+chunk pair
          jointly, the pool is re-ranked, and ONLY THEN deduped to article_id for scoring.
          The pool is a fixed top-k (20/50), not "rerank the whole corpus with no
          first-stage" — skipping first-stage just because the corpus is small today
          (174 chunks) would measure an artificial ceiling that's unreachable in production
          (corpus ~1M).
--hybrid-pool: builds the first-stage pool from BM25 top-k ∪ Nomic top-k instead of a
          single source (Nomic top-k) — otherwise the cross-encoder is at the mercy of
          Nomic's first-stage blind spots,
          whereas hybrid RRF doesn't miss, because it merges BM25's and Nomic's independent
          full rankings). Pool size is no longer fixed (up to 2x pool-size, depending on
          overlap) — both sources' proven top-k are merged without being trimmed.
--backend vllm: routes second-stage scoring through a vLLM server's /rerank endpoint
          (JinaAI-compatible, query+documents) over HTTP instead of the local
          sentence-transformers CrossEncoder — an alternative to the slowdown caused by
          falling back to local SDPA/eager on RTX 5090/sm_120 (no flash-attn available).
          The server (docker/serve_qwen3_reranker_vllm.sh) must be started separately;
          this script is only the client side. --vllm-concurrency sends queries
          concurrently so vLLM's continuous batching actually kicks in (sequential
          one-at-a-time requests never trigger it). Results are written to a separate file
          (slug gets a "-vllm" suffix), never overwriting local results. Verified:
          recall@10 matches the local result exactly (0.6566, n=151), ~1.1-1.2s/query —
          recommended.
--rewrite-pool: adds the Nomic top-k for the LLM-rewritten, closer-to-legal-language query
          (output of generate_rewrites.py, supplied via --rewrite-cache) to the first-stage
          pool alongside the original query; combined with --hybrid-pool, also adds its
          BM25 top-k (a four-source pool: Nomic_orig ∪ BM25_orig ∪ Nomic_rewrite ∪
          BM25_rewrite). The reranker is ALWAYS given the original query — the rewrite is
          only used to widen first-stage candidate diversity, so a rewrite hallucination
          can't leak into the reranker's decision, since ground truth is labeled against
          the original question's intent.
--first-stage-backend vllm: also pulls first-stage (Nomic v2-moe) embeddings from vLLM
          (/v1/embeddings). WARNING — NOT RECOMMENDED: vLLM's embeddings don't match local
          sentence-transformers (for the same corpus documents, mean cosine similarity is
          ~0.94, min 0.82 — ~0.999+ would be expected at the same dtype), and it drops
          recall@10 from 0.6566 to 0.5206. fp16 and bf16 give the SAME result, so it isn't
          a dtype/precision issue — most likely an inconsistency in vLLM's NomicMoE
          (bert_with_rope.py) reimplementation. Root cause was not investigated (first-stage
          isn't the bottleneck anyway, it finishes in ~1s from local cache). This code path
          stays for reference; the default (local) should be used for production/comparison.
          See docker/serve_nomic_embed_vllm.sh.
Process : first-stage (dense) and second-stage (cross-encoder, when backend=local) run in
          SEPARATE processes (--stage pool / --stage rerank), communicating through an
          intermediate JSON file. Unloading one model from the GPU and immediately loading
          the next in the same process (del + empty_cache) caused a segfault on WSL2/CUDA —
          each stage now starts in its own clean CUDA context. The default --stage all
          launches these two processes in sequence itself. With backend=vllm the rerank
          stage never loads a local GPU model, so there's no segfault risk, but the same
          two-process flow is kept for consistency.
Usage:
    python src/benchmark/cross_encoder_rerank.py --reranker-model BAAI/bge-reranker-v2-m3 --pool-size 20
    python src/benchmark/cross_encoder_rerank.py --reranker-model Qwen/Qwen3-Reranker-8B --pool-size 50 --exclude-boilerplate
    python src/benchmark/cross_encoder_rerank.py --reranker-model Qwen/Qwen3-Reranker-8B --pool-size 20 --backend vllm --vllm-url http://localhost:8000
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from bm25_baseline import (
    load_test,
    tokenize,
    dedupe_ranked,
    recall_at_k,
    precision_at_k,
    compute_boilerplate_ids,
    compute_out_of_corpus_ids,
    filter_valid_queries,
    load_rewrite_cache,
    resolve_query,
)
import dense_baseline as db

DATA_DIR = db.DATA_DIR
TMP_DIR = DATA_DIR / "benchmark" / "_tmp_pool"
KS = [1, 3, 5, 10]

# The bge-reranker-v2-m3 base model (bge-m3) supports 8192, but the longest chunk in the
# corpus is ~7500 characters (p95 ~2924) — this practically eliminates any risk of
# 2048-token truncation.
# qwen3-reranker-8b: the actual chunk length measured with the Qwen tokenizer maxes out at
# 2558 tokens (p95=1057, p99=1331) — the 8192 default is needlessly large: flash-attention
# isn't installed (eager/SDPA fallback, O(n^2) memory), VRAM is nearly maxed out at 8192
# (32000/32607 MiB), and a single query balloons to 20+ minutes. 3072 (including room for
# query+chunk+instruction template) restores normal VRAM/speed without truncating any
# content.
RERANKER_MAX_LENGTH = {
    "bge-reranker-v2-m3": 2048,
    "qwen3-reranker-8b": 3072,
    "qwen3-reranker-4b": 3072,
    "qwen3-reranker-0.6b": 3072,
}

# Loading the 8B model in fp32 carries VRAM risk (same reasoning as BF16_MODELS in dense_baseline.py).
RERANKER_BF16_MODELS = {"qwen3-reranker-8b", "qwen3-reranker-4b", "qwen3-reranker-0.6b"}

# 4-bit quantization (bitsandbytes): the real benefit isn't freeing up VRAM, it's reducing
# the memory-bandwidth bottleneck — confirmed experimentally: bf16 4B (~8GB weights) came
# out SLOWER than quantized 8B (~4GB weights, 4-bit), because inference is mostly bound by
# how fast weights can be read from memory (not compute). This gain is independent of model
# size — that's why we also quantize 4B/0.6B.
RERANKER_QUANTIZE_4BIT_MODELS = {"qwen3-reranker-8b", "qwen3-reranker-4b", "qwen3-reranker-0.6b"}


def reranker_is_quantized(model_name: str, no_quantize_4bit: bool) -> bool:
    return reranker_slug(model_name).lower() in RERANKER_QUANTIZE_4BIT_MODELS and not no_quantize_4bit

# On RTX 5090 (Blackwell/sm_120), the attention kernels in the current torch version
# aren't optimized yet — even with attn_implementation="sdpa" requested, it can fall back
# to the "math" backend (O(n^2) memory). This cost scales with batch_size × sequence_length^2
# and is INDEPENDENT OF WEIGHT QUANTIZATION (quantization only reduces weight memory, not
# the attention computation) — a batch=48 trial on 4B confirmed this: it pushed VRAM to a
# risky level (28/32.6GB) without any speed gain. So even when quantized, we don't push the
# batch size aggressively — we only scale it up modestly based on the smaller model's lower
# weight footprint.
RERANKER_DEFAULT_BATCH_SIZE = {
    "qwen3-reranker-8b": 4,
    "qwen3-reranker-4b": 8,
    "qwen3-reranker-0.6b": 16,
}


def reranker_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reranker-model", required=True)
    parser.add_argument("--first-stage-model", default="nomic-ai/nomic-embed-text-v2-moe")
    parser.add_argument("--corpus", choices=["article", "chunk", "versioned"], default="chunk")
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--hybrid-pool", action="store_true")
    parser.add_argument("--rewrite-pool", action="store_true",
                         help="Also adds Nomic (and, combined with --hybrid-pool, BM25) results "
                              "for the rewritten query to the first-stage pool")
    parser.add_argument("--rewrite-cache", type=Path, default=None)
    parser.add_argument("--exclude-boilerplate", action="store_true")
    parser.add_argument("--include-train", action="store_true")
    parser.add_argument("--no-quantize-4bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--stage", choices=["all", "pool", "rerank"], default="all")
    parser.add_argument("--backend", choices=["local", "vllm"], default="local")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--vllm-model", default=None, help="--served-model-name passed to the server; defaults to --reranker-model")
    parser.add_argument("--vllm-concurrency", type=int, default=8)
    parser.add_argument("--first-stage-backend", choices=["local", "vllm"], default="local")
    parser.add_argument("--first-stage-vllm-url", default="http://localhost:8001")
    parser.add_argument("--first-stage-vllm-model", default=None, help="defaults to --first-stage-model")
    parser.add_argument("--date-filter", action="store_true",
                         help="Point-in-time Variant C: with --corpus versioned, filters the pool by "
                              "each query's ozelgeTarih against the chunk's valid_from/valid_until window")
    parser.add_argument("--dump-ranked", type=Path, default=None,
                         help="Writes the final ranked chunk_id list per query (no dedupe, not "
                              "article-level) to a separate JSON -- for scoring against custom "
                              "ground truth later, e.g. point-in-time scenarios")
    args = parser.parse_args()
    if args.rewrite_pool and args.rewrite_cache is None:
        parser.error("--rewrite-pool requires --rewrite-cache")
    if args.date_filter and args.corpus != "versioned":
        parser.error("--date-filter is only meaningful with --corpus versioned")
    if args.batch_size is None:
        args.batch_size = RERANKER_DEFAULT_BATCH_SIZE.get(reranker_slug(args.reranker_model).lower(), 16)
    if args.vllm_model is None:
        args.vllm_model = args.reranker_model
    if args.first_stage_vllm_model is None:
        args.first_stage_vllm_model = args.first_stage_model
    return args


def reranker_cache_slug(model_name: str, quantized: bool, backend: str = "local") -> str:
    slug = reranker_slug(model_name) + ("-4bit" if quantized else "")
    if backend == "vllm":
        slug += "-vllm"
    return slug


def out_path(args: argparse.Namespace, corpus: str) -> Path:
    quantized = args.backend == "local" and reranker_is_quantized(args.reranker_model, args.no_quantize_4bit)
    slug = reranker_cache_slug(args.reranker_model, quantized, args.backend)
    return DATA_DIR / "benchmark" / f"rerank_results_{slug}_pool{args.pool_size}_{corpus}.json"


def pool_path(args: argparse.Namespace) -> Path:
    suffix = "_noboilerplate" if args.exclude_boilerplate else ""
    suffix += "_traintest" if args.include_train else ""
    suffix += "_hybridpool" if args.hybrid_pool else ""
    suffix += "_rewritepool" if args.rewrite_pool else ""
    suffix += "_datefilter" if args.date_filter else ""
    return TMP_DIR / f"pool_{db.model_slug(args.first_stage_model)}_{args.corpus}_top{args.pool_size}{suffix}.json"


def chunk_text(doc: dict) -> str:
    return f"{(doc.get('title') or '')} {(doc.get('metin') or '')}"


def valid_at(doc: dict, ref_date: str) -> bool:
    """--date-filter (point-in-time Variant C): does the chunk's valid_from/valid_until
    (from kdv_maddeler_versioned_chunks.jsonl, None = +/-infinity) cover ref_date."""
    vf, vu = doc.get("valid_from"), doc.get("valid_until")
    if vf is not None and ref_date < vf:
        return False
    if vu is not None and ref_date >= vu:
        return False
    return True


def valid_mask(corpus_docs: list[dict], ref_date: str) -> np.ndarray:
    """--date-filter: boolean mask the same length as corpus_docs, in order. For
    PRE-filtering — used to push invalid candidates to -inf BEFORE ranking, otherwise
    different-dated versions of the same article can eliminate each other for the top-k
    and the correct version may never make it into the pool at all (post-filtering, i.e.
    eliminating AFTER the pool is built, can't fix this — it can't bring back a candidate
    that's already been excluded)."""
    return np.array([valid_at(doc, ref_date) for doc in corpus_docs])


def masked_scores(scores: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    return scores if mask is None else np.where(mask, scores, -np.inf)


def run_subprocess(stage: str, args: argparse.Namespace) -> None:
    argv = [
        sys.executable, __file__,
        "--reranker-model", args.reranker_model,
        "--first-stage-model", args.first_stage_model,
        "--corpus", args.corpus,
        "--pool-size", str(args.pool_size),
        "--batch-size", str(args.batch_size),
        "--stage", stage,
        "--backend", args.backend,
        "--vllm-url", args.vllm_url,
        "--vllm-model", args.vllm_model,
        "--vllm-concurrency", str(args.vllm_concurrency),
        "--first-stage-backend", args.first_stage_backend,
        "--first-stage-vllm-url", args.first_stage_vllm_url,
        "--first-stage-vllm-model", args.first_stage_vllm_model,
    ]
    if args.hybrid_pool:
        argv.append("--hybrid-pool")
    if args.rewrite_pool:
        argv.extend(["--rewrite-pool", "--rewrite-cache", str(args.rewrite_cache)])
    if args.exclude_boilerplate:
        argv.append("--exclude-boilerplate")
    if args.include_train:
        argv.append("--include-train")
    if args.no_quantize_4bit:
        argv.append("--no-quantize-4bit")
    if args.date_filter:
        argv.append("--date-filter")
    if stage == "rerank" and args.dump_ranked:
        argv.extend(["--dump-ranked", str(args.dump_ranked)])
    subprocess.run(argv, check=True)


def vllm_embed(session: requests.Session, url: str, model: str, texts: list, batch_size: int = 32) -> np.ndarray:
    """Embeds texts via vLLM's /v1/embeddings, L2-normalizing before returning
    (client-side normalization so we don't have to trust whether the server already
    normalized). truncate_prompt_tokens=-1: texts exceeding max-model-len are rejected by
    vLLM with a 400 by default (local sentence-transformers silently truncates to
    max_seq_length instead) — we explicitly request truncation to match local behavior."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = session.post(
            f"{url}/v1/embeddings",
            json={"model": model, "input": batch, "truncate_prompt_tokens": -1},
            timeout=300,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        vectors.extend(d["embedding"] for d in data)
    embeddings = np.array(vectors, dtype="float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def get_corpus_embeddings_vllm(
    session: requests.Session, url: str, model: str, docs: list, model_name: str, corpus: str
) -> np.ndarray:
    """Same .npz cache pattern as dense_baseline.get_corpus_embeddings, in a separate file
    with a '-vllm' suffix (so it never collides with the locally-computed cache)."""
    id_field = "id" if corpus == "article" else "chunk_id"
    ids = [doc[id_field] for doc in docs]
    path = db.CACHE_DIR / f"{db.model_slug(model_name)}-vllm_kdv_maddeler_{corpus}.npz"

    if path.exists():
        cached = np.load(path, allow_pickle=True)
        if list(cached["ids"]) == ids:
            print(f"[pool] Corpus embeddings loaded from cache (vLLM): {path}")
            return cached["embeddings"]

    print("[pool] Computing corpus embeddings via vLLM...")
    texts = [db.passage_text(doc, model_name) for doc in docs]
    embeddings = vllm_embed(session, url, model, texts)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), embeddings=embeddings)
    print(f"[pool] Written to cache: {path}")
    return embeddings


# --- Stage 1: first-stage dense retrieval, build the chunk-level top-k candidate pool ---
def run_pool_stage(args: argparse.Namespace) -> None:
    print(f"[pool] First-stage: {args.first_stage_model} (pool={args.pool_size})  Corpus: {args.corpus}"
          f"  Backend: {args.first_stage_backend}")

    corpus_docs = db.load_corpus(args.corpus)
    id_field = "id" if args.corpus == "article" else "chunk_id"
    corpus_ids = [doc[id_field] for doc in corpus_docs]
    corpus_by_id = {doc[id_field]: doc for doc in corpus_docs}

    bm25 = None
    if args.hybrid_pool:
        corpus_tokens = [tokenize(chunk_text(doc)) for doc in corpus_docs]
        bm25 = BM25Okapi(corpus_tokens)

    test = load_test(include_train=args.include_train)
    boilerplate_ids = compute_boilerplate_ids() if args.exclude_boilerplate else set()
    out_of_corpus_ids = compute_out_of_corpus_ids()
    valid, relevant_ids_list, n_skipped_out_of_corpus = filter_valid_queries(
        test, boilerplate_ids | out_of_corpus_ids
    )
    rewrite_cache = load_rewrite_cache(args.rewrite_cache) if args.rewrite_pool else {}
    rewrite_texts = [resolve_query(r, rewrite_cache) for r in valid] if args.rewrite_pool else None

    if args.first_stage_backend == "vllm":
        session = requests.Session()
        corpus_embeddings = get_corpus_embeddings_vllm(
            session, args.first_stage_vllm_url, args.first_stage_vllm_model,
            corpus_docs, args.first_stage_model, args.corpus,
        )
        query_texts = [db.query_text(r["soru"], args.first_stage_model) for r in valid]
        query_embeddings = vllm_embed(session, args.first_stage_vllm_url, args.first_stage_vllm_model, query_texts)
        if args.rewrite_pool:
            rewrite_query_texts = [db.query_text(t, args.first_stage_model) for t in rewrite_texts]
            rewrite_query_embeddings = vllm_embed(
                session, args.first_stage_vllm_url, args.first_stage_vllm_model, rewrite_query_texts
            )
    else:
        first_stage_quantized = db.is_quantized(args.first_stage_model, args.no_quantize_4bit)
        model = db.load_dense_model(args.first_stage_model, args.no_quantize_4bit)
        corpus_embeddings = db.get_corpus_embeddings(
            model, corpus_docs, args.first_stage_model, args.corpus, first_stage_quantized
        )
        query_texts = [db.query_text(r["soru"], args.first_stage_model) for r in valid]
        query_embeddings = model.encode(
            query_texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True
        ).astype("float32")
        if args.rewrite_pool:
            rewrite_query_texts = [db.query_text(t, args.first_stage_model) for t in rewrite_texts]
            rewrite_query_embeddings = model.encode(
                rewrite_query_texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True
            ).astype("float32")

    queries = []
    for i, (record, relevant_ids, q_emb) in enumerate(zip(valid, relevant_ids_list, query_embeddings)):
        mask = valid_mask(corpus_docs, record["ozelgeTarih"]) if args.date_filter else None

        sims = masked_scores(corpus_embeddings @ q_emb, mask)
        ranked = sorted(zip(corpus_ids, sims), key=lambda x: x[1], reverse=True)
        pool_ids = [cid for cid, _ in ranked[: args.pool_size]]

        if bm25 is not None:
            bm25_scores = masked_scores(bm25.get_scores(tokenize(record["soru"])), mask)
            bm25_ranked = sorted(zip(corpus_ids, bm25_scores), key=lambda x: x[1], reverse=True)
            bm25_pool_ids = [cid for cid, _ in bm25_ranked[: args.pool_size]]
            pool_ids = dedupe_ranked(pool_ids + bm25_pool_ids)

        if args.rewrite_pool:
            rw_sims = masked_scores(corpus_embeddings @ rewrite_query_embeddings[i], mask)
            rw_ranked = sorted(zip(corpus_ids, rw_sims), key=lambda x: x[1], reverse=True)
            pool_ids = dedupe_ranked(pool_ids + [cid for cid, _ in rw_ranked[: args.pool_size]])

            if bm25 is not None:
                rw_bm25_scores = masked_scores(bm25.get_scores(tokenize(rewrite_texts[i])), mask)
                rw_bm25_ranked = sorted(zip(corpus_ids, rw_bm25_scores), key=lambda x: x[1], reverse=True)
                pool_ids = dedupe_ranked(pool_ids + [cid for cid, _ in rw_bm25_ranked[: args.pool_size]])

        if args.date_filter:
            # Safety net: pre-filtering (the mask above) already prevents invalid candidates
            # from leaking into the top-k; this only guards against the (rare) case where
            # the number of valid candidates is smaller than pool-size, letting -inf-scored
            # candidates spill into the top-k.
            pool_ids = [cid for cid in pool_ids if valid_at(corpus_by_id[cid], record["ozelgeTarih"])]

        queries.append({
            "key": f"{record['ozelgeTarih']}|{record['baslik']}",
            "soru": record["soru"],
            "ozelgeTarih": record["ozelgeTarih"],
            "baslik": record["baslik"],
            "relevant_ids": sorted(relevant_ids),
            "pool_ids": pool_ids,
        })

    payload = {
        "queries": queries,
        "n_skipped_out_of_corpus": n_skipped_out_of_corpus,
        "out_of_corpus_ids": sorted(out_of_corpus_ids),
        "boilerplate_ids": sorted(boilerplate_ids),
    }
    path = pool_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"[pool] Wrote top-{args.pool_size} candidate pool for {len(queries)} queries: {path}")


def build_local_reranker(args: argparse.Namespace) -> tuple["CrossEncoder", bool]:
    slug_lower = reranker_slug(args.reranker_model).lower()
    quantized = reranker_is_quantized(args.reranker_model, args.no_quantize_4bit)
    ce_kwargs: dict = {}
    if quantized:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        ce_kwargs["model_kwargs"] = {"quantization_config": bnb_config}
    elif slug_lower in RERANKER_BF16_MODELS:
        # flash-attn isn't installed (no prebuilt wheel yet for RTX 5090/sm_120); we
        # explicitly force SDPA, otherwise the eager fallback can use O(n^2) memory/time.
        ce_kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    max_length = RERANKER_MAX_LENGTH.get(slug_lower)
    if max_length:
        ce_kwargs["max_length"] = max_length
    reranker = CrossEncoder(args.reranker_model, **ce_kwargs)
    print(f"[rerank] device = {reranker.model.device}" + ("  [4-bit quantized]" if quantized else ""))
    return reranker, quantized


def vllm_rerank_pool(
    session: requests.Session, vllm_url: str, vllm_model: str,
    soru: str, pool_ids: list, corpus_by_id: dict,
) -> list:
    documents = [chunk_text(corpus_by_id[cid]) for cid in pool_ids]
    resp = session.post(
        f"{vllm_url}/rerank",
        json={"model": vllm_model, "query": soru, "documents": documents, "top_n": 0},
        timeout=300,
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    return [(pool_ids[r["index"]], r["relevance_score"]) for r in results]


def iter_reranked(args: argparse.Namespace, payload: dict, corpus_by_id: dict):
    """Yields (query_dict, reranked[(chunk_id, score), ...]) for each query. Depending on
    backend, uses either the local CrossEncoder (sequential) or vLLM /rerank (concurrent
    HTTP requests)."""
    if args.backend == "local":
        reranker, _ = build_local_reranker(args)
        for q in payload["queries"]:
            if not q["pool_ids"]:  # --date-filter may have emptied the pool (rare)
                yield q, []
                continue
            pairs = [(q["soru"], chunk_text(corpus_by_id[cid])) for cid in q["pool_ids"]]
            scores = reranker.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
            reranked = sorted(zip(q["pool_ids"], scores), key=lambda x: x[1], reverse=True)
            yield q, reranked
    else:
        print(f"[rerank] vLLM backend: {args.vllm_url}  model={args.vllm_model}  concurrency={args.vllm_concurrency}")
        session = requests.Session()

        def score_one(q: dict):
            if not q["pool_ids"]:
                return q, []
            return q, vllm_rerank_pool(session, args.vllm_url, args.vllm_model, q["soru"], q["pool_ids"], corpus_by_id)

        with ThreadPoolExecutor(max_workers=args.vllm_concurrency) as ex:
            yield from ex.map(score_one, payload["queries"])


# --- Stage 2: re-rank the candidate pool with the cross-encoder, score it ---
def run_rerank_stage(args: argparse.Namespace) -> None:
    print(f"[rerank] Reranker: {args.reranker_model}  Corpus: {args.corpus}  Backend: {args.backend}")

    path = pool_path(args)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    corpus_docs = db.load_corpus(args.corpus)
    id_field = "id" if args.corpus == "article" else "chunk_id"
    corpus_by_id = {doc[id_field]: doc for doc in corpus_docs}

    quantized = args.backend == "local" and reranker_is_quantized(args.reranker_model, args.no_quantize_4bit)

    metrics: dict[str, list[float]] = {f"recall@{k}": [] for k in KS}
    metrics.update({f"precision@{k}": [] for k in KS})
    missed: list[dict] = []
    per_query_keys: list[str] = []
    per_query_recall_10: list[float] = []
    ranked_dump: dict[str, list] = {}

    n_total = len(payload["queries"])
    t0 = time.monotonic()
    for i, (q, reranked) in enumerate(iter_reranked(args, payload, corpus_by_id), start=1):
        relevant_ids = set(q["relevant_ids"])

        if args.dump_ranked:
            ranked_dump[q["key"]] = [cid for cid, _ in reranked]

        if i % 10 == 0 or i == n_total:
            elapsed = time.monotonic() - t0
            rate = elapsed / i
            eta = rate * (n_total - i)
            print(f"[rerank] {i}/{n_total}  {elapsed:.0f}s elapsed  ~{eta:.0f}s remaining", flush=True)

        pool_article_ids = [corpus_by_id[cid]["article_id"] for cid, _ in reranked]
        retrieved_ids = dedupe_ranked(pool_article_ids)

        for k in KS:
            metrics[f"recall@{k}"].append(recall_at_k(retrieved_ids, relevant_ids, k))
            metrics[f"precision@{k}"].append(precision_at_k(retrieved_ids, relevant_ids, k))

        per_query_keys.append(q["key"])
        per_query_recall_10.append(recall_at_k(retrieved_ids, relevant_ids, 10))

        if recall_at_k(retrieved_ids, relevant_ids, 10) == 0.0:
            missed.append({
                "ozelgeTarih": q["ozelgeTarih"],
                "baslik": q["baslik"],
                "relevant": list(relevant_ids),
                "top5": retrieved_ids[:5],
            })

    total_elapsed = time.monotonic() - t0
    n = len(metrics[f"recall@{KS[0]}"])
    results: dict = {
        "method": "cross_encoder_rerank",
        "reranker_model": args.reranker_model,
        "first_stage_model": args.first_stage_model,
        "pool_size": args.pool_size,
        "corpus": args.corpus,
        "n_queries": n,
        "scores": {},
        "quantized_4bit": quantized,
        "hybrid_pool": args.hybrid_pool,
        "rewrite_pool": args.rewrite_pool,
        "backend": args.backend,
        "vllm_concurrency": args.vllm_concurrency if args.backend == "vllm" else None,
        "elapsed_seconds": round(total_elapsed, 1),
        "seconds_per_query": round(total_elapsed / n, 3) if n else None,
    }

    print(f"\nQuery count: {n}  (skipped for out-of-corpus reference: {payload['n_skipped_out_of_corpus']})", end="")
    if args.exclude_boilerplate:
        print(f"  (boilerplate articles: {payload['boilerplate_ids']})")
    else:
        print()
    print()
    print(f"{'Metric':<15} {'Average':>10}")
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
    results["out_of_corpus_ids"] = payload["out_of_corpus_ids"]
    results["n_skipped_out_of_corpus"] = payload["n_skipped_out_of_corpus"]
    results["per_query"] = {"keys": per_query_keys, "recall@10": per_query_recall_10}
    if args.exclude_boilerplate:
        results["boilerplate_ids"] = payload["boilerplate_ids"]

    suffix = "_noboilerplate" if args.exclude_boilerplate else ""
    suffix += "_traintest" if args.include_train else ""
    suffix += "_hybridpool" if args.hybrid_pool else ""
    suffix += "_rewritepool" if args.rewrite_pool else ""
    suffix += "_datefilter" if args.date_filter else ""
    out = out_path(args, args.corpus + suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if args.dump_ranked:
        args.dump_ranked.parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_ranked, "w", encoding="utf-8") as f:
            json.dump(ranked_dump, f, ensure_ascii=False)
        print(f"Ranked chunk_id lists (no dedupe): {args.dump_ranked}")

    print(f"Queries with recall@10 = 0: {len(missed)}/{n}")
    if missed:
        print("\nSample failed queries:")
        for m in missed[:3]:
            print(f"  {m['ozelgeTarih']} | {m['baslik'][:60]}")
            print(f"    relevant: {m['relevant']}")

    print(f"\nOutput: {out}")

    path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.stage == "all":
        run_subprocess("pool", args)
        run_subprocess("rerank", args)
    elif args.stage == "pool":
        run_pool_stage(args)
    else:
        run_rerank_stage(args)


if __name__ == "__main__":
    main()
