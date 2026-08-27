"""
BM25 baseline.
Corpus  : VAT Law articles — --corpus article (110 articles, default) or
          --corpus chunk (174 chunks, long articles split — article_chunker.py)
Query   : the question from a test ruling (ozelge)
Relevant: madde_atiflar (matched by id)
Metric  : recall@k, precision@k  (k = 1, 3, 5, 10)
Note    : in the chunk corpus multiple chunks can belong to the same article
          (article_id); the ranked result list is deduped by article_id (best
          rank kept) before computing recall/precision, for a fair
          article-level comparison.
--exclude-boilerplate: articles cited in more than BOILERPLATE_THRESHOLD of
          rulings (e.g. Article 28 Rate, Article 1 Subject of the Tax —
          procedural articles cited in nearly every ruling regardless of
          topic) are removed from the ground truth. These articles can never
          be found by content-matching retrieval (BM25 or dense) because
          there is no word/meaning overlap with the query's actual topic.
          Queries whose ground truth is entirely boilerplate are skipped in
          this mode.
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

from rank_bm25 import BM25Okapi

DATA_DIR          = Path(__file__).parent.parent.parent / "data"
ARTICLE_CORPUS_PATH = DATA_DIR / "processed" / "kdv_maddeler_parsed.jsonl"
CHUNK_CORPUS_PATH   = DATA_DIR / "chunks" / "kdv_maddeler_chunks.jsonl"
VERSIONED_CORPUS_PATH = DATA_DIR / "chunks" / "kdv_maddeler_versioned_chunks.jsonl"
TEST_PATH           = DATA_DIR / "benchmark" / "test.jsonl"
TRAIN_PATH           = DATA_DIR / "benchmark" / "train.jsonl"

CORPUS_PATHS = {
    "article": ARTICLE_CORPUS_PATH,
    "chunk": CHUNK_CORPUS_PATH,
    "versioned": VERSIONED_CORPUS_PATH,
}

BOILERPLATE_THRESHOLD = 0.15  # an article cited in more than this fraction of rulings = boilerplate

KS = [1, 3, 5, 10]

STOPWORDS = {
    "ve", "ile", "bir", "bu", "da", "de", "için", "olan", "olan", "olan",
    "olarak", "veya", "daha", "gibi", "kadar", "üzere", "ise", "hem",
    "ya", "ki", "ne", "mi", "mu", "mı", "mü", "en", "çok", "her",
    "ancak", "aynı", "bazı", "buna", "den", "dan", "ten", "tan", "nin",
    "nın", "nun", "nün", "ın", "in", "un", "ün", "ler", "lar",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-ZçğışöüÇĞİŞÖÜ0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def load_corpus(corpus: str = "article") -> tuple[list[dict], list[list[str]]]:
    path = CORPUS_PATHS[corpus]
    docs, tokens = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc["article_id"] = doc["id"] if corpus == "article" else doc["article_id"]
            text = (doc.get("title") or "") + " " + (doc.get("metin") or "")
            docs.append(doc)
            tokens.append(tokenize(text))
    return docs, tokens


def dedupe_ranked(ids: list[int]) -> list[int]:
    """Deduplicates a ranked id list, keeping first-seen order (chunk -> article)."""
    seen: set[int] = set()
    result = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            result.append(i)
    return result


def load_test(include_train: bool = False) -> list[dict]:
    """With --include-train, train.jsonl is also included (154 -> 882 queries). Since no
    model is fine-tuned on train.jsonl (all are pretrained/zero-shot), adding it to eval
    carries no leakage risk — it only increases statistical power."""
    records = []
    paths = [TEST_PATH] + ([TRAIN_PATH] if include_train else [])
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    return records


def load_rewrite_cache(path: Path | None) -> dict[str, str]:
    """key -> rewrite text (output of generate_rewrites.py). If path is None, rewrite is off."""
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v["rewrite"] for k, v in data["rewrites"].items()}


def resolve_query(record: dict, rewrite_cache: dict[str, str]) -> str:
    """Falls back to the original `soru` if rewrite_cache is empty or the key is missing
    (generation failed or was skipped)."""
    key = f"{record['ozelgeTarih']}|{record['baslik']}"
    return rewrite_cache.get(key, record["soru"])


def compute_boilerplate_ids(threshold: float = BOILERPLATE_THRESHOLD) -> set[int]:
    """Article ids whose citation frequency across all train+test rulings exceeds threshold."""
    records = []
    for path in (TRAIN_PATH, TEST_PATH):
        with open(path, encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f)

    counts: Counter = Counter()
    for r in records:
        for a in r.get("madde_atiflar") or []:
            counts[a["id"]] += 1

    n = len(records)
    return {mid for mid, c in counts.items() if c / n >= threshold}


def compute_out_of_corpus_ids() -> set[int]:
    """Citations in train+test that are not present in the VAT Law article corpus
    (kdv_maddeler_parsed.jsonl, 110 articles). These belong to another law (e.g. the Tax
    Procedure Law) that was never scraped — no method can ever find them, and they
    artificially lower the ceiling."""
    madde_ids = set()
    with open(ARTICLE_CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            madde_ids.add(json.loads(line)["id"])

    atif_ids = set()
    for path in (TRAIN_PATH, TEST_PATH):
        with open(path, encoding="utf-8") as f:
            for line in f:
                for a in json.loads(line).get("madde_atiflar") or []:
                    atif_ids.add(a["id"])

    return atif_ids - madde_ids


def filter_valid_queries(
    records: list[dict], exclude_ids: set[int]
) -> tuple[list[dict], list[set[int]], int]:
    """Skips records with an empty question or no citations; exclude_ids (boilerplate +
    out-of-corpus ids) is subtracted from relevant_ids; if relevant_ids ends up empty
    after subtraction, that query is skipped too (counted in n_skipped) — recall@k could
    never be >0 for it under any method."""
    valid, relevant_ids_list = [], []
    n_skipped = 0
    for r in records:
        if not (r.get("soru") or "").strip() or not r.get("madde_atiflar"):
            continue
        relevant_ids = {a["id"] for a in r["madde_atiflar"]} - exclude_ids
        if not relevant_ids:
            n_skipped += 1
            continue
        valid.append(r)
        relevant_ids_list.append(relevant_ids)
    return valid, relevant_ids_list, n_skipped


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hit = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hit / len(relevant_ids)


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    hit = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hit / k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["article", "chunk", "versioned"], default="article")
    parser.add_argument("--exclude-boilerplate", action="store_true")
    parser.add_argument("--include-train", action="store_true")
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


def main() -> None:
    args = parse_args()
    corpus_docs, corpus_tokens = load_corpus(args.corpus)
    bm25 = BM25Okapi(corpus_tokens)

    test = load_test(include_train=args.include_train)
    corpus_article_ids = [doc["article_id"] for doc in corpus_docs]

    boilerplate_ids = compute_boilerplate_ids() if args.exclude_boilerplate else set()
    out_of_corpus_ids = compute_out_of_corpus_ids()
    valid, relevant_ids_list, n_skipped_out_of_corpus = filter_valid_queries(
        test, boilerplate_ids | out_of_corpus_ids
    )
    rewrite_cache = load_rewrite_cache(args.rewrite_cache) if args.query_mode == "rewrite" else {}

    metrics: dict[str, list[float]] = defaultdict(list)
    missed: list[dict] = []   # recall@10 = 0
    per_query_keys: list[str] = []
    per_query_recall_10: list[float] = []
    ranked_dump: dict[str, list[int]] = {}

    for record, relevant_ids in zip(valid, relevant_ids_list):
        q_tokens = tokenize(resolve_query(record, rewrite_cache))
        scores = bm25.get_scores(q_tokens)

        # Pair scores with corpus order, sort descending, dedupe to article level
        ranked = sorted(
            zip(corpus_article_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        retrieved_ids = dedupe_ranked([article_id for article_id, _ in ranked])

        for k in KS:
            r = recall_at_k(retrieved_ids, relevant_ids, k)
            p = precision_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"recall@{k}"].append(r)
            metrics[f"precision@{k}"].append(p)

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
        "n_queries": n,
        "scores": {},
    }

    print(f"Query count: {n}  (skipped for out-of-corpus citations: {n_skipped_out_of_corpus})", end="")
    if args.exclude_boilerplate:
        print(f"  (also includes boilerplate-only skips, boilerplate articles: {sorted(boilerplate_ids)})")
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
    results["corpus"] = args.corpus
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
    out_path = DATA_DIR / "benchmark" / f"bm25_results_{args.corpus}{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
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

    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
