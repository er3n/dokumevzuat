"""
Point-in-time scorer: computes version-level recall@k for the three variants.

Variant "blind" (A, current/date-blind): no retrieval needed. The champion pipeline's
EXISTING (unversioned, current-text) chunk-corpus result's `per_query.recall@10` is
masked based on whether ALL of the query's citations are consistent with the CURRENT
version as of that date — if even one is inconsistent (the law changed after the ruling
date), it's zeroed out. An accepted simplification: for multi-citation queries, partial
correctness (some citations right, some wrong) isn't distinguished, it's all-or-nothing.

Variant "all_versions"/"all_versions_filtered" (B/C): uses the ranked chunk_id lists
(no dedup) produced by cross_encoder_rerank.py --corpus versioned --dump-ranked, plus
the versioned corpus's chunk_id -> (article_id, version) map, to dedupe by
(article_id, version) pair and compute recall@k (dedupe_ranked/recall_at_k/precision_at_k
are the generic, same functions as the article_id-only usage elsewhere, just keyed by tuple).

All three variants only score the "point-in-time subset" (queries citing at least one
multi-version article, point_in_time_ground_truth.point_in_time_subset) — so all three
are comparable on the same query set.

Output: the standard per_query{keys, recall@10} format (compatible with bootstrap_compare.py).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from bm25_baseline import (
    compute_out_of_corpus_ids,
    dedupe_ranked,
    filter_valid_queries,
    load_test,
    precision_at_k,
    recall_at_k,
)
from point_in_time_ground_truth import load_version_index, point_in_time_subset, resolve_version

DATA_DIR = Path(__file__).parent.parent.parent / "data"
VERSIONED_CHUNK_CORPUS = DATA_DIR / "chunks" / "kdv_maddeler_versioned_chunks.jsonl"
KS = [1, 3, 5, 10]


def load_chunk_version_map() -> dict[str, tuple[int, int]]:
    m = {}
    with open(VERSIONED_CHUNK_CORPUS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            m[r["chunk_id"]] = (r["article_id"], r["version"])
    return m


def resolve_targets(record: dict, index: dict) -> set[tuple[int, int]]:
    """The set of (article_id, version) targets that are correct as of that date, derived
    from the query's madde_atiflar. Out-of-corpus citations (e.g. VUK) get None from
    resolve_version and are skipped."""
    targets = set()
    for atif in record.get("madde_atiflar") or []:
        version = resolve_version(atif["id"], record["ozelgeTarih"], index)
        if version is not None:
            targets.add((atif["id"], version))
    return targets


def finalize(metrics: dict, per_query_keys: list, per_query_recall_10: list) -> dict:
    scores = {key: round(sum(vals) / len(vals), 4) for key, vals in metrics.items() if vals}
    return {
        "n_queries": len(per_query_keys),
        "scores": scores,
        "per_query": {"keys": per_query_keys, "recall@10": per_query_recall_10},
    }


def score_variant_bc(
    ranked_path: Path, records_by_key: dict, chunk_version_map: dict, index: dict, subset_keys: set[str]
) -> dict:
    with open(ranked_path, encoding="utf-8") as f:
        ranked_by_key = json.load(f)

    metrics: dict[str, list[float]] = defaultdict(list)
    per_query_keys, per_query_recall_10 = [], []
    for key in sorted(subset_keys):
        if key not in ranked_by_key:
            continue
        record = records_by_key[key]
        targets = resolve_targets(record, index)
        if not targets:
            continue

        ranked_pairs = dedupe_ranked([
            chunk_version_map[cid] for cid in ranked_by_key[key] if cid in chunk_version_map
        ])

        for k in KS:
            metrics[f"recall@{k}"].append(recall_at_k(ranked_pairs, targets, k))
            metrics[f"precision@{k}"].append(precision_at_k(ranked_pairs, targets, k))
        per_query_keys.append(key)
        per_query_recall_10.append(recall_at_k(ranked_pairs, targets, 10))

    return finalize(metrics, per_query_keys, per_query_recall_10)


def score_variant_blind(
    champion_result_path: Path, records_by_key: dict, index: dict, subset_keys: set[str]
) -> dict:
    with open(champion_result_path, encoding="utf-8") as f:
        champion = json.load(f)
    champion_per_query = dict(zip(champion["per_query"]["keys"], champion["per_query"]["recall@10"]))

    per_query_keys, per_query_recall_10 = [], []
    for key in sorted(subset_keys):
        if key not in champion_per_query:
            continue
        record = records_by_key[key]
        touched_ids = {a["id"] for a in (record.get("madde_atiflar") or []) if a["id"] in index}
        consistent = all(
            resolve_version(aid, record["ozelgeTarih"], index) == index[aid][-1]["version"]
            for aid in touched_ids
        )
        recall = champion_per_query[key] if consistent else 0.0
        per_query_keys.append(key)
        per_query_recall_10.append(recall)

    return finalize({"recall@10": per_query_recall_10}, per_query_keys, per_query_recall_10)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["blind", "all_versions", "all_versions_filtered"])
    p.add_argument("--champion-result", type=Path,
                    help="for variant=blind: the current champion (unversioned) chunk-corpus result json")
    p.add_argument("--ranked", type=Path,
                    help="for variant=all_versions[_filtered]: --dump-ranked output")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.variant == "blind" and args.champion_result is None:
        p.error("--variant blind requires --champion-result")
    if args.variant != "blind" and args.ranked is None:
        p.error(f"--variant {args.variant} requires --ranked")
    return args


def main() -> None:
    args = parse_args()
    index = load_version_index()

    records = load_test(include_train=True)
    valid, _, _ = filter_valid_queries(records, compute_out_of_corpus_ids())
    records_by_key = {f"{r['ozelgeTarih']}|{r['baslik']}": r for r in valid}

    subset, n_law_changed = point_in_time_subset(valid, index)
    subset_keys = {f"{r['ozelgeTarih']}|{r['baslik']}" for r in subset}
    print(f"Point-in-time subset: {len(subset_keys)} queries ({n_law_changed} where the law genuinely changed)")

    if args.variant == "blind":
        result = score_variant_blind(args.champion_result, records_by_key, index, subset_keys)
    else:
        chunk_version_map = load_chunk_version_map()
        result = score_variant_bc(args.ranked, records_by_key, chunk_version_map, index, subset_keys)

    result["variant"] = args.variant
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"n_queries={result['n_queries']}  recall@10={result['scores'].get('recall@10')}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
