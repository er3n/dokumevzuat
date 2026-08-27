"""
Point-in-time ground truth: resolves, for a ruling query's `ozelgeTarih`, the VERSION of
the cited article that was in force on that date. Source: data/processed/kdv_maddeler_versiyonlu.jsonl
(reconstructed from historical edit records, published on HF) — 8 articles (13, 16, 17,
24, 29, 36, 46, 58) are genuinely multi-version; the remaining 102 articles (no record =
assumed never changed) are added as single-version synthetic records.

When run directly (`python point_in_time_ground_truth.py`), reports how many queries the
point-in-time experiment could plausibly "make a difference" for — a checkpoint before
running the B/C variants.
"""

import json
from collections import defaultdict
from pathlib import Path

from bm25_baseline import (
    ARTICLE_CORPUS_PATH,
    compute_boilerplate_ids,
    compute_out_of_corpus_ids,
    filter_valid_queries,
    load_test,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
VERSIONED_PATH = DATA_DIR / "processed" / "kdv_maddeler_versiyonlu.jsonl"


def load_version_index() -> dict[int, list[dict]]:
    """article_id -> [{version, valid_from, valid_until}, ...] in increasing version order."""
    index: dict[int, list[dict]] = defaultdict(list)
    with open(VERSIONED_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            index[r["madde_id"]].append({
                "version": r["version"],
                "valid_from": r["valid_from"],
                "valid_until": r["valid_until"],
            })
    with open(ARTICLE_CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            article_id = json.loads(line)["id"]
            if article_id not in index:
                index[article_id] = [{"version": 1, "valid_from": None, "valid_until": None}]
    for versions in index.values():
        versions.sort(key=lambda v: v["version"])
    return dict(index)


def multi_version_article_ids(index: dict[int, list[dict]]) -> set[int]:
    return {aid for aid, versions in index.items() if len(versions) > 1}


def resolve_version(article_id: int, ref_date: str, index: dict[int, list[dict]]) -> int | None:
    """Returns the version number in force on ref_date (ISO 'YYYY-MM-DD').
    valid_from <= ref_date < valid_until (None = +/-infinity); since versions are contiguous
    by construction, exactly one version should match. Returns None if article_id isn't in
    the index at all (an out-of-corpus citation, e.g. a different law)."""
    versions = index.get(article_id)
    if not versions:
        return None
    for v in versions:
        vf, vu = v["valid_from"], v["valid_until"]
        if (vf is None or ref_date >= vf) and (vu is None or ref_date < vu):
            return v["version"]
    return versions[-1]["version"]  # safety net — should never actually be reached


def point_in_time_subset(
    records: list[dict], index: dict[int, list[dict]]
) -> tuple[list[dict], int]:
    """Returns records that cite at least one multi-version article. Also reports how many
    of those queries have a ruling date where the CURRENT (latest) version is actually wrong
    — i.e. the law changed AFTER the ruling date. This count is a ceiling on how much "stale
    text" the date-blind/current approach could possibly return."""
    multi_ids = multi_version_article_ids(index)
    subset = []
    n_law_changed = 0
    for r in records:
        atif_ids = {a["id"] for a in (r.get("madde_atiflar") or [])}
        touched = atif_ids & multi_ids
        if not touched:
            continue
        subset.append(r)
        for aid in touched:
            resolved = resolve_version(aid, r["ozelgeTarih"], index)
            latest = index[aid][-1]["version"]
            if resolved != latest:
                n_law_changed += 1
                break
    return subset, n_law_changed


def report(label: str, records: list[dict], exclude: set[int], index: dict[int, list[dict]]) -> None:
    valid, _, n_skipped = filter_valid_queries(records, exclude)
    subset, n_law_changed = point_in_time_subset(valid, index)
    print(f"\n[{label}] valid queries: {len(valid)}  (skipped: {n_skipped})")
    print(f"[{label}] queries citing a multi-version article: {len(subset)}  "
          f"({100 * len(subset) / len(valid):.1f}%)")
    if subset:
        print(f"[{label}]   -- of these, the law GENUINELY changed for {n_law_changed} "
              f"(current text would have been wrong on that date, {100 * n_law_changed / len(subset):.1f}%)")


def main() -> None:
    index = load_version_index()
    multi_ids = multi_version_article_ids(index)
    print(f"Multi-version article count: {len(multi_ids)}  ({sorted(multi_ids)})")

    records = load_test(include_train=True)
    out_of_corpus_ids = compute_out_of_corpus_ids()
    boilerplate_ids = compute_boilerplate_ids()

    report("raw", records, out_of_corpus_ids, index)
    report("clean", records, out_of_corpus_ids | boilerplate_ids, index)


if __name__ == "__main__":
    main()
