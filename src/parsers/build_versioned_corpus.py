"""
Versioned chunk corpus for point-in-time retrieval.

Builds on data/chunks/kdv_maddeler_chunks.jsonl (174 chunks, current text) — instead of
chunking from scratch, version/valid_from/valid_until tags are added to the existing
corpus (each chunk represents the MOST CURRENT version of that article). For the 8
multi-version articles (13, 16, 17, 24, 29, 36, 46, 58 — reconstructed from historical
edit records, see data/processed/kdv_maddeler_versiyonlu.jsonl, published on HF), the
non-current versions are additionally chunked with split_text and added to the corpus.
The other 102 articles (no record in kdv_maddeler_versiyonlu.jsonl, or single-version)
are treated as unchanged, valid_from=valid_until=None (always valid).

Note: an article can have more than one base chunk (e.g. Article 17 -> 20 chunks); the
historical versions are added once PER ARTICLE, not per chunk (otherwise an article with
N chunks would have its history repeated N times).

Output: data/chunks/kdv_maddeler_versioned_chunks.jsonl
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from chunk_text import split_text
from article_chunker import SPLIT_THRESHOLD

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHUNK_CORPUS_PATH = DATA_DIR / "chunks" / "kdv_maddeler_chunks.jsonl"
VERSIONED_PATH = DATA_DIR / "processed" / "kdv_maddeler_versiyonlu.jsonl"
OUT_PATH = DATA_DIR / "chunks" / "kdv_maddeler_versioned_chunks.jsonl"


def load_version_history() -> dict[int, list[dict]]:
    by_article: dict[int, list[dict]] = defaultdict(list)
    with open(VERSIONED_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_article[r["madde_id"]].append(r)
    for versions in by_article.values():
        versions.sort(key=lambda v: v["version"])
    return dict(by_article)


def historical_chunks(article_id: int, version: dict, title: str | None, kanun_title: str | None) -> list[dict]:
    metin = (version["metin"] or "").strip()
    if not metin:
        return []
    parts = [(None, metin)] if len(metin) <= SPLIT_THRESHOLD else split_text(metin)

    chunks = []
    for part_idx, (section, chunk_text) in enumerate(parts):
        if not chunk_text.strip():
            continue
        chunks.append({
            "chunk_id":     f"MADDE_{article_id}_v{version['version']}_{part_idx:03d}",
            "article_id":   article_id,
            "title":        title,
            "kanunNoTitle": kanun_title,
            "section":      section,
            "part":         part_idx,
            "metin":        chunk_text,
            "metin_len":    len(chunk_text),
            "version":      version["version"],
            "valid_from":   version["valid_from"],
            "valid_until":  version["valid_until"],
        })
    return chunks


def main() -> None:
    version_history = load_version_history()

    base_chunks = []
    titles: dict[int, str | None] = {}
    kanun_titles: dict[int, str | None] = {}
    with open(CHUNK_CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            base_chunks.append(chunk)
            titles.setdefault(chunk["article_id"], chunk.get("title"))
            kanun_titles.setdefault(chunk["article_id"], chunk.get("kanunNoTitle"))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_base = n_extra = n_multi_version_articles = 0
    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for chunk in base_chunks:
            versions = version_history.get(chunk["article_id"])
            latest = versions[-1] if versions else None
            chunk["version"] = latest["version"] if latest else 1
            chunk["valid_from"] = latest["valid_from"] if latest else None
            chunk["valid_until"] = latest["valid_until"] if latest else None
            fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            n_base += 1

        for article_id, versions in version_history.items():
            if len(versions) <= 1:
                continue
            n_multi_version_articles += 1
            title = titles.get(article_id)
            kanun_title = kanun_titles.get(article_id)
            for version in versions[:-1]:  # the latest version is already represented in base_chunks
                for extra_chunk in historical_chunks(article_id, version, title, kanun_title):
                    fout.write(json.dumps(extra_chunk, ensure_ascii=False) + "\n")
                    n_extra += 1

    print(f"Current (base) chunks  : {n_base}")
    print(f"Historical extra chunks: {n_extra}  ({n_multi_version_articles} multi-version articles)")
    print(f"Total                  : {n_base + n_extra}")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
