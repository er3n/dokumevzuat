"""
VAT Law article chunker.
Splits the longer articles (>TARGET_SIZE characters, 27 of the 110 articles in
kdv_maddeler_parsed.jsonl) at numbered clause/sentence boundaries; the remaining
83 articles stay whole as a single chunk (to avoid unnecessary fragmentation).

The split logic is reused from chunk_text.py (split_text) — same SECTION_RE /
sentence-boundary strategy.

Each chunk: chunk_id, article_id, title, kanunNoTitle, section, part, metin, metin_len
article_id = the original article id (matches benchmark ground truth via madde_atiflar).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from chunk_text import split_text, TARGET_SIZE

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SRC_PATH = DATA_DIR / "processed" / "kdv_maddeler_parsed.jsonl"
OUT_PATH = DATA_DIR / "chunks" / "kdv_maddeler_chunks.jsonl"

SPLIT_THRESHOLD = TARGET_SIZE  # articles below this size are not split, they stay a single chunk


def chunk_madde(doc: dict) -> list[dict]:
    metin = (doc.get("metin") or "").strip()
    if not metin:
        return []

    if len(metin) <= SPLIT_THRESHOLD:
        parts = [(None, metin)]
    else:
        parts = split_text(metin)

    results = []
    for part_idx, (section, chunk_text) in enumerate(parts):
        if not chunk_text.strip():
            continue
        results.append({
            "chunk_id":     f"MADDE_{doc['id']}_{part_idx:03d}",
            "article_id":   doc["id"],
            "title":        doc.get("title"),
            "kanunNoTitle": doc.get("kanunNoTitle"),
            "section":      section,
            "part":         part_idx,
            "metin":        chunk_text,
            "metin_len":    len(chunk_text),
        })
    return results


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_docs = n_chunks = n_split_docs = 0
    with open(SRC_PATH, encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            doc = json.loads(line)
            chunks = chunk_madde(doc)
            if len(chunks) > 1:
                n_split_docs += 1
            for chunk in chunks:
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                n_chunks += 1
            n_docs += 1

    print(f"Articles: {n_docs}  (split: {n_split_docs})")
    print(f"Chunks: {n_chunks}")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
