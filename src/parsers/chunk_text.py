"""
Shared text-splitting logic for chunking law-article-style text.

Strategy:
  1. MADDE X- pattern -> per-article boundaries
  2. Numbered section headers (1. / 1.1. / 1.1.1.) -> numbered clause boundaries
  3. Neither pattern present -> split on sentence boundaries at ~TARGET_SIZE
"""

import re

TARGET_SIZE = 2000  # characters
MAX_SIZE    = 4000  # anything above this is always split

MADDE_RE    = re.compile(r'(?=MADDE\s+\d+\s*[–\-])')
# Lookbehind: the section number must start after whitespace (avoids a mid-word split on "0.")
SECTION_RE  = re.compile(r'(?<=\s)(?=\d+(?:\.\d+)*\.\s+[A-ZÇĞIÖŞÜ])')
SENT_END_RE = re.compile(r'(?<=[.!?"])\s+(?=[A-ZÇĞIÖŞÜ0-9"\(])')
MIN_CHUNK   = 100  # chunks below this size are merged into the previous one


def split_by_sentences(text: str, target: int = TARGET_SIZE) -> list[str]:
    """Split on sentence boundaries, each part ~target chars."""
    parts = SENT_END_RE.split(text)
    chunks, current = [], ""
    for part in parts:
        if len(current) + len(part) <= target or not current:
            current += (" " if current else "") + part
        else:
            chunks.append(current.strip())
            current = part
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


def merge_tiny(pairs: list[tuple[str | None, str]]) -> list[tuple[str | None, str]]:
    """Merge chunks below MIN_CHUNK into the previous one."""
    result: list[tuple[str | None, str]] = []
    for header, text in pairs:
        if result and len(text) < MIN_CHUNK:
            prev_h, prev_t = result[-1]
            result[-1] = (prev_h, prev_t + " " + text)
        else:
            result.append((header, text))
    return result


def split_text(text: str) -> list[tuple[str | None, str]]:
    """
    Splits text into a list of (section_header, chunk_text) pairs.
    section_header: the first line / section title (None if there isn't one).
    """
    if not text.strip():
        return []

    # Split on MADDE boundaries
    if MADDE_RE.search(text):
        raw = MADDE_RE.split(text)
        raw = [r.strip() for r in raw if r.strip()]
        results = []
        for block in raw:
            header = block.split(" ", 3)[:3]  # "MADDE 7 -"
            header_str = " ".join(header).rstrip("-– ").strip()
            if len(block) <= MAX_SIZE:
                results.append((header_str, block))
            else:
                for sub in split_by_sentences(block):
                    results.append((header_str, sub))
        return merge_tiny(results)

    # Split on numbered section boundaries
    if SECTION_RE.search(text):
        raw = SECTION_RE.split(text)
        raw = [r.strip() for r in raw if r.strip()]
        results = []
        for block in raw:
            header_str = block[:80].split(".")[0].strip() if block else None
            if len(block) <= MAX_SIZE:
                results.append((header_str, block))
            else:
                for sub in split_by_sentences(block):
                    results.append((header_str, sub))
        return merge_tiny(results)

    # Unstructured text: split on sentence boundaries
    if len(text) <= MAX_SIZE:
        return [(None, text)]
    return [(None, sub) for sub in split_by_sentences(text)]
