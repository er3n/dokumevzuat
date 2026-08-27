"""
Fetch the benchmark dataset from Hugging Face into the local data/ layout that
src/parsers and src/benchmark expect.

This repo does not ship raw scraped data or the scrapers that produced it (see
README.md). Everything the benchmark scripts need is already published at
https://huggingface.co/datasets/er3nhf/kdv-rag-benchmark - this script just
downloads it and lays it out on disk under the paths the code reads.
"""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID   = "er3nhf/kdv-rag-benchmark"
REPO_TYPE = "dataset"
ROOT      = Path(__file__).parent.parent / "data"

# (path in the HF repo, local destination path)
FILES = [
    ("data/benchmark/train.jsonl",                              ROOT / "benchmark" / "train.jsonl"),
    ("data/benchmark/test.jsonl",                                ROOT / "benchmark" / "test.jsonl"),
    ("data/corpus/kdv_maddeler.jsonl",                           ROOT / "processed" / "kdv_maddeler_parsed.jsonl"),
    ("data/corpus_chunked/kdv_maddeler_chunks.jsonl",            ROOT / "chunks" / "kdv_maddeler_chunks.jsonl"),
    ("data/corpus_versioned/kdv_maddeler_versiyonlu.jsonl",      ROOT / "processed" / "kdv_maddeler_versiyonlu.jsonl"),
    ("data/corpus_versioned_chunked/kdv_maddeler_versioned_chunks.jsonl", ROOT / "chunks" / "kdv_maddeler_versioned_chunks.jsonl"),
]


def main() -> None:
    for repo_path, dest in FILES:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  fetching: {repo_path}")
        cached = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=repo_path)
        shutil.copyfile(cached, dest)

    print(f"\nDone -> {ROOT}")


if __name__ == "__main__":
    main()
