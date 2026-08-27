"""
Publish the benchmark dataset to Hugging Face.
Repo: dokukoza/kdv-rag-benchmark
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path

REPO_ID   = "dokukoza/kdv-rag-benchmark"
REPO_TYPE = "dataset"
ROOT      = Path(__file__).parent.parent

FILES = [
    # (local_path, repo_path)
    (ROOT / "hf_dataset/README.md",                        "README.md"),
    (ROOT / "data/benchmark/train.jsonl",                   "data/benchmark/train.jsonl"),
    (ROOT / "data/benchmark/test.jsonl",                    "data/benchmark/test.jsonl"),
    (ROOT / "data/processed/kdv_maddeler_parsed.jsonl",     "data/corpus/kdv_maddeler.jsonl"),
    (ROOT / "data/chunks/kdv_maddeler_chunks.jsonl",        "data/corpus_chunked/kdv_maddeler_chunks.jsonl"),
    (ROOT / "data/processed/kdv_maddeler_versiyonlu.jsonl", "data/corpus_versioned/kdv_maddeler_versiyonlu.jsonl"),
    (ROOT / "data/chunks/kdv_maddeler_versioned_chunks.jsonl", "data/corpus_versioned_chunked/kdv_maddeler_versioned_chunks.jsonl"),
]

# Baseline result files for the target protocol (VUK-attribution cleaned, train+test
# combined, n=813 raw / n=270 clean) — published as the raw experiment records.
RESULTS_FILES = [
    "bm25_results_chunk_traintest.json",
    "bm25_results_article_traintest.json",
    "bm25_results_article_noboilerplate_traintest.json",
    "dense_results_multilingual-e5-large_chunk_traintest.json",
    "dense_results_multilingual-e5-large_chunk_noboilerplate_traintest.json",
    "dense_results_multilingual-e5-large_article_traintest.json",
    "dense_results_bge-m3_chunk_traintest.json",
    "dense_results_bge-m3_chunk_noboilerplate_traintest.json",
    "dense_results_bge-m3_article_traintest.json",
    "dense_results_Qwen3-Embedding-0.6B_chunk_traintest.json",
    "dense_results_Qwen3-Embedding-0.6B_chunk_noboilerplate_traintest.json",
    "dense_results_Qwen3-Embedding-4B_chunk_traintest.json",
    "dense_results_Qwen3-Embedding-4B_chunk_noboilerplate_traintest.json",
    "dense_results_turkish-embedding-model-fine-tuned_chunk_traintest.json",
    "dense_results_turkish-embedding-model-fine-tuned_chunk_noboilerplate_traintest.json",
    "rerank_results_Qwen3-Reranker-8B-vllm_pool50_chunk_traintest.json",
    "rerank_results_Qwen3-Reranker-8B-vllm_pool50_chunk_noboilerplate_traintest.json",
]
FILES += [
    (ROOT / "data/benchmark" / fn, f"data/benchmark_results/{fn}")
    for fn in RESULTS_FILES
]


def main() -> None:
    api = HfApi()

    print(f"Creating repo: {REPO_ID}")
    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        exist_ok=True,
        private=False,
    )

    for local, remote in FILES:
        print(f"  uploading: {remote}")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )

    print(f"\nDone -> https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
