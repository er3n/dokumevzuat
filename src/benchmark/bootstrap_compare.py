"""
Compares two result files (json produced by the bm25/dense/hybrid baselines) with
paired bootstrap resampling — the method used throughout this project's statistical
comparisons (e.g. Nomic v2 vs Qwen3-Embedding-8B, on the raw vs boilerplate-excluded
splits).
Both systems must have been run on the same test set; queries are matched via the
`per_query.keys` field (order doesn't matter).
The metric defaults to recall@10 (every baseline stores this in `per_query`).

Usage:
    python src/benchmark/bootstrap_compare.py \
        --a data/benchmark/dense_results_Qwen3-Embedding-8B_chunk.json \
        --b data/benchmark/hybrid_results_Qwen3-Embedding-8B_chunk.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, type=Path, help="baseline result json (A)")
    parser.add_argument("--b", required=True, type=Path, help="result json to compare against (B)")
    parser.add_argument("--metric", default="recall@10")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_per_query(path: Path, metric: str) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    per_query = data.get("per_query")
    if not per_query or metric not in per_query:
        raise ValueError(f"{path}: 'per_query.{metric}' not found — this file isn't ready for bootstrap comparison")
    return dict(zip(per_query["keys"], per_query[metric]))


def main() -> None:
    args = parse_args()
    a = load_per_query(args.a, args.metric)
    b = load_per_query(args.b, args.metric)

    common_keys = sorted(set(a) & set(b))
    if not common_keys:
        raise ValueError("No common queries found between the two files (per_query.keys don't match)")
    if len(common_keys) < len(a) or len(common_keys) < len(b):
        print(f"Warning: A={len(a)} queries, B={len(b)} queries, common={len(common_keys)} queries "
              f"— comparing only on the common queries")

    scores_a = np.array([a[k] for k in common_keys])
    scores_b = np.array([b[k] for k in common_keys])
    n = len(common_keys)

    point_diff = scores_b.mean() - scores_a.mean()

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.n_boot, n))
    diffs = scores_b[idx].mean(axis=1) - scores_a[idx].mean(axis=1)

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    prob_b_better = (diffs > 0).mean()
    confidence = max(prob_b_better, 1 - prob_b_better)
    direction = "B > A" if prob_b_better >= 0.5 else "A > B"

    print(f"Metric: {args.metric}  n={n} queries  n_boot={args.n_boot}")
    print(f"  A ({args.a.name}): mean={scores_a.mean():.4f}")
    print(f"  B ({args.b.name}): mean={scores_b.mean():.4f}")
    print(f"  diff (B - A): {point_diff:+.4f}")
    print(f"  95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"  P(B > A) = {prob_b_better:.4f}  ->  {direction}, {confidence * 100:.0f}% confidence")
    if ci_low <= 0 <= ci_high:
        print("  CI crosses zero -> difference is not statistically significant (within noise)")
    else:
        print("  CI does not cross zero -> difference is statistically significant")


if __name__ == "__main__":
    main()
