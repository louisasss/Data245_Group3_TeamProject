"""
characterize_clusters.py

Quick analysis to support writing human-readable cluster labels for the demo.
For each K, prints:
  1. Distinctiveness (z-scored centroids) — what defines each cluster
  2. Subreddit composition (preview in terminal, full lists saved to disk)
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data/derived"
OUT_DIR = BASE_DIR / "results/cluster_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "emotional_entropy",
    "valence_mixing",
    "multi_label_count",
    "annotator_disagreement_score",
    "example_very_unclear",
]


def analyze(k):
    print(f"\n{'=' * 70}")
    print(f"  K = {k}")
    print(f"{'=' * 70}")

    df = pd.read_csv(DATA_DIR / f"comments_with_clusters_k{k}.csv")

    # Distinctiveness
    centroids = df.groupby("cluster")[FEATURES].mean()
    overall_mean = df[FEATURES].mean()
    overall_std = df[FEATURES].std()
    distinctiveness = (centroids - overall_mean) / overall_std

    print(f"\n--- Distinctiveness z-scores (K={k}) ---")
    print("(positive = unusually high on this feature, negative = unusually low)")
    print(distinctiveness.round(2))

    # Subreddit composition
    print(f"\n--- Subreddits per cluster (K={k}) ---")
    sub_clusters = df.groupby("subreddit")["cluster"].first()
    for c in sorted(sub_clusters.unique()):
        members = sorted(sub_clusters[sub_clusters == c].index.tolist())
        print(f"\nCluster {c} ({len(members)} subreddits):")
        for m in members[:25]:
            print(f"  r/{m}")
        if len(members) > 25:
            print(f"  ... and {len(members) - 25} more")
        # Save full list
        out_path = OUT_DIR / f"subreddits_k{k}_cluster{c}.txt"
        out_path.write_text("\n".join(f"r/{m}" for m in members))
    print(f"\n(Full subreddit lists saved to {OUT_DIR})")


if __name__ == "__main__":
    for k in [3, 4]:
        analyze(k)