import pandas as pd
import os

base_path = os.path.join("data", "derived")

# Phase 1 features (now includes subreddit)
features_file = os.path.join(base_path, "comment_complexity_features.csv")

# Cluster files
cluster_files = {
    "k2": "cluster_labels.csv",
    "k4": "cluster_labelsk4.csv",
    "k5": "cluster_labelsk5.csv",
}

features_df = pd.read_csv(features_file)

# Merge subreddit-level clusters onto the feature table
for label, filename in cluster_files.items():
    cluster_path = os.path.join(base_path, filename)
    cluster_df = pd.read_csv(cluster_path)

    merged_df = features_df.merge(cluster_df, how="left", on="subreddit")

    output_file = os.path.join(base_path, f"comments_with_clusters_{label}.csv")
    merged_df.to_csv(output_file, index=False)

    print(f"Saved {output_file}")
    print(f"Rows: {merged_df.shape[0]}")