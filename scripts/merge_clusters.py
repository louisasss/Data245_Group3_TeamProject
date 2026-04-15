import pandas as pd
import os

base_path = os.path.join("data", "derived")

features_path = os.path.join(base_path, "comment_complexity_features.csv")

cluster_files = {
    "k2": "cluster_labels.csv",
    "k4": "cluster_labels4.csv",
    "k5": "cluster_labels5.csv",
}

features_df = pd.read_csv(features_path)


for label, filename in cluster_files.items():
    cluster_path = os.path.join(base_path, filename)
    cluster_df = pd.read_csv(cluster_path)

    print(f"\n--- Processing {filename} ({label}) ---")
    print("cluster columns:", cluster_df.columns.tolist())

    merged_df = features_df.merge(cluster_df, how="left", on="comment_id")

    output_path = os.path.join(base_path, f"comments_with_clusters_{label}.csv")

    print("Merged shape:", merged_df.shape)
    print("NaNs per column:")
    print(merged_df.isna().sum())

    merged_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")