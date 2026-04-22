#!/usr/bin/env python
"""
Phase 2: Clustering Analysis
Analyzes K-means results, creates plots, and saves cluster label files
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import silhouette_score


def main():
    # Paths
    data_path = Path("data") / "derived"
    output_path = Path("results") / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("PHASE 2: CLUSTERING ANALYSIS")
    print("="*60)
    
    # Load data
    print("\nLoading subreddit vectors and kmeans results...")
    subreddit_vectors = pd.read_csv(data_path / "subreddit_vectors.csv")
    kmeans_results = pd.read_csv(data_path / "kmeans_results.csv")
    
    print(f"Loaded {len(subreddit_vectors)} subreddits")
    print(f"K values tested: {kmeans_results['k'].tolist()}")
    
    # Check for NaN
    nan_count = subreddit_vectors.isnull().sum().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found")
    
    # Feature columns
    ignore_cols = ["subreddit", "comment_count"]
    feature_cols = [c for c in subreddit_vectors.columns if c not in ignore_cols]
    print(f"\nUsing {len(feature_cols)} features for analysis")
    
    # Elbow plot
    print("\n" + "="*60)
    print("CREATING ELBOW PLOT")
    print("="*60)
    plt.figure(figsize=(8, 4))
    plt.plot(kmeans_results["k"], kmeans_results["inertia"], marker="o")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title("Elbow Plot")
    plt.grid(True, alpha=0.3)
    elbow_path = output_path / "elbow_plot.png"
    plt.savefig(elbow_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {elbow_path}")
    
    # Calculate silhouette scores
    print("\n" + "="*60)
    print("CALCULATING SILHOUETTE SCORES")
    print("="*60)
    X = subreddit_vectors[feature_cols].values
    k_values = kmeans_results["k"].tolist()
    silhouette_scores = []
    
    for _, row in kmeans_results.iterrows():
        labels = json.loads(row["cluster_labels"])
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        print(f"  K={int(row['k'])}: silhouette={score:.4f}")
    
    # Silhouette plot
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, silhouette_scores, marker="o")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores by K")
    plt.grid(True, alpha=0.3)
    sil_path = output_path / "silhouette_plot.png"
    plt.savefig(sil_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {sil_path}")
    
    # Find optimal K
    optimal_k_idx = np.argmax(silhouette_scores)
    optimal_k = k_values[optimal_k_idx]
    print(f"\nOptimal K by silhouette score: {optimal_k}")
    
    # Save cluster labels
    print("\n" + "="*60)
    print("SAVING CLUSTER LABELS")
    print("="*60)
    
    # Save for optimal K (K=2)
    save_cluster_labels(subreddit_vectors, kmeans_results, optimal_k, data_path, "cluster_labels.csv")
    
    # Save for K=4 (your project's chosen K)
    save_cluster_labels(subreddit_vectors, kmeans_results, 4, data_path, "cluster_labelsk4.csv")
    
    # Save for K=5
    save_cluster_labels(subreddit_vectors, kmeans_results, 5, data_path, "cluster_labelsk5.csv")

    # Save for k = 3 (based on silhouette score))
    save_cluster_labels(subreddit_vectors, kmeans_results, 3, data_path, "cluster_labelsk3.csv")

    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS COMPLETE")
    print("="*60)


def save_cluster_labels(subreddit_vectors, kmeans_results, k, output_dir, filename):
    """Save cluster labels for a specific K value"""
    optimal_labels = json.loads(
        kmeans_results[kmeans_results["k"] == k]["cluster_labels"].values[0]
    )
    
    cluster_df = pd.DataFrame({
        "subreddit": subreddit_vectors["subreddit"],
        "cluster": optimal_labels
    })
    
    output_path = output_dir / filename
    cluster_df.to_csv(output_path, index=False)
    
    print(f"\nSaved: {output_path}")
    print(f"  Clusters: {sorted(cluster_df['cluster'].unique())}")
    print(f"  Distribution: {dict(cluster_df['cluster'].value_counts().sort_index())}")


if __name__ == "__main__":
    main()