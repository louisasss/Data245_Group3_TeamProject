import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def build_subreddit_vectors(df) -> tuple:
    numeric_cols = df.columns.drop(["comment_id", "subreddit"], errors="ignore")

    agg_funcs = {col: ["mean", "std", "median", "min", "max"] for col in numeric_cols}

    grouped = df.groupby("subreddit").agg(agg_funcs)
    grouped.columns = ["_".join(c) for c in grouped.columns]
    grouped = grouped.reset_index()
    
    # Check comment count for each subreddit which might be needed for analysis later
    grouped["comment_count"] = (df.groupby("subreddit")["comment_id"].count().reset_index(drop=True))

    vector_feature_cols = [c for c in grouped.columns if c not in ("subreddit", "comment_count")]

    return grouped, vector_feature_cols

def run_kmeans(vectors, cols) -> tuple:
    X = vectors[cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []
    labels_per_k = {}

    for k in range(2, 15 + 1):
        # ref: https://www.geeksforgeeks.org/machine-learning/k-means-clustering-on-the-handwritten-digits-data-using-scikit-learn-in-python/
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels)
        results.append({
            "k": k,
            "inertia": round(inertia, 4),
            "silhouette_score": round(sil, 4),
        })
        labels_per_k[k] = labels

    summary = pd.DataFrame(results)

    labels_col = []
    # Get cluseter labels for each k value
    for k in summary["k"]:
        labels_col.append(str(labels_per_k[k].tolist()))

    summary["cluster_labels"] = labels_col

    return summary, vectors


def main():
    print("Loading phase1_derived_complexity_features.csv")
    df_feat = pd.read_csv("data/derived/phase1_derived_complexity_features.csv")

    # Convert bool columns to int so can aggregate numerically
    for col in df_feat.select_dtypes(include="bool").columns:
        df_feat[col] = df_feat[col].astype(int)
    
    print(f"Loaded {len(df_feat)} comments from {df_feat['subreddit'].nunique()} subreddits")

    vectors, vector_cols = build_subreddit_vectors(df_feat)

    summary, subreddit_vectors = run_kmeans(vectors, vector_cols)
    
    summary.to_csv("data/derived/kmeans_results.csv", index=False)
    subreddit_vectors.to_csv("data/derived/subreddit_vectors.csv", index=False)
    
    print(f"\nSaved kmeans_results.csv and subreddit_vectors.csv")


if __name__ == "__main__":
    main()
