import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from cluster_profiles import CLUSTER_PROFILES

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

model_map = {
    "Logistic Regression": "logistic_regression_final",
    "Random Forest": "random_forest_final",
    "XGBoost": "xgboost_final",
    "LightGBM": "lightgbm_final"
}

st.set_page_config(
    page_title="Model Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data
def load_json(path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data/derived/comments_with_clusters_k4.csv")

@st.cache_data
def load_inference_data(k):
    """Build unified per-comment DataFrame for inference tab."""
    # Load features + true cluster + subreddit
    features = pd.read_csv(BASE_DIR / f"data/derived/comments_with_clusters_k{k}.csv").set_index("comment_id")
    
    # Load text + annotator labels
    text_lookup = pd.read_csv(BASE_DIR / "data/derived/test_set_text_lookup.csv").set_index("comment_id")
    
    # Inner-join: keeps only comments that have text (≈ all of them)
    df = features.join(text_lookup[["text", "annotator_1_emotions", "annotator_2_emotions", "annotator_3_emotions"]], how="inner")
    
    # Join each model's predictions; inner-join filters to current K's test set
    for display_name, fname in model_map.items():
        pred_path = RESULTS_DIR / f"k{k}/predictions/y_pred_{fname.replace('_final', '')}_k{k}.csv"
        pred = pd.read_csv(pred_path).set_index("comment_id").rename(columns={"cluster": f"pred_{fname}"})
        df = df.join(pred, how="inner")
    
    return df


def render_cluster_profile(k, cluster_id):
    """Render a cluster's identity panel: label + z-score bars + examples."""
    profile = CLUSTER_PROFILES[k][cluster_id]
    
    # Header — metric format gives us a free hover tooltip
    st.metric(
        label=f"Cluster {cluster_id}",
        value=profile["label"],
        help=profile.get("help", "")
    )
    
    # Z-score bar chart
    z_scores = profile["z_scores"]
    feature_labels = {
        "entropy": "Entropy",
        "valence_mix": "Valence mix",
        "multi_label": "Multi-label",
        "disagreement": "Disagreement",
    }
    
    fig, ax = plt.subplots(figsize=(4, 1.8))
    features = list(z_scores.keys())
    values = list(z_scores.values())
    colors = ["#10B981" if v > 0 else "#EF4444" for v in values]
    
    ax.barh(
        [feature_labels[f] for f in features],
        values,
        color=colors,
    )
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlim(-0.3, 0.3)
    # Add a shaded "neutral zone" band
    ax.axvspan(-0.1, 0.1, color="gray", alpha=0.15, zorder=0)
    ax.set_xlabel("z-score vs overall mean (gray band: |z| < 0.1, weak signal)", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.invert_yaxis()  # entropy at top
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    # Example subreddits
    examples = ", ".join(profile["examples"])
    st.caption(f"**Example subreddits:** {examples}")


header_left, header_right = st.columns([4, 1])
with header_left:
    st.title("Model Performance Dashboard (K=3 vs K=4)")
with header_right:
    st.write("")
    k = st.selectbox("K value", [3, 4])

models = {
    name: load_model(MODELS_DIR / f"{fname}_k{k}.pkl")
    for name, fname in model_map.items()
}

scaler = load_model(MODELS_DIR / f"scaler_k{k}.pkl")

results_path = RESULTS_DIR / f"k{k}"
pred_path = results_path / "predictions"

metrics_k3 = load_json(RESULTS_DIR / "k3" / "model_metrics.json")
metrics_k4 = load_json(RESULTS_DIR / "k4" / "model_metrics.json")

tab1, tab2, tab3, tab4 = st.tabs([
    "Metrics",
    "Confusion Matrices",
    "Feature Importance",
    "Inference"
])

with tab1:
    st.subheader(f"Model Performance (K={k})")

    metrics = load_json(results_path / "model_metrics.json")
    table_rows = []

    for model_name, vals in metrics.items():
        table_rows.append({
            "Model": model_name.replace("_", " ").title(),
            "Accuracy": vals["test"]["accuracy"],
            "Macro Precision": vals["test"]["macro_precision"],
            "Macro Recall": vals["test"]["macro_recall"],
            "Macro F1": vals["test"]["macro_f1"],
            "Weighted F1": vals["test"]["weighted_f1"]
        })

    df_table = pd.DataFrame(table_rows)

    st.dataframe(
        df_table.style.format({
            "Accuracy": "{:.3f}",
            "Macro Precision": "{:.3f}",
            "Macro Recall": "{:.3f}",
            "Macro F1": "{:.3f}",
            "Weighted F1": "{:.3f}",
        }),
        use_container_width=True
    )

    st.divider()
    
    st.subheader("Model Performance Comparison (K=3 vs K=4 with Random Baselines)")

    if metrics_k3 is None or metrics_k4 is None:
        st.error("Missing K=3 or K=4 metrics")
        st.stop()

    metrics_map = {3: metrics_k3, 4: metrics_k4}

    models_list = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    labels = ["LR", "RF", "XGB", "LGBM"]

    metrics_to_plot = ["macro_precision", "macro_recall", "macro_f1"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    x = np.arange(len(models_list))
    width = 0.35

    colors = {3: "#1f77b4", 4: "#ff7f0e"}
    bg_colors = ["#EAEAEA", "#D5E8D4", "#FFF2CC", "#F8CECC"]

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]

        for j in range(len(models_list)):
            ax.axvspan(j - 0.45, j + 0.45, color=bg_colors[j], alpha=0.35)

        # baseline lines
        ax.axhline(1/3, linestyle="--", color=colors[3], label="K=3 Random Baseline (0.33)")
        ax.axhline(1/4, linestyle="--", color=colors[4], label="K=4 Random Baseline (0.25)")

        for idx, k_val in enumerate([3, 4]):
            vals = [metrics_map[k_val][m]["test"][metric] for m in models_list]
            offset = (idx - 0.5) * width

            ax.bar(
                x + offset,
                vals,
                width=width,
                label=f"K={k_val} Result",
                color=colors[k_val]
            )

        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Model Type")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        ax.legend()

    axes[0].set_ylabel("Performance Score")

    st.pyplot(fig)

with tab2:
    st.subheader("Confusion Matrices")

    y_true = load_csv(pred_path / f"y_true_k{k}.csv")["cluster"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    i = 0
    for name in models:
        file = pred_path / f"y_pred_{name.lower().replace(' ', '_')}_k{k}.csv"

        if not file.exists():
            axes[i].axis("off")
            axes[i].set_title(f"{name} (missing)")
            i += 1
            continue

        y_pred = load_csv(file)["cluster"].values
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i],
            cbar=False
        )

        axes[i].set_title(name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

        i += 1

    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("Feature Importance")

    feature_names = scaler.feature_names_in_

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    models_to_plot = ["Random Forest", "XGBoost", "LightGBM", "Logistic Regression"]

    for i, name in enumerate(models_to_plot):
        ax = axes[i]

        model = models[name]

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_

        elif hasattr(model, "coef_"):
            importance = np.mean(np.abs(model.coef_), axis=0)

        else:
            ax.set_title(f"{name} (No importance available)")
            ax.axis("off")
            continue

        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False).head(10)

        sns.barplot(
            data=df_imp,
            x="importance",
            y="feature",
            ax=ax
        )

        ax.set_title(name)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="x", linestyle=":", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

with tab4:
    st.subheader("Live Inference")
    st.caption("Pick a Reddit comment from the held-out test set to see its emotional fingerprint and how each model classifies it.")

    inf_df = load_inference_data(k)
    
    # --- Comment selector ---
    def make_label(cid):
        text = inf_df.loc[cid, "text"]
        snippet = text[:80] + "..." if len(text) > 80 else text
        return f"{cid} — {snippet}"
    
    chosen_id = st.selectbox(
        "Pick a comment to inspect",
        inf_df.index.tolist(),
        format_func=make_label
    )
    row = inf_df.loc[chosen_id]
    
    st.divider()
    
    with st.container(border=True):
        left, right = st.columns([2, 2])

        with left:
            # --- Comment display ---
            st.markdown("### Comment")
            st.markdown(f"> {row['text']}")
            st.markdown(f"**Subreddit:** r/{row['subreddit']}  |  **True cluster:** {row['cluster']}")
        
        with right:
            # --- Annotators ---
            emotions = [row["annotator_1_emotions"], row["annotator_2_emotions"], row["annotator_3_emotions"]]
            all_agree = len(set(emotions)) == 1
            
            st.markdown("### Annotator labels " + ("✓ (all agreed)" if all_agree else "! (disagreement)"))
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Annotator 1**\n\n{emotions[0]}")
            c2.markdown(f"**Annotator 2**\n\n{emotions[1]}")
            c3.markdown(f"**Annotator 3**\n\n{emotions[2]}")


    with st.container():
        left, right = st.columns([2, 2])

        with left:
            with st.container(border=True):
                # --- Complexity features ---
                st.markdown("### Complexity Features")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Entropy", f"{row['emotional_entropy']:.2f}")
                m2.metric("Valence mixing", f"{row['valence_mixing']:.2f}")
                m3.metric("Multi-label count", f"{row['multi_label_count']:.0f}")
                m4.metric("Disagreement score", f"{row['annotator_disagreement_score']:.0f}")       
    
        with right:     
            # --- Model predictions ---
            with st.container(border=True):
                st.markdown("### Model Predictions")
                true_cluster = int(row["cluster"])
                cols = st.columns(4)
                for i, (display_name, fname) in enumerate(model_map.items()):
                    pred = int(row[f"pred_{fname}"])
                    correct = pred == true_cluster
                    with cols[i]:
                        st.metric(label=display_name, value=f"Cluster {pred}")
                        if correct:
                            st.markdown(
                                "<span style='color:#10B981;font-weight:600;'>✓ Correct</span>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                "<span style='color:#EF4444;font-weight:600;'>✗ Wrong</span>",
                                unsafe_allow_html=True
                            )

    with st.container(border=True):
        st.markdown("### Cluster Definitions")
        st.caption(f"How the K={k} clusters are characterized.")
        
        n_clusters = len(CLUSTER_PROFILES[k])
        cols = st.columns(n_clusters)
        for i, cluster_id in enumerate(sorted(CLUSTER_PROFILES[k].keys())):
            is_true = cluster_id == int(row["cluster"])
            with cols[i]:
                render_cluster_profile(k, cluster_id)
                if is_true:
                    st.markdown(
                        "<div style='background:#FEF3D9; padding:8px; border-radius:8px; "
                        "border:2px solid #F59E0B; text-align:center; "
                        "font-weight:600; color:#92400E; margin-bottom:8px;'>"
                        "⭐ True Cluster"
                        "</div>",
                        unsafe_allow_html=True,
                    )


    with st.expander("View all 27 emotion categories"):
        st.caption("Annotators select any subset of these per comment.")
        
        c1, c2, c3, c4 = st.columns(4)
        
        def category_box(title, emotions, count, bg_color, border_color):
            return f"""
            <div style='
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 12px 14px;
                height: 100%;
            '>
                <div style='font-weight: 600; margin-bottom: 8px; color: {border_color};'>
                    {title} ({count})
                </div>
                <div style='font-size: 1.05em; line-height: 1.6;'>
                    {emotions}
                </div>
            </div>
            """
        
        with c1:
            st.markdown(category_box(
                "Positive",
                "admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief",
                12,
                bg_color="#ACEDD3",
                border_color="#0E7050",
            ), unsafe_allow_html=True)
        
        with c2:
            st.markdown(category_box(
                "Negative",
                "anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness",
                11,
                bg_color="#F9CFD7",
                border_color="#EF4444",
            ), unsafe_allow_html=True)
        
        with c3:
            st.markdown(category_box(
                "Ambiguous",
                "confusion, curiosity, realization, surprise",
                4,
                bg_color="#F8E8B0",
                border_color="#F59E0B",
            ), unsafe_allow_html=True)
        
        with c4:
            st.markdown(category_box(
                "Neutral",
                "neutral",
                1,
                bg_color="#D7D7D7",
                border_color="#888888",
            ), unsafe_allow_html=True)