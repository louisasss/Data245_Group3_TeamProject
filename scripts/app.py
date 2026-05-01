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

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

st.set_page_config(page_title="Model Dashboard", layout="wide")
st.title("Model Performance Dashboard (K=3 vs K=4)")


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


def split_xy(df):
    y = df["cluster"]
    X = df.drop(columns=["cluster", "comment_id", "subreddit", "unclear_fraction"], errors="ignore")
    return X, y


def align_features(X, scaler):
    return X.reindex(columns=scaler.feature_names_in_, fill_value=0)

df = load_data()

k = st.sidebar.selectbox("Select K", [3, 4])

model_map = {
    "Logistic Regression": "logistic_regression_final",
    "Random Forest": "random_forest_final",
    "XGBoost": "xgboost_final",
    "LightGBM": "lightgbm_final"
}

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

    y_true = load_csv(pred_path / f"y_true_k{k}.csv").values.ravel()

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

        y_pred = load_csv(file).values.ravel()
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
    st.subheader("Holdout Evaluation")

    model_choice = st.selectbox("Model", list(models.keys()))

    y_true_file = pred_path / f"y_true_k{k}.csv"
    y_pred_file = pred_path / f"y_pred_{model_choice.lower().replace(' ', '_')}_k{k}.csv"

    if not y_true_file.exists():
        st.error("Missing y_true file")
    elif not y_pred_file.exists():
        st.error("Missing prediction file")
    else:
        y_true = load_csv(y_true_file).values.ravel()
        y_pred = load_csv(y_pred_file).values.ravel()
        report_dict = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report_dict).T

        report_df = report_df.drop(
            index=[row for row in ["accuracy", "macro avg", "weighted avg"] if row in report_df.index]
        )

        report_df = report_df.rename(columns={"support": "samples"})
        report_df = report_df[["precision", "recall", "f1-score", "samples"]]
        report_df.index.name = "Cluster"

        st.subheader("Classification Summary (Per Cluster)")
        st.dataframe(report_df.style.format({
            "precision": "{:.3f}",
            "recall": "{:.3f}",
            "f1-score": "{:.3f}",
            "samples": "{:.0f}"
        }), use_container_width=True)

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(4.5, 4.5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            square=True,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"size": 10},
            ax=ax
        )

        ax.set_title("Holdout Confusion Matrix", fontsize=11)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)

        st.pyplot(fig, use_container_width=False)