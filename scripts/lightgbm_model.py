# lightgbm.py
# LightGBM classifier script matching the structure/output style of classifiers.py

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


# set random seed value
random_seed = 42
np.random.seed(random_seed)

# default data path
DATA_PATH = Path("data") / "derived" / "comments_with_clusters_k3.csv"


def create_dummy_data():
    """
    Create temporary dummy data for testing
    """
    df = pd.DataFrame({
        "emotional_entropy": np.random.random(100),
        "valence_mixing": np.random.random(100),
        "multi_label_count": np.random.randint(0, 28, 100),
        "example_very_unclear": np.random.randint(0, 2, 100),
        "annotator_disagreement_score": np.random.randint(0, 28, 100),
        "subreddit": np.random.choice(["a", "b", "c", "d"], 100),
        "cluster": np.random.randint(0, 3, 100),
    })
    return df


def load_data(filepath=None):
    """
    Load feature-engineered dataset from CSV
    """
    if filepath is None:
        filepath = DATA_PATH
    df = pd.read_csv(filepath)
    print(f"Loaded data from: {filepath}")
    print(f"Initial shape: {df.shape}")
    return df


def preprocess_features(df):
    """
    Clean and transform dataframe into model-ready format
    """
    df = df.copy()

    # encode subreddit if present
    if "subreddit" in df.columns:
        print("Encoding 'subreddit'...")
        le = LabelEncoder()
        df["subreddit"] = le.fit_transform(df["subreddit"].astype(str))

    # remove constant columns except target if present
    constant_cols = [col for col in df.columns if col != "cluster" and df[col].nunique() <= 1]
    if constant_cols:
        print(f"Removing constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

        print(f"Post-preprocessing shape: {df.shape}")
    return df


def load_and_prep_data(df):
    """
    Splits data into 90% train, 10% test (stratified by cluster)
    """
    drop_cols = ["cluster", "comment_id", "unclear_fraction"]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(existing_drop_cols, axis=1)
    y = df["cluster"]

    print(f"Final features used: {X.columns.tolist()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.9,
        stratify=y,
        random_state=random_seed
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale features using standard scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def build_lightgbm_model(y_train=None):
    """
    Build LightGBM multiclass classifier
    """
    num_classes = len(np.unique(y_train)) if y_train is not None else 3

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        metric="multi_logloss",
        boosting_type="gbdt",
        random_state=random_seed,
        verbosity=-1,
        n_estimators=100
    )
    return model


def train_with_cv(model_builder, X_train, y_train, n_folds=5):
    """
    Train LightGBM with stratified 5-fold cross validation
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    fold_metrics = []

    for fold_num, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
        X_fold_train = X_train[train_index]
        X_fold_val = X_train[val_index]
        y_fold_train = y_train.iloc[train_index]
        y_fold_val = y_train.iloc[val_index]

        model = model_builder(y_fold_train)
        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)

        metrics = evaluate_model(y_fold_val, y_pred)
        fold_metrics.append(metrics)

    results = {}
    metric_keys = fold_metrics[0].keys()

    for metric_name in metric_keys:
        values = [fold[metric_name] for fold in fold_metrics]
        results[f"{metric_name}_mean"] = np.mean(values)
        results[f"{metric_name}_std"] = np.std(values)

    return results


def evaluate_model(y_true, y_pred):
    """
    Return metrics from predicted labels
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"]
    }
    return metrics


def create_metrics_table(lgbm_cv_results, lgbm_test_metrics, output_path):
    """
    Create a formatted table of model metrics and save as PNG
    """
    metrics_data = [
        ["Metric", "LightGBM (CV)", "LightGBM (Test)"],
        ["Accuracy",
         f"{lgbm_cv_results['accuracy_mean']:.3f} ± {lgbm_cv_results['accuracy_std']:.3f}",
         f"{lgbm_test_metrics['accuracy']:.3f}"],
        ["Macro Precision",
         f"{lgbm_cv_results['macro_precision_mean']:.3f} ± {lgbm_cv_results['macro_precision_std']:.3f}",
         f"{lgbm_test_metrics['macro_precision']:.3f}"],
        ["Macro Recall",
         f"{lgbm_cv_results['macro_recall_mean']:.3f} ± {lgbm_cv_results['macro_recall_std']:.3f}",
         f"{lgbm_test_metrics['macro_recall']:.3f}"],
        ["Macro F1",
         f"{lgbm_cv_results['macro_f1_mean']:.3f} ± {lgbm_cv_results['macro_f1_std']:.3f}",
         f"{lgbm_test_metrics['macro_f1']:.3f}"],
        ["Weighted F1",
         f"{lgbm_cv_results['weighted_f1_mean']:.3f} ± {lgbm_cv_results['weighted_f1_std']:.3f}",
         f"{lgbm_test_metrics['weighted_f1']:.3f}"]
    ]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(cellText=metrics_data, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.75)

    for i in range(3):
        table[(0, i)].set_facecolor("#299216")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for i in range(1, len(metrics_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#E7E6E6")

    plt.title("Model Performance Comparison:\nLightGBM", fontsize=14, weight="bold", pad=10)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Metrics table saved: {output_path}")


def main():
    # create output directories
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    # 1. load data
    try:
        df = load_data()
        df['cluster'] = df['cluster'].replace({3: 2})
    except FileNotFoundError:
        print("Feature data not ready yet, using dummy data...")
        df = create_dummy_data()

    # 2. preprocess
    df = preprocess_features(df)

    # 3. prep data
    X_train, X_test, y_train, y_test = load_and_prep_data(df)

    # 4. scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 5. cross-validation
    print("\n" + "=" * 50)
    print("5-Fold Cross-Validation")
    print("=" * 50)

    print("\nLightGBM - 5-Fold CV:")
    lgbm_cv_results = train_with_cv(build_lightgbm_model, X_train_scaled, y_train)
    print("LightGBM CV metrics:")
    for metric, value in lgbm_cv_results.items():
        print(f"  {metric}: {value:.6f}")

    # 6. train final model on all training data
    print("\n" + "=" * 50)
    print("FINAL MODEL TRAINING")
    print("=" * 50)

    final_lgbm = build_lightgbm_model(y_train)
    final_lgbm.fit(X_train_scaled, y_train)
    print("LightGBM model trained on full training data set.")

    # 7. evaluate on holdout test set
    print("\n" + "=" * 50)
    print("HOLDOUT TEST SET EVALUATION")
    print("=" * 50)

    final_lgbm_pred = final_lgbm.predict(X_test_scaled)
    final_lgbm_metrics = evaluate_model(y_test, final_lgbm_pred)

    print("\nLightGBM - Test Set Results:")
    for metric, value in final_lgbm_metrics.items():
        print(f"{metric}: {value:.6f}")

    # 8. save model and scaler
    joblib.dump(final_lgbm, "models/lightgbm_final.pkl")
    joblib.dump(scaler, "models/lightgbm_scaler.pkl")
    print("\nModel saved!")

    # 9. confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=final_lgbm_pred)
    plt.title("LightGBM - Test Set")
    plt.savefig("results/figures/lightgbm_confusion_matrix.png")
    plt.close()

    # 10. save results json
    results = {
        "lightgbm": {
            "cv": lgbm_cv_results,
            "test": final_lgbm_metrics
        }
    }

    with open("results/lightgbm_model_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # 11. create metrics table
    create_metrics_table(
        lgbm_cv_results,
        final_lgbm_metrics,
        "results/figures/lightgbm_metrics_table.png"
    )


if __name__ == "__main__":
    main()