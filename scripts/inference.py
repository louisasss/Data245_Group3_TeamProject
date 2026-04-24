import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_prepared_data(k: int, data_dir: Path) -> pd.DataFrame:
    csv_path = data_dir / f"comments_with_clusters_k{k}.csv"
    print(f"Looking for prepared dataset at: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded prepared dataset with shape: {df.shape}")
    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    constant_cols = [col for col in df.columns if col != "cluster" and df[col].nunique() <= 1]
    if constant_cols:
        print(f"Removing constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    return df


def recover_exact_holdout(df: pd.DataFrame):
    """
    Recreate the exact same split logic used in training.
    """
    if "cluster" not in df.columns:
        raise ValueError("Expected 'cluster' column in prepared dataset.")

    y = df["cluster"]

    train_idx, test_idx = train_test_split(
        df.index,
        train_size=0.9,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    return train_df, test_df


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["cluster", "comment_id", "unclear_fraction", "subreddit"]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop_cols)
    return X


def get_model_filename(model_name: str, k: int) -> str:
    mapping = {
        "logistic_regression": f"logistic_regression_final_k{k}.pkl",
        "random_forest": f"random_forest_final_k{k}.pkl",
        "xgboost": f"xgboost_final_k{k}.pkl",
        "lightgbm": f"lightgbm_final_k{k}.pkl",
    }
    if model_name not in mapping:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return mapping[model_name]


def load_artifacts(models_dir: Path, model_name: str, k: int):
    model_path = models_dir / get_model_filename(model_name, k)
    scaler_path = models_dir / f"scaler_k{k}.pkl"

    print(f"Looking for model at: {model_path}")
    print(f"Looking for scaler at: {scaler_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print(f"Loaded model: {type(model)}")
    print(f"Loaded scaler: {type(scaler)}")

    return model, scaler


def run_inference(k: int, model_name: str, data_dir: Path, models_dir: Path, output_dir: Path):
    print("=" * 60)
    print("HOLDOUT INFERENCE")
    print("=" * 60)

    df = load_prepared_data(k, data_dir)
    df = preprocess_features(df)

    print("\nRecovering exact holdout split...")
    train_df, test_df = recover_exact_holdout(df)
    print(f"Train rows: {len(train_df)}")
    print(f"Holdout rows: {len(test_df)}")

    print("\nHoldout cluster distribution:")
    print(test_df["cluster"].value_counts().sort_index())

    X_test = prepare_X(test_df)
    y_test = test_df["cluster"].copy()

    print("\nFeature columns used for inference:")
    print(list(X_test.columns))
    print(f"X_test shape: {X_test.shape}")

    model, scaler = load_artifacts(models_dir, model_name, k)

    print("\nScaling holdout features...")
    X_test_scaled = scaler.transform(X_test)

    print("Running predictions...")
    y_pred = model.predict(X_test_scaled)

    results_df = test_df.copy()
    results_df["predicted_cluster"] = y_pred
    results_df["correct"] = (results_df["cluster"] == results_df["predicted_cluster"]).astype(int)

    if hasattr(model, "predict_proba"):
        print("Adding probability columns...")
        prob_matrix = model.predict_proba(X_test_scaled)

        if hasattr(model, "classes_"):
            class_names = model.classes_
        else:
            class_names = [f"class_{i}" for i in range(prob_matrix.shape[1])]

        for i, cls in enumerate(class_names):
            results_df[f"prob_cluster_{cls}"] = prob_matrix[:, i]

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / f"holdout_predictions_{model_name}_k{k}.csv"
    holdout_only_path = output_dir / f"holdout_rows_k{k}.csv"

    results_df.to_csv(predictions_path, index=False)
    test_df.to_csv(holdout_only_path, index=False)

    print(f"\nSaved holdout predictions to: {predictions_path}")
    print(f"Saved recovered holdout rows to: {holdout_only_path}")

    print("\nPreview:")
    print(results_df.head())


def parse_args():
    parser = argparse.ArgumentParser(description="Recover exact holdout split and run inference.")
    parser.add_argument("--k", type=int, required=True, choices=[3, 4], help="Cluster setting, e.g. 3 or 4")
    parser.add_argument(
        "--model_name",
        required=True,
        choices=["logistic_regression", "random_forest", "xgboost", "lightgbm"],
        help="Which saved model to use",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("Starting inference script...")

    args = parse_args()
    project_root = get_project_root()

    data_dir = project_root / "data" / "derived"
    models_dir = project_root / "models"
    output_dir = project_root / "results" / "inference"

    print(f"Project root: {project_root}")
    print(f"Data dir: {data_dir}")
    print(f"Models dir: {models_dir}")
    print(f"Output dir: {output_dir}")

    run_inference(
        k=args.k,
        model_name=args.model_name,
        data_dir=data_dir,
        models_dir=models_dir,
        output_dir=output_dir,
    )