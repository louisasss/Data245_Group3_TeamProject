# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold  # For 5-fold CV
from sklearn.preprocessing import StandardScaler  # For scaling features
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone 

#set random seed value
random_seed = 42
np.random.seed(42)


def load_data(filepath=None):
    """
    Load feature-engineered dataset from CSV
    
    Parameters:
    filepath: path to CSV file with engineered features
    
    Returns:
    DataFrame with features and cluster labels
    
    Columns expected from csv:
    - emotional_entropy  
    - valence_mixing
    - multi_label_count
    - example_very_unclear
    - unclear_fraction
    - annotator_disagreement_score
    - cluster
    """
    if filepath is None:
        filepath = Path('data') / 'derived' / 'comments_with_clusters_k4.csv'
    df = pd.read_csv(filepath)
    return df

def preprocess_features(df):
    """
    Clean and transform dataframe into model-ready format
    """
    df = df.copy()

    # remove constant columns except target if present
    constant_cols = [col for col in df.columns if col != "cluster" and df[col].nunique() <= 1]
    if constant_cols:
        print(f"Removing constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    return df
def load_and_prep_data(df):
    """"
    Splits data into 90% train, 10% test (stratified by cluster)
    """
    drop_cols = ['cluster', 'comment_id', 'unclear_fraction', 'subreddit']
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(existing_drop_cols, axis=1)

    y = df['cluster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, stratify=y, random_state=random_seed)
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale features using standard scalar
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

def train_with_cv(model, X_train, y_train, n_folds=5):
    """
    Train input model with stratified 5 fold cross validation
    """
    # create stratified k fold object
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    #create empty list to store metrics from each fold
    fold_metrics = []

    # loop through folds
    for fold_num , (train_index, val_index) in enumerate(skf.split(X_train,y_train), 1):
        # split data into training and validation using index
        X_fold_train = X_train[train_index]
        X_fold_val = X_train[val_index]
        y_fold_train = y_train.iloc[train_index]
        y_fold_val = y_train.iloc[val_index]

        # train model on training folds
        # model.fit(X_fold_train, y_fold_train) # changed this to discard model after each fold and avoid data leakage
        fold_model = clone(model)
        fold_model.fit(X_fold_train, y_fold_train)
        # predict on current folds validation data
        y_pred = fold_model.predict(X_fold_val)

        # call evaluate_model function
        metrics = evaluate_model(y_fold_val, y_pred)
        fold_metrics.append(metrics)
    
    # calculate mean and std dev for each metric
    results = {}
    metric_keys = fold_metrics[0].keys()

    for metric_name in metric_keys:
        values = [fold[metric_name] for fold in fold_metrics]

        results[f'{metric_name}_mean'] = np.mean(values)
        results[f'{metric_name}_std'] = np.std(values)
        
    return results

def evaluate_model(y_true, y_pred):
    """
    Return metrics from predicted labels
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {
    'accuracy': report['accuracy'],
    'macro_precision': report['macro avg']['precision'],
    'macro_recall': report['macro avg']['recall'],
    'macro_f1': report['macro avg']['f1-score'],
    'weighted_precision': report['weighted avg']['precision'],
    'weighted_recall': report['weighted avg']['recall'],
    'weighted_f1': report['weighted avg']['f1-score']
    }
    return metrics

def create_metrics_table(lr_cv_results, lr_test_metrics, rf_cv_results, rf_test_metrics, xgb_cv_results, xgb_test_metrics, lgbm_cv_results, lgbm_test_metrics, output_path, k):
    """
    Create a formatted table of model metrics and save as PNG
    """
    import matplotlib.pyplot as plt
    
    # Prepare data for table
    metrics_data = [
        ['Metric', 'LR (CV)', 'LR (Test)', 'RF (CV)', 'RF (Test)', 'XGB (CV)', 'XGB (Test)', 'LGBM (CV)', 'LGBM (Test)'],
        ['Accuracy', 
         f"{lr_cv_results['accuracy_mean']:.3f} ± {lr_cv_results['accuracy_std']:.3f}",
         f"{lr_test_metrics['accuracy']:.3f}",
         f"{rf_cv_results['accuracy_mean']:.3f} ± {rf_cv_results['accuracy_std']:.3f}",
         f"{rf_test_metrics['accuracy']:.3f}",
         f"{xgb_cv_results['accuracy_mean']:.3f} ± {xgb_cv_results['accuracy_std']:.3f}",
         f"{xgb_test_metrics['accuracy']:.3f}",
         f"{lgbm_cv_results['accuracy_mean']:.3f} ± {lgbm_cv_results['accuracy_std']:.3f}",
         f"{lgbm_test_metrics['accuracy']:.3f}"],
        ['Macro Precision',
         f"{lr_cv_results['macro_precision_mean']:.3f} ± {lr_cv_results['macro_precision_std']:.3f}",
         f"{lr_test_metrics['macro_precision']:.3f}",
         f"{rf_cv_results['macro_precision_mean']:.3f} ± {rf_cv_results['macro_precision_std']:.3f}",
         f"{rf_test_metrics['macro_precision']:.3f}",
         f"{xgb_cv_results['macro_precision_mean']:.3f} ± {xgb_cv_results['macro_precision_std']:.3f}",
         f"{xgb_test_metrics['macro_precision']:.3f}",
         f"{lgbm_cv_results['macro_precision_mean']:.3f} ± {lgbm_cv_results['macro_precision_std']:.3f}",
         f"{lgbm_test_metrics['macro_precision']:.3f}"],
        ['Macro Recall',
         f"{lr_cv_results['macro_recall_mean']:.3f} ± {lr_cv_results['macro_recall_std']:.3f}",
         f"{lr_test_metrics['macro_recall']:.3f}",
         f"{rf_cv_results['macro_recall_mean']:.3f} ± {rf_cv_results['macro_recall_std']:.3f}",
         f"{rf_test_metrics['macro_recall']:.3f}",
         f"{xgb_cv_results['macro_recall_mean']:.3f} ± {xgb_cv_results['macro_recall_std']:.3f}",
         f"{xgb_test_metrics['macro_recall']:.3f}",
         f"{lgbm_cv_results['macro_recall_mean']:.3f} ± {lgbm_cv_results['macro_recall_std']:.3f}",
         f"{lgbm_test_metrics['macro_recall']:.3f}"],
        ['Macro F1',
         f"{lr_cv_results['macro_f1_mean']:.3f} ± {lr_cv_results['macro_f1_std']:.3f}",
         f"{lr_test_metrics['macro_f1']:.3f}",
         f"{rf_cv_results['macro_f1_mean']:.3f} ± {rf_cv_results['macro_f1_std']:.3f}",
         f"{rf_test_metrics['macro_f1']:.3f}",
         f"{xgb_cv_results['macro_f1_mean']:.3f} ± {xgb_cv_results['macro_f1_std']:.3f}",
         f"{xgb_test_metrics['macro_f1']:.3f}",
         f"{lgbm_cv_results['macro_f1_mean']:.3f} ± {lgbm_cv_results['macro_f1_std']:.3f}",
         f"{lgbm_test_metrics['macro_f1']:.3f}"],
        ['Weighted F1',
         f"{lr_cv_results['weighted_f1_mean']:.3f} ± {lr_cv_results['weighted_f1_std']:.3f}",
         f"{lr_test_metrics['weighted_f1']:.3f}",
         f"{rf_cv_results['weighted_f1_mean']:.3f} ± {rf_cv_results['weighted_f1_std']:.3f}",
         f"{rf_test_metrics['weighted_f1']:.3f}",
         f"{xgb_cv_results['weighted_f1_mean']:.3f} ± {xgb_cv_results['weighted_f1_std']:.3f}",
         f"{xgb_test_metrics['weighted_f1']:.3f}",
         f"{lgbm_cv_results['weighted_f1_mean']:.3f} ± {lgbm_cv_results['weighted_f1_std']:.3f}",
         f"{lgbm_test_metrics['weighted_f1']:.3f}"]
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 3))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=metrics_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.75)
    
    # Style header row
    for i in range(9):
        table[(0, i)].set_facecolor("#299216")
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(metrics_data)):
        for j in range(9):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')

    plt.title(f'Model Performance Comparison (K={k}):\nLogistic Regression vs Random Forest vs XGBoost vs LightGBM', fontsize=14, weight='bold', pad=10)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics table saved: {output_path}")

def create_combined_k_comparison_plot(all_results, output_path):
    metrics = ['macro_precision', 'macro_recall', 'macro_f1']
    models = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    model_labels = ['LR', 'RF', 'XGB', 'LGBM']
    
    available_ks = sorted(all_results.keys())
    if not available_ks:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors = {3: '#1f77b4', 4: '#ff7f0e'}  # Blue for K=3, Orange for K=4
    baselines = {3: 0.333, 4: 0.25}        # Random chance baselines
    
    width = 0.8 / len(available_ks)
    x = np.arange(len(models))

    bg_colors = ['#EAEAEA', '#D5E8D4', '#FFF2CC', '#F8CECC']

    for i, metric in enumerate(metrics):
        # Add background zones for each model type
        for idx in range(len(models)):
            axes[i].axvspan(idx - 0.45, idx + 0.45, color=bg_colors[idx], alpha=0.5, zorder=1)

        # Plot the horizontal baseline lines
        if 3 in available_ks:
            axes[i].axhline(y=baselines[3], color=colors[3], linestyle='--', 
                            linewidth=1.5, label='K=3 Random Baseline (0.33)', zorder=2)
        if 4 in available_ks:
            axes[i].axhline(y=baselines[4], color=colors[4], linestyle='--', 
                            linewidth=1.5, label='K=4 Random Baseline (0.25)', zorder=2)

        # Plot bars
        for j, k_val in enumerate(available_ks):
            offset = (j - (len(available_ks)-1)/2) * width
            vals = [all_results[k_val][m]['test'][metric] for m in models]
            axes[i].bar(x + offset, vals, width, label=f'K={k_val} Result', 
                        color=colors.get(k_val, None), alpha=1.0, zorder=3)

        # Formatting
        axes[i].set_title(metric.replace('_', ' ').title(), fontsize=14, weight='bold')
        axes[i].set_xlabel('Model Type', fontsize=12, weight='semibold')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(model_labels)
        axes[i].set_ylim(0, 1.1)
        
        axes[i].grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
        
        if i == 0:
            axes[i].set_ylabel('Performance Score', fontsize=12, weight='semibold')
        
        # Adjust legend to show baselines clearly
        axes[i].legend(loc='upper right', fontsize=9, frameon=True, facecolor='white', framealpha=0.9)

    plt.suptitle('Model Performance Comparison: K=3 vs K=4 with Random Baselines', fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Initialize dictionary to store results for both K values
    all_k_results = {}
    # Test both K=3 and K=4
    for k in [3, 4]:
        print(f"\n{'='*60}")
        print(f"RUNNING MODELS FOR K={k} CLUSTERS")
        print(f"{'='*60}\n")
        
        # Set filepath based on k
        filepath = Path('data') / 'derived' / f'comments_with_clusters_k{k}.csv'
        
        # Create output directory
        Path(f'results/k{k}').mkdir(parents=True, exist_ok=True)
        
        # Load data
        try:
            df = load_data(filepath)
        except FileNotFoundError:
            print(f"File not found: {filepath}, skipping K={k}")
            continue

        df = preprocess_features(df)

        # prep data
        X_train, X_test, y_train, y_test = load_and_prep_data(df)

        # 2. scale features function
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        print("\nCluster distribution (TRAIN):")
        print(y_train.value_counts())

        print("\nCluster distribution (TEST):")
        print(y_test.value_counts())

        # 3. cross-validation on both models
        print("\n" + "="*50)
        print("5-Fold Cross-Validation")
        print("="*50)

        # Logistic Regression CV
        print("\nLogistic Regression - 5-Fold CV:")
        lr_model = LogisticRegression(max_iter=1000, random_state=random_seed)
        lr_cv_results = train_with_cv(lr_model, X_train_scaled, y_train)
        print("Logistic Regression CV metrics:")
        for metric, value in lr_cv_results.items():
            print(f"  {metric}: {value:.6f}")

        # Random Forest CV
        print("\nRandom Forest - 5-Fold CV:")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=random_seed)
        rf_cv_results = train_with_cv(rf_model, X_train_scaled, y_train)
        print("Random Forest CV metrics:")
        for metric, value in rf_cv_results.items():
            print(f"  {metric}: {value:.6f}")

        # XGBoost CV
        print("\nXGBoost - 5-Fold CV:")
        xgb_model = XGBClassifier(n_estimators=100, random_state=random_seed, eval_metric='mlogloss')
        xgb_cv_results = train_with_cv(xgb_model, X_train_scaled, y_train)
        print("XGBoost CV metrics:")
        for metric, value in xgb_cv_results.items():
            print(f"  {metric}: {value:.6f}")

        # LightGBM CV
        print("\nLightGBM - 5-Fold CV:")
        lgbm_model = build_lightgbm_model(y_train)
        lgbm_cv_results = train_with_cv(lgbm_model, X_train_scaled, y_train)
        print("LightGBM CV metrics:")
        for metric, value in lgbm_cv_results.items():
            print(f"  {metric}: {value:.6f}")
        # 4. Train final models on ALL training data
        print("\n" + "="*50)
        print("FINAL MODEL TRAINING")
        print("="*50)
        # retrain both models on full X_train_scaled
        # create fresh model instances
        final_lr = LogisticRegression(max_iter=1000, random_state=random_seed)
        final_rf = RandomForestClassifier(n_estimators=100, random_state=random_seed)
        final_xgb = XGBClassifier(n_estimators=100, random_state=random_seed, eval_metric='mlogloss')
        final_lgbm = build_lightgbm_model(y_train)
        # train on all 90% of training data
        final_lr.fit(X_train_scaled,  y_train)
        final_rf.fit(X_train_scaled,  y_train)
        final_xgb.fit(X_train_scaled, y_train)
        final_lgbm.fit(X_train_scaled, y_train)
        print("Logistic Regression, Random Forest, XGBoost, and LightGBM models trained on full training data set.")

        # 5. Evaluate on holdout test set
        print("\n" + "="*50)
        print("HOLDOUT TEST SET EVALUATION")
        print("="*50)
        # predict and evaluate on X_test_scaled
        final_lr_pred = final_lr.predict(X_test_scaled)
        final_rf_pred = final_rf.predict(X_test_scaled)
        final_xgb_pred = final_xgb.predict(X_test_scaled)
        final_lgbm_pred = final_lgbm.predict(X_test_scaled)
        
        # to save predicted and ground truth 
        pred_dir = Path(f'results/k{k}/predictions')
        pred_dir.mkdir(parents=True, exist_ok=True)

        # true labels (only once per k)
        pd.DataFrame(y_test).to_csv(pred_dir / f'y_true_k{k}.csv', index=False)

        # predictions per model
        pd.DataFrame(final_lr_pred).to_csv(pred_dir / f'y_pred_logistic_regression_k{k}.csv', index=False)
        pd.DataFrame(final_rf_pred).to_csv(pred_dir / f'y_pred_random_forest_k{k}.csv', index=False)
        pd.DataFrame(final_xgb_pred).to_csv(pred_dir / f'y_pred_xgboost_k{k}.csv', index=False)
        pd.DataFrame(final_lgbm_pred).to_csv(pred_dir / f'y_pred_lightgbm_k{k}.csv', index=False)

        print(f"Saved predictions for K={k}")

        final_lr_metrics = evaluate_model(y_test, final_lr_pred)
        final_rf_metrics = evaluate_model(y_test, final_rf_pred)
        final_xgb_metrics = evaluate_model(y_test, final_xgb_pred)
        final_lgbm_metrics = evaluate_model(y_test, final_lgbm_pred)

        print("\nLogistic Regression - Test Set Results:")
        for metric, value in final_lr_metrics.items():
            print(f"{metric}: {value:.6f}")

        print("\nRandom Forest - Test Set Results:")
        for metric, value in final_rf_metrics.items():
            print(f"{metric}: {value:.6f}")

        print("\nXGBoost - Test Set Results:")
        for metric, value in final_xgb_metrics.items():
            print(f"{metric}: {value:.6f}")

        print("\nLightGBM - Test Set Results:")
        for metric, value in final_lgbm_metrics.items():
            print(f"{metric}: {value:.6f}")

        # save models
        joblib.dump(final_lr, f'models/logistic_regression_final_k{k}.pkl')
        joblib.dump(final_rf, f'models/random_forest_final_k{k}.pkl')
        joblib.dump(final_xgb, f'models/xgboost_final_k{k}.pkl')
        joblib.dump(scaler, f'models/scaler_k{k}.pkl')
        joblib.dump(final_lgbm, f'models/lightgbm_final_k{k}.pkl')

        print("\nModels saved!")

        # 6. Confusion matrices
        # logistic regression
        ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=final_lr_pred)
        plt.title("Logistic Regression - Test Set")
        plt.savefig(f'results/k{k}/confusion_matrix_lr.png')
        plt.close()
        # random forest
        ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=final_rf_pred)
        plt.title("Random Forest - Test Set")
        plt.savefig(f'results/k{k}/confusion_matrix_rf.png')
        plt.close()
        # xgboost
        ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=final_xgb_pred)
        plt.title("XGBoost - Test Set")
        plt.savefig(f'results/k{k}/confusion_matrix_xgb.png')
        plt.close()

        # lightgbm
        ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=final_lgbm_pred)
        plt.title("LightGBM - Test Set")
        plt.savefig(f'results/k{k}/confusion_matrix_lgbm.png')
        plt.close()

        # save results
        results = {
            'logistic_regression': {
                'cv': lr_cv_results,
                'test': final_lr_metrics
            },
            'random_forest': {
                'cv': rf_cv_results,
                'test': final_rf_metrics
            },
            'xgboost': {
                'cv': xgb_cv_results,
                'test': final_xgb_metrics
            },
            'lightgbm': {
                'cv': lgbm_cv_results,
                'test': final_lgbm_metrics
            }
        }

        # Save to the dictionary using the current k
        all_k_results[k] = results 
        print(f"Successfully stored results for K={k}")

        with open(f'results/k{k}/model_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Create metrics table
        create_metrics_table(
            lr_cv_results, final_lr_metrics,
            rf_cv_results, final_rf_metrics,
            xgb_cv_results, final_xgb_metrics,
            lgbm_cv_results, final_lgbm_metrics,
            f'results/k{k}/metrics_table.png',
            k
        )

        # Generate the combined plot after the loop finishes
        print("\n" + "="*50)
        print("GENERATING COMBINED K-COMPARISON CHART")
        print("="*50)
        
        # Create a global figures directory if it doesn't exist
        Path('results/figures').mkdir(parents=True, exist_ok=True)
        
        if all_k_results:
            create_combined_k_comparison_plot(all_k_results, 'results/figures/combined_k_comparison.png')
        else:
            print("No K-results were stored in all_k_results. Check your loop logic.")
        
    return


if __name__ == "__main__":
    main()
