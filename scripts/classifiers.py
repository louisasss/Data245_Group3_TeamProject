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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

#set random seed value
random_seed = 42
np.random.seed(42)


def create_dummy_data():
    """
    Create temporary dummy data for testing
    """
    df = pd.DataFrame({
        'entropy': np.random.random(100),
        'annotator_disagreement': np.random.randint(0, 28, 100),
        'valence_mixing': np.random.random(100),
        'multi_label_count': np.random.randint(0, 28, 100),
        'example_very_unclear': np.random.randint(0, 2, 100),
        'cluster_label': np.random.randint(0, 4, 100)
    })
    return df

def load_data(filepath=None):
    """
    Load feature-engineered dataset from CSV
    
    Parameters:
    filepath: path to CSV file with engineered features
    
    Returns:
    DataFrame with features and cluster labels
    
    Columns expected from csv:
    - entropy
    - annotator_disagreement  
    - valence_mixing
    - multi_label_count
    - example_very_unclear
    - cluster_label
    """
    if filepath is None:
        filepath = Path('data') / 'derived' / 'feature_engineered_data.csv'
    df = pd.read_csv(filepath)
    return df


def load_and_prep_data(df):
    """"
    Splits data into 90% train, 10% test (stratified by cluster)
    """
    X = df.drop('cluster_label', axis=1)
    y = df['cluster_label']
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
        model.fit(X_fold_train, y_fold_train)
        # predict on current folds validation data
        y_pred = model.predict(X_fold_val)

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
    report = classification_report(y_true, y_pred, output_dict=True)
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


def main():
    # 1. load data
    try:
        df = load_data()
    except FileNotFoundError:
        print("Feature data not ready yet, using dummy data...")
        df = create_dummy_data()
    
    # prep data
    X_train, X_test, y_train, y_test = load_and_prep_data(df)

    # 2. scale features function
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

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
    
    # 4. Train final models on ALL training data
    print("\n" + "="*50)
    print("FINAL MODEL TRAINING")
    print("="*50)
    # retrain both models on full X_train_scaled
    # create fresh model instances
    final_lr = LogisticRegression(max_iter=1000, random_state=random_seed)
    final_rf = RandomForestClassifier(n_estimators=100, random_state=random_seed)

    # train on all 90% of training data
    final_lr.fit(X_train_scaled,  y_train)
    final_rf.fit(X_train_scaled,  y_train)
    print("Logistic Regression and Random Forest models trained on full training data set.")
    
    # 5. Evaluate on holdout test set
    print("\n" + "="*50)
    print("HOLDOUT TEST SET EVALUATION")
    print("="*50)
    # predict and evaluate on X_test_scaled
    final_lr_pred = final_lr.predict(X_test_scaled)
    final_rf_pred = final_rf.predict(X_test_scaled)

    final_lr_metrics = evaluate_model(y_test,final_lr_pred)
    final_rf_metrics = evaluate_model(y_test,final_rf_pred)

    print("\nLogistic Regression - Test Set Results:")
    for metric, value in final_lr_metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\nRandom Forest - Test Set Results:")
    for metric, value in final_rf_metrics.items():
        print(f"{metric}: {value:.6f}")

    # save models
    joblib.dump(final_lr, 'models/logistic_regression_final.pkl')
    joblib.dump(final_rf, 'models/random_forest_final.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\nModels saved!")

    # 6. Confusion matrices
    # logistic regression
    ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=final_lr_pred)
    plt.title("Logistic Regression - Test Set")
    plt.savefig('results/figures/confusion_matrix_lr.png')
    plt.close()
    # random forest
    ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=final_rf_pred)
    plt.title("Random Forest - Test Set")
    plt.savefig('results/figures/confusion_matrix_rf.png')
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
        }
    }

    with open('results/model_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    return


if __name__ == "__main__":
    main()