import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

# Load dataset
df = pd.read_csv("comment_multilabel_unclear.csv")

# Clip rare label counts to ensure valid stratification
df["multi_label_count_strat"] = df["multi_label_count"].clip(upper=5)

# 1. Stratified Train / Test Split (10% holdout)
train_df, test_df = train_test_split(
    df,
    test_size=0.10,
    stratify=df["multi_label_count_strat"],
    random_state=42
)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# 2. Stratified 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_df, train_df["multi_label_count_strat"]), start=1
):
    fold_train = train_df.iloc[train_idx]
    fold_val = train_df.iloc[val_idx]
    print(f"Fold {fold}: train={fold_train.shape}, val={fold_val.shape}")

# Drop helper column before saving or modeling
train_df = train_df.drop(columns=["multi_label_count_strat"])
test_df = test_df.drop(columns=["multi_label_count_strat"])
