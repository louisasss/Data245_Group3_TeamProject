import pandas as pd
import os

# define variables for input csv file paths
entropy_valence = os.path.join("data", "derived", "comment_entropy_valence.csv")
multilabel_unclear = os.path.join("data", "derived", "comment_multilabel_unclear.csv")
disagreement_score = os.path.join("data", "derived", "disagreement_scores.csv")
# define variable for output file path
complexity_features = os.path.join("data", "derived", "comment_complexity_features.csv")

# read csvs
entropy_valence_df = pd.read_csv(entropy_valence)
multilabel_unclear_df = pd.read_csv(multilabel_unclear)
disagreement_score_df = pd.read_csv(disagreement_score)

# chain merge the three dataframes together
entropy_multilabel_merge_df = entropy_valence_df.merge(multilabel_unclear_df, how='outer', on='comment_id')
merged_df = entropy_multilabel_merge_df.merge(disagreement_score_df, how='outer', on='comment_id')

print("Count NaNs per column:", merged_df.isna().sum())

merged_df.to_csv(complexity_features, index=False)
print("Feature csvs merged successfully!")
print("Number of rows merged:", merged_df.shape[0])