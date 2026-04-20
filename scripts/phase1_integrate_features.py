import pandas as pd
import os
from datasets import load_dataset

# getting raw data to get subreddit info for each comment_id
dataset = load_dataset("google-research-datasets/go_emotions", "raw")
df_raw = dataset["train"].to_pandas()

# define variables for input csv file paths
entropy_valence = os.path.join("data", "derived", "phase1_comment_entropy_valence.csv")
multilabel_unclear = os.path.join("data", "derived", "phase1_comment_multilabel_unclear.csv")
disagreement_score = os.path.join("data", "derived", "phase1_disagreement_scores.csv")
# define variable for output file path
complexity_features = os.path.join("data", "derived", "phase1_derived_complexity_features.csv")

# read csvs
entropy_valence_df = pd.read_csv(entropy_valence)
multilabel_unclear_df = pd.read_csv(multilabel_unclear)
disagreement_score_df = pd.read_csv(disagreement_score)

# chain merge the three dataframes together
entropy_multilabel_merge_df = entropy_valence_df.merge(multilabel_unclear_df, how='outer', on='comment_id')
merged_df = entropy_multilabel_merge_df.merge(disagreement_score_df, how='outer', on='comment_id')

# getting subreddit info for each comment_id to merge onto the features dataframe
df_raw = df_raw.rename(columns={"id": "comment_id"})
subreddit_df = df_raw[["comment_id", "subreddit"]].drop_duplicates()

merged_df = merged_df.merge(subreddit_df, how='left', on='comment_id')

merged_df.to_csv(complexity_features, index=False)
print("Feature csvs merged successfully!")
print("Number of rows merged:", merged_df.shape[0])