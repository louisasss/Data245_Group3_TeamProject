import numpy as np
import pandas as pd
from datasets import load_dataset #to load in dataset as per recommendation from hugging face

# Emotions dataset from Hugging Face (27)
EMOTION_COLS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
]

def disagreement_score(comment_group):
    # Compute per-comment annotator disagreement score (|union − intersection| / 27)
    # union = number of emotions labeled by at least one annotator
    # intersection = number of emotions labeled by all annotators
    # disagreement score = (union - intersection) / 27
    matrix = comment_group[EMOTION_COLS].values.astype(float)
    # axis=0 means to go down the rows and >= 1 converts True/False and astype converts back to float
    union = (matrix.sum(axis=0) >= 1).astype(float)     # union is at least one rater said yes
    intersection = (matrix.sum(axis=0) == len(matrix)).astype(float)  # intersection is all raters said yes
    disagreement_score = (union - intersection).sum() / 27  # divide by 27 for 27 emotions

    return disagreement_score

def main():
    print("Loading GoEmotions raw dataset from HuggingFace")
    # Using raw file because it includes raters annotations for each comment
    dataset = load_dataset("google-research-datasets/go_emotions", "raw")
    df_raw = dataset["train"].to_pandas()
    print("Successfully loaded")

    # Rename 'id' to 'comment_id' for consistency with other files in feature engineering
    df_raw = df_raw.rename(columns={"id": "comment_id"})

    print("\nComputing per-comment annotator_disagreement_score and saving to disagreement_scores.csv")
    # reset_index needed to make comment_id move it back to a normal column instead of index after groupby
    features = (df_raw.groupby("comment_id", sort=False).apply(disagreement_score).reset_index())
    features.columns = ["comment_id", "annotator_disagreement_score"]
    print("Successfully computed")

    # Save the results to a CSV file
    output_path = "disagreement_scores.csv"
    features.to_csv(output_path, index=False)
    print(f"Saved disagreement scores to {output_path}")

if __name__ == "__main__":
    main()


