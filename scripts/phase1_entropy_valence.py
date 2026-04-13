import numpy as np
import pandas as pd
from datasets import load_dataset #to load in dataset as per recommendation from hugging face

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
    "neutral",
]

# 27 emotions grouped into their valence buckets
POSITIVE_EMOTIONS = {
    "admiration", "amusement", "approval", "caring", "excitement",
    "gratitude", "joy", "love", "optimism", "pride", "relief",
}
NEGATIVE_EMOTIONS = {
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness",
}
AMBIGUOUS_EMOTIONS = {
    "confusion", "curiosity", "desire", "realization", "surprise",
}

# Shannon entropy calculation using the given equation
# ref: https://www.geeksforgeeks.org/machine-learning/entropy-in-information-theory/
def shannon_entropy(counts) -> float:
    """
    Compute Shannon entropy (base 2) from a raw count vector.

    If the total count is 0 (all raters selected only neutral),
    we return 0.0 by convention: unanimous neutrality = minimum complexity.

    Args:
        counts: 1-D array of non-negative integers

    Returns:
        Entropy in bits (float), 0.0 for degenerate zero-count distributions
    """
    # Use log base of 2 bc typically assume it
    # Values will range from 0 to log2(27) and 0 means only one emotion or netural emotion, higher values show many emotions are identified in the comment by the annotators
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    # Filter out zero probabilities to avoid log2(0) values
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    if entropy == 0.0: # prevent -0.0 from showing up for cleaner look on csv file
        entropy = 0.0  
    return float(round(entropy, 7)) # can adjust later


# Each unique comment row is used to calculate emotional_entropy and valence_mixing
def compute_comment_features(group) -> pd.Series:
    # group is df slice for all rater's annotations on a single comment
    # max(axis=0) to do OR across all raters' annotations for emotions identified
    union_vec = group[EMOTION_COLS[:-1]].max(axis=0).values.astype(float)  # all except neutral emotion

    # Emotional entropy across 27 emotions using shannon entropy
    emotional_entropy = shannon_entropy(union_vec)

    # Sum the union counts within each valence bucket to get shannon entropy across 3 valence buckets
    positive_count = sum(union_vec[EMOTION_COLS[:-1].index(e)] for e in POSITIVE_EMOTIONS)
    negative_count = sum(union_vec[EMOTION_COLS[:-1].index(e)] for e in NEGATIVE_EMOTIONS)
    ambiguous_count = sum(union_vec[EMOTION_COLS[:-1].index(e)] for e in AMBIGUOUS_EMOTIONS)

    valence_counts = np.array([positive_count, negative_count, ambiguous_count])
    valence_mixing = shannon_entropy(valence_counts)

    return pd.Series({
        "emotional_entropy": emotional_entropy,
        "valence_mixing": valence_mixing,
    })


def main():
    print("Loading GoEmotions raw dataset from HuggingFace")
    # Using raw file because it includes raters annotations for each comment
    dataset = load_dataset("google-research-datasets/go_emotions", "raw")

    # Data set is accessed in raw subset and train in the split (there is only the train in the split)
    # ref: https://huggingface.co/datasets/google-research-datasets/go_emotions/viewer/raw to see the data we are using
    df_raw = dataset["train"].to_pandas()
    print("Sucessfully loaded")
    
    # Rename 'id' to 'comment_id' for consistency with other files in feature engineering
    df_raw = df_raw.rename(columns={"id": "comment_id"})
    
    # Group by id, pk for dataset, and compute the two columns
    print("\nComputing per-comment emotional_entropy and valence_mixing")
    
    # Need reset_index() so the groupby key id is a regular column after being used
    features = (df_raw.groupby("comment_id", sort=False).apply(compute_comment_features).reset_index())
    print("Scuessfully computed")
    
    output_path = "comment_entropy_valence.csv"
    features.to_csv(output_path, index=False)
    print(f"\nSaved output to {output_path}")


if __name__ == "__main__":
    main()
