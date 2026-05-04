import pandas as pd
from datasets import load_dataset
from pathlib import Path


# 1. Get target comment_ids from both test sets
ids_k3 = pd.read_csv("results/k3/predictions/y_true_k3.csv")["comment_id"]
ids_k4 = pd.read_csv("results/k4/predictions/y_true_k4.csv")["comment_id"]
target_ids = set(ids_k3) | set(ids_k4)
print(f"Target comment_ids: {len(target_ids)}")


# 2. Load raw GoEmotions from HuggingFace
ds = load_dataset("google-research-datasets/go_emotions", "raw", split="train")
df = ds.to_pandas()
print(f"Loaded {len(df)} raw rows from HuggingFace")


# 3. Filter to target comment_ids
df = df[df["id"].isin(target_ids)]
print(f"Filtered to {len(df)} rows ({df['id'].nunique()} unique comments)")


# 4. Identify emotion columns (everything that isn't metadata)
meta_cols = {"text", "id", "author", "subreddit", "link_id", "parent_id",
             "created_utc", "rater_id", "example_very_unclear"}
emotion_cols = [c for c in df.columns if c not in meta_cols]


# 5. Build emotion string per row
def row_to_emotion_string(row):
    active = [emo for emo in emotion_cols if row[emo] == 1]
    return ", ".join(active) if active else "neutral"

df["emotion_string"] = df.apply(row_to_emotion_string, axis=1)


# 6. Collapse 3 annotator rows per comment into 1 row
grouped = df.groupby("id").agg({
    "text": "first",
    "subreddit": "first",
    "emotion_string": list,
})


# 7. Pad emotion lists to length 3 (handles edge case of <3 annotators)
grouped["emotion_string"] = grouped["emotion_string"].apply(
    lambda lst: (lst + ["", "", ""])[:3]
)


# 8. Split the list into 3 separate columns
emotions_split = pd.DataFrame(
    grouped["emotion_string"].tolist(),
    index=grouped.index,
    columns=["annotator_1_emotions", "annotator_2_emotions", "annotator_3_emotions"]
)
grouped = grouped.drop(columns=["emotion_string"]).join(emotions_split)


# 9. Rename index for consistency with other files
grouped.index.name = "comment_id"


# 10. Save
Path("data/derived").mkdir(parents=True, exist_ok=True)
out_path = "data/derived/test_set_text_lookup.csv"
grouped.to_csv(out_path, index=True)
print(f"Saved {len(grouped)} rows to {out_path}")