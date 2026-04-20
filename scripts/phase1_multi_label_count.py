
"""
Phase 1: Multi-label Count and Unclear Fraction
Calculates multi_label_count, example_very_unclear, and unclear_fraction per comment
"""

import pandas as pd
from pathlib import Path

def main():
    # Load raw data
    print("Loading raw GoEmotions data...")
    df = pd.read_parquet("hf://datasets/google-research-datasets/go_emotions/raw/train-00000-of-00001.parquet")
    
    # Define emotion columns
    non_emotion_cols = [
        'text', 'id', 'author', 'subreddit', 'link_id',
        'parent_id', 'created_utc', 'rater_id', 'example_very_unclear', 'neutral'
    ]
    emotion_cols = [col for col in df.columns if col not in non_emotion_cols]
    print(f"Found {len(emotion_cols)} emotion columns")
    
    # Aggregate to comment level (max across 3 raters)
    print("Aggregating to comment level...")
    agg_dict = {col: 'max' for col in emotion_cols}
    agg_dict['example_very_unclear'] = 'max'
    
    df_clean = (
        df.groupby('id', as_index=False)
          .agg(agg_dict)
          .rename(columns={'id': 'comment_id'})
    )
    
    # Compute multi-label count
    df_clean['multi_label_count'] = df_clean[emotion_cols].sum(axis=1).astype(int)
    
    # Compute unclear_fraction (mean across 3 raters)
    unclear_fraction = (
        df.groupby('id')['example_very_unclear']
          .mean()
          .reset_index()
          .rename(columns={'id': 'comment_id', 'example_very_unclear': 'unclear_fraction'})
    )
    
    df_clean = df_clean.merge(unclear_fraction, on='comment_id')
    
    # Create final feature dataset
    feature_df = df_clean[
        ['comment_id', 'multi_label_count', 'example_very_unclear', 'unclear_fraction']
    ].copy()
    
    # Save output
    output_path = Path("data") / "derived" / "phase1_comment_multilabel_unclear.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    
    print(f"\nSaved to: {output_path}")
    print(f"Shape: {feature_df.shape}")
    print(f"Unique comments: {feature_df['comment_id'].nunique()}")
    print("\nMulti-label count stats:")
    print(feature_df['multi_label_count'].describe())
    
    return feature_df

if __name__ == "__main__":
    main()