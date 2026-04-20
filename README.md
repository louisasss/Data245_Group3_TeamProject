# From Noise to Signal: Leveraging Annotator Disagreement to Measure Emotional Complexity Across Online Communities

**DATA 245 Machine Learning | Spring 2026 | San José State University**

## Project Overview

This project reframes annotator disagreement in emotion labeling as a meaningful signal of emotional complexity rather than noise. Using the GoEmotions dataset (58,000 Reddit comments across 483 subreddits), we ask: **Can subreddit community clusters be predicted from emotional complexity features derived from annotator disagreement patterns?**

### Key Findings
- **36% classification accuracy** (44% above 25% random baseline)
- Demonstrates that emotional complexity carries predictive signal about community type
- Suggests communities have distinct emotional profiles beyond topic alone

---

## Pipeline Architecture

```
Phase 1: Feature Engineering (Comment-Level)
├── phase1_multi_label_count.py          → multi-label counts & unclear flags
├── phase1_ann_disagree_score.py         → annotator disagreement scoring
├── phase1_entropy_valence.py            → emotional & valence entropy
└── phase1_integrate_features.py         → merge into single feature set

Phase 2: Clustering (Subreddit-Level)
├── phase2_subreddit_vectors_and_kmeans_clustering.py  → aggregate & cluster
├── phase2_clustering_analysis.py        → evaluate K, generate plots
└── phase2_merge_clusters.py             → assign cluster labels to comments

Phase 3: Classification (Supervised Learning)
└── classifiers.py                       → train LR & RF, evaluate performance
```

**Output:** Trained models achieve ~36% accuracy predicting K=4 community clusters from 5 emotional complexity features.

---

## Repository Structure

```
Data245_Group3_TeamProject/
│
├── README.md                  ← you are here
├── requirements.txt           ← python dependencies
├── .gitignore
│
├── data/
│   └── derived/               ← pipeline outputs 
│
├── scripts/                   ← pipeline scripts (phase1_*, phase2_*, classifiers.py)
│
├── notebooks/
│   ├── basic_EDA.ipynb        ← exploratory analysis
│   └── archive/               ← converted notebooks (now .py scripts)
│
├── models/                    ← trained model files (.pkl) [gitignored]
│
└── results/
    ├── figures/               ← confusion matrices, elbow/silhouette plots
    └── model_metrics.json     ← CV & test set performance metrics
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/louisasss/Data245_Group3_TeamProject.git
cd Data245_Group3_TeamProject
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**Note:** The raw GoEmotions dataset is NOT stored in this repo. It is loaded directly from Hugging Face:
```python
from datasets import load_dataset
ds = load_dataset("google-research-datasets/go_emotions", "raw")
```

---

## Running the Pipeline

### Full Pipeline (End-to-End)
```bash
# Phase 1: Feature Engineering
python scripts/phase1_multi_label_count.py
python scripts/phase1_ann_disagree_score.py
python scripts/phase1_entropy_valence.py
python scripts/phase1_integrate_features.py

# Phase 2: Clustering
python scripts/phase2_subreddit_vectors_and_kmeans_clustering.py
python scripts/phase2_clustering_analysis.py
python scripts/phase2_merge_clusters.py

# Phase 3: Classification
python scripts/classifiers.py
```

---

## Features

### Emotional Complexity Metrics (5 features)
1. **Emotional Entropy** - Shannon entropy across 27 emotion categories
2. **Valence Mixing** - Shannon entropy across positive/negative/ambiguous valence
3. **Multi-Label Count** - Number of emotions assigned per comment (union across 3 annotators)
4. **Annotator Disagreement Score** - Count of emotions where annotators disagreed (0-27 scale)
5. **Example Very Unclear Flag** - Binary flag indicating annotator uncertainty


---

## Data Management

- **Raw data:** Loaded on-demand from Hugging Face (not stored locally)
- **Intermediate outputs:** `data/derived/phase1_*.csv` and `data/derived/phase2_*.csv`
- **Final outputs:** `data/derived/comments_with_clusters_k{2,4,5}.csv`
- **Models:** Saved to `models/` (gitignored - regenerate by running pipeline)
- **Results:** Confusion matrices, plots, and metrics in `results/`


---

## Team Members

- **Natalie Leung** - Feature engineering (entropy, valence), K-means clustering
- **Sang Ah Lee** - Feature engineering (disagreement scoring), clustering analysis (K-selection, elbow plot), data visualization  
- **Louisa Stumpf** - Pipeline integration, classifier training (Logistic Regression, Random Forest), GitHub management
- **Ananya Yallapragada** - Feature engineering (multi-label count, unclear flag), data processing (comment-level aggregation, cluster merging)


## Acknowledgments

This project was completed under the guidance of Professor Vishnu Pendyala for DATA 245 Machine Learning at San José State University, Spring 2026.

---

## Citation

If using this code or methodology:

```
Stumpf, L., Lee, S., Leung, N., & Yallapragada, A. (2026). 
From Noise to Signal: Leveraging Annotator Disagreement to Measure 
Emotional Complexity Across Online Communities. 
DATA 245 Machine Learning Project, San José State University.
```

**Dataset:**
```
Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). 
GoEmotions: A Dataset of Fine-Grained Emotions. 
In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (pp. 4040–4054).
```

---

## License

This project is for academic purposes as part of DATA 245 coursework at San José State University.
