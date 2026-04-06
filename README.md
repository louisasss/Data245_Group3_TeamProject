# Data245_Group3_TeamProject

## Repository Structure
Data245_Group3_TeamProject/
│
├── README.md                  ← you are here
├── data_dictionary.csv        ← variable definitions for all datasets
├── requirements.txt           ← python dependencies
├── .gitignore
│
├── data/
│   └── derived/               ← push output CSVs here
│
├── scripts/                   ← all scripts should have clear comments
│   ├── integrate_features.py  ← merges all feature CSVs (Louisa)
│   ├── clustering.py          ← K-Means clustering (Natalie & Sarah)
│   ├── classify.py            ← Logistic Regression & Random Forest (Louisa)
│   └── data_splits.py         ← train/test split logic (Ananya)
│
├── notebooks/
│   └──                        ← exploratory analysis notebooks go here (if exists)
│
└── results/
├── model_results.json         ← evaluation metrics
└── figures/                   ← confusion matrices, elbow plots, etc.

## Setup

Clone the repo and install dependencies.  

The raw GoEmotions dataset is not stored in this repo. It is loaded directly from Hugging Face in each script:
```python
from datasets import load_dataset
ds = load_dataset("google-research-datasets/go_emotions", "raw")
```



## Team Members
- Sang Ah Lee
- Natalie Leung
- Louisa Stumpf
- Ananya Yallapragada
