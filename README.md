# Ensemble Learning on the Adult Income Census Dataset

Predicting whether an individual's annual income exceeds **$50K** using three ensemble
learning algorithms — **Random Forest**, **AdaBoost**, and **XGBoost** — on the UCI
Adult Income Census dataset.

## 📌 Problem Statement

Binary classification task: predict whether a person's income is `<=50K` or `>50K` per
year based on demographic and employment attributes (age, education, occupation,
marital status, hours worked per week, etc.).

## 📂 Repository Structure

```
.
├── Ensemble_Learning_Adult_Income.ipynb   # Main Jupyter/Colab notebook (full pipeline)
├── adult.csv                              # Dataset (UCI Adult Income Census)
├── report.tex                             # LaTeX source for the final report
├── report.pdf                             # Compiled final report (3–6 pages)
├── figs/                                  # Plots used in the report (EDA, confusion matrices, feature importance)
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

## 📊 Dataset

- **Source:** [UCI Adult Income Census Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income)
- **Records:** 32,561 (32,537 after removing duplicates)
- **Features:** 14 demographic/employment attributes + target
- **Target:** `income` — `<=50K` (75.9%) vs `>50K` (24.1%)

## ⚙️ Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

## 🚀 How to Run

### Option 1: Google Colab
1. Upload `Ensemble_Learning_Adult_Income.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Upload `adult.csv` to the Colab session (or mount Google Drive — instructions in the notebook).
3. Run all cells top to bottom.

### Option 2: Local Jupyter
```bash
jupyter notebook Ensemble_Learning_Adult_Income.ipynb
```

## 🧪 Methodology

1. **EDA** — structure, missing values, duplicates, target distribution, feature distributions/correlations.
2. **Preprocessing** — missing-value imputation (mode), duplicate removal, dropped `fnlwgt`/`education`, engineered `capital.net`, one-hot encoding + standard scaling, stratified 80:20 train/test split.
3. **Model Building** — Random Forest, AdaBoost, XGBoost, each in a `scikit-learn` Pipeline.
4. **Hyperparameter Tuning** — `GridSearchCV` (3-fold CV, F1-score scoring).
5. **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrices.
6. **Feature Importance** — top 10 features for Random Forest and XGBoost.

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | **0.8677** | 0.7734 | **0.6378** | **0.6991** | **0.9221** |
| Random Forest | 0.8594 | 0.7793 | 0.5810 | 0.6657 | 0.9131 |
| AdaBoost | 0.8523 | **0.7871** | 0.5306 | 0.6339 | 0.9026 |

**Best model: XGBoost**, offering the strongest balance of accuracy, F1-score, and ROC-AUC.

### Top predictive features
`marital.status_Married-civ-spouse`, `education.num`, `capital.gain`/`capital.loss`,
`relationship`, `occupation`, `hours.per.week`, `age`.

See [`report.pdf`](./report.pdf) for the full write-up, including bagging-vs-boosting
analysis, strengths/limitations of each model, and detailed feature-importance discussion.

## 🛠️ Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn
- XGBoost
- Matplotlib, Seaborn

## 📄 License

This project is for educational purposes as part of an ensemble learning assignment.
