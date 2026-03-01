# practical_classifiers_comparison

This repository contains a **comprehensive comparison of classifiers** for binary classification problems, implemented as a final university assignment in **Data Science**.

The script uses **nested stratified cross-validation** to fairly compare **KNN**, **Decision Tree**, and **Random Forest** across two classic tabular datasets.

---

## Why This Project?

I created this repo to:

- Document my **final Data Science assignment** comparing multiple classifiers with proper cross-validation.
- Practice **scikit-learn pipelines** for preprocessing (imputation, scaling, one-hot encoding) + modeling.
- Learn **nested CV** (outer for test, inner for hyperparameter tuning) to avoid overfitting in model selection.
- Generate reproducible results with metrics like accuracy, F1-score, ROC-AUC, and confusion matrices.

---

## Datasets

The analysis uses two standard datasets:

1. **Adult Census Income** (`adult.csv`)  
   - Predict if income > $50K based on census data (age, education, occupation, etc.).
   - Binary target: `>50K` (positive) vs `<=50K` (negative).

2. **Bank Marketing** (`bank.csv`)  
   - Predict if client subscribes to a term deposit (`yes` / `no`).
   - Binary target: `yes` (positive) vs `no` (negative).

---

## Key Features

- **Automated preprocessing pipeline**:
  - Numeric: median imputation + StandardScaler.
  - Categorical: most-frequent imputation + OneHotEncoder (dense output for compatibility).
- **Hyperparameter grids** for each model (≥2 configurations per algorithm).
- **Nested stratified K-fold CV**: 5 outer folds (test evaluation), 3 inner folds (tuning).
- **Metrics**: Accuracy, F1 (refit), ROC-AUC, confusion matrix per fold.
- **Results**: CSV files with per-fold details and summary statistics (mean ± std).

**Models compared**:
- **KNN**: distance metrics (Manhattan/Euclidean), neighbors, weights.
- **Decision Tree**: criterion, max depth, min samples leaf (with class balancing).
- **Random Forest**: n_estimators, max depth, min samples leaf (with balanced subsampling).