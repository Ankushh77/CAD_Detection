# CAD Detection

Machine learning project that predicts the presence of coronary artery disease (CAD) from patient clinical data. It compares five classic classifiers, each combined with an ensemble method, using cross-validation and a held-out test set.

> **Disclaimer:** This is an educational project. It is not a medical tool and must not be used for diagnosis.

## Overview

- Loads a heart disease dataset (`heart.csv`) with clinical features such as age, sex, chest pain type (`cp`), resting blood pressure (`trestbps`), `thal` and a binary `target` column.
- Checks for missing values and duplicate rows, and explores the data with plots (age vs sex, thal, trestbps, cp vs trestbps).
- Splits the data once into 70% training and 30% held-out test (stratified).
- Runs 10-fold stratified cross-validation on the training set only, then scores each model once on the test set.
- Saves a results table and charts computed from the actual runs.

## Models

| Base model | Ensemble method |
|---|---|
| Support Vector Classifier (linear kernel) | AdaBoost (150 estimators) |
| K-Nearest Neighbors (k = 1) | Bagging (300 estimators) |
| Decision Tree | AdaBoost (50 estimators) |
| Logistic Regression | AdaBoost (150 estimators) |
| Random Forest | AdaBoost (150 estimators) |

SVM, KNN and Logistic Regression are preceded by a `StandardScaler`, fitted inside each cross-validation fold.

## Metrics

Accuracy, precision, recall (sensitivity), specificity and F1, plus a classification report per model. In a medical setting, recall matters because a missed case is costly.

## Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn

## Project Structure

```
CAD_Detection/
├── cad_detection.py      # Main script: clean evaluation pipeline
├── CAD_Detection.ipynb   # Original Colab notebook (kept for reference)
└── README.md
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Ankushh77/CAD_Detection.git
   cd CAD_Detection
   ```
2. Install the requirements:
   ```bash
   pip install "scikit-learn>=1.6" pandas numpy matplotlib seaborn
   ```
3. Put a heart disease CSV named `heart.csv` in the project folder. The dataset is not included in this repository. It needs the usual columns (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`).
4. Run:
   ```bash
   python cad_detection.py --data heart.csv --out results
   ```

Outputs go to the `results/` folder: `results.csv` (all metrics), `accuracy_comparison.png` and `eda.png`.

Options: `--folds 10` sets the number of CV folds, and `--keep-duplicates` stops duplicate rows from being dropped before the split.

## Evaluation Notes

- **No data leakage:** cross-validation uses the training set only, and the test set is scored once at the end. The original notebook trained on folds of the full dataset and then scored on rows it had already seen, which inflated results (100% accuracy for tree models). `cad_detection.py` fixes this, so its numbers are lower and more realistic.
- **Duplicates:** many public heart disease CSVs contain repeated rows. If the same row lands in both the train and test sets, scores are inflated, so duplicates are dropped by default.
- Hyperparameters were not tuned. Tuning with a nested cross-validation loop is a natural next step.

## Results

Run the script on your copy of the dataset to generate `results/results.csv`, then paste the table here.

## Author

Ankush ([@Ankushh77](https://github.com/Ankushh77))
