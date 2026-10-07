# CAD Detection

Machine learning project that predicts the presence of coronary artery disease (CAD) from patient clinical data. It compares five classic classifiers, each combined with an ensemble method, using 10-fold cross-validation and a held-out test set.

> **Disclaimer:** This is an educational project. It is not a medical tool and must not be used for diagnosis.

## Dataset

[heart.csv on Kaggle](https://www.kaggle.com/datasets/arezaei81/heartcsv): 303 patients, 13 clinical features (age, sex, chest pain type `cp`, resting blood pressure `trestbps`, cholesterol `chol`, `thalach`, `oldpeak`, `ca`, `thal` and others) and a binary `target`.

- 1 duplicate row is dropped before splitting, leaving 302 patients (164 with `target = 1`, 138 with `target = 0`).
- Stratified split: 211 patients for training, 91 for the held-out test set.
- The dataset is not included in this repository. Download `heart.csv` from the link above and place it in the project folder.

## Method

- Data checks and a few exploratory plots.
- One stratified 70/30 train/test split, made before any model is fitted.
- **10-fold stratified cross-validation on the training set only**, then a single final score on the held-out test set. The test set never influences training.
- SVM, KNN and Logistic Regression are preceded by a `StandardScaler`, fitted inside each CV fold.

| Base model | Ensemble method |
|---|---|
| Support Vector Classifier (linear kernel) | AdaBoost (150 estimators) |
| K-Nearest Neighbors (k = 1) | Bagging (300 estimators) |
| Decision Tree | AdaBoost (50 estimators) |
| Logistic Regression | AdaBoost (150 estimators) |
| Random Forest | AdaBoost (150 estimators) |

## Results

All values in %. CV = 10-fold cross-validation on the training set (mean ± std). The other columns are on the held-out test set (91 patients).

| Model | CV accuracy | Test accuracy | Precision | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|
| SVM (linear) + AdaBoost | 79.1 ± 4.9 | 84.6 | 79.7 | 95.9 | 71.4 | 87.0 |
| KNN (k=1) + Bagging | 74.9 ± 7.0 | 79.1 | 76.8 | 87.8 | 69.0 | 81.9 |
| Decision Tree + AdaBoost | 68.7 ± 5.5 | 71.4 | 76.7 | 67.3 | 76.2 | 71.7 |
| Logistic Regression + AdaBoost | 77.7 ± 6.1 | 85.7 | 81.0 | 95.9 | 73.8 | 87.9 |
| Random Forest + AdaBoost | 79.1 ± 9.1 | 81.3 | 83.3 | 81.6 | 81.0 | 82.5 |

**How to read this**
- SVM, Logistic Regression and Random Forest are close in cross-validation (about 78 to 79%). With a spread of 5 to 9 points between folds, the differences between them are not meaningful.
- With only 91 test patients, one patient moves test accuracy by about 1.1 points, so the test-set ranking should be read loosely.
- The linear models catch about 96% of disease cases (sensitivity) at the cost of more false alarms, a trade-off that is usually preferred in screening.
- AdaBoost stopped after one round for the Decision Tree and the Random Forest, because a full-depth tree fits the training data perfectly. Those two rows are therefore effectively a single tree and a single forest. Boosting shallow trees (stumps) would be a better set-up.

## Notes on Evaluation

- An earlier version of this project trained on folds of the full dataset and then scored on rows that had already been used for training, which gave 99 to 100% accuracy for the Decision Tree, Random Forest and KNN models. This version removes that leakage, so the numbers are lower and realistic.
- Hyperparameters were not tuned.
- The label encoding follows the Kaggle file. Check it against the original UCI documentation before drawing any clinical conclusion.

## Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn

## Project Structure

```
CAD_Detection/
├── CAD_Detection.ipynb   # Notebook with explanations, outputs and charts
├── cad_detection.py      # Same pipeline as a command-line script
└── README.md
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Ankushh77/CAD_Detection.git
   cd CAD_Detection
   ```
2. Install the requirements (scikit-learn 1.6 or newer):
   ```bash
   pip install "scikit-learn>=1.6" pandas numpy matplotlib seaborn jupyter
   ```
3. Download `heart.csv` from the Kaggle link above into the project folder.
4. Either open `CAD_Detection.ipynb` in Jupyter or Colab (in Colab it asks you to upload `heart.csv`), or run the script:
   ```bash
   python cad_detection.py --data heart.csv --out results
   ```
   The script saves `results.csv`, `accuracy_comparison.png` and `eda.png` to the `results/` folder. Use `--keep-duplicates` to keep duplicate rows.

## Team

Developed as a group project by:

- Ankush ([@Ankushh77](https://github.com/Ankushh77))
- TEAMMATE NAME (GITHUB LINK)
- TEAMMATE NAME ([@MrTig-afk](https://github.com/MrTig-afk)). Original repository: https://github.com/MrTig-afk/CAD_Detection
