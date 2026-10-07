"""CAD Detection: predict coronary artery disease from clinical data.

Compares five classifiers, each combined with an ensemble method, using a
leak-free evaluation:

1. The data is split once into a training set (70%) and a held-out test set (30%).
2. 10-fold stratified cross-validation runs on the TRAINING set only.
3. Each model is then fitted on the full training set and scored ONCE on the
   held-out test set.

Usage:
    python cad_detection.py --data heart.csv --out results

Requires scikit-learn >= 1.6, pandas, numpy, matplotlib, seaborn.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 1
TARGET = "target"


def load_data(path, keep_duplicates):
    data = pd.read_csv(path)
    print(f"Loaded {path}: {data.shape[0]} rows, {data.shape[1]} columns")
    print("Missing values per column:")
    print(data.isnull().sum().to_string())

    n_dup = int(data.duplicated().sum())
    if n_dup:
        print(f"\nFound {n_dup} duplicate rows.")
        if keep_duplicates:
            print("Keeping them (--keep-duplicates). Scores may be inflated, because the "
                  "same patient row can land in both the train and test sets.")
        else:
            data = data.drop_duplicates().reset_index(drop=True)
            print(f"Dropped them. {data.shape[0]} rows remain.")

    data = data.dropna(axis=0, how="any")
    print("\nClass balance (target):")
    print(data[TARGET].value_counts().to_string())
    return data


def plot_eda(data, out_dir):
    needed = {"age", "sex", "thal", "trestbps", "cp"}
    if not needed.issubset(data.columns):
        print("Skipping EDA plots: expected columns are missing.")
        return
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    ax = sns.scatterplot(x="age", y="sex", data=data, ax=axes[0, 0], hue="sex",
                         palette={0: "blue", 1: "red"})
    ax.set(xlabel="Age", ylabel="Sex (0: Female, 1: Male)", title="Age vs Sex")
    ax.spines[["top", "right"]].set_visible(False)

    data["thal"].plot(kind="hist", bins=20, title="Thal", ax=axes[0, 1])
    axes[0, 1].spines[["top", "right"]].set_visible(False)

    data["trestbps"].plot(kind="hist", bins=20, title="Resting blood pressure (trestbps)", ax=axes[1, 0])
    axes[1, 0].spines[["top", "right"]].set_visible(False)

    ax = sns.scatterplot(x="cp", y="trestbps", data=data, ax=axes[1, 1], hue="cp", palette="viridis")
    ax.set(xlabel="Chest pain type (cp)", ylabel="trestbps", title="cp vs trestbps")

    plt.tight_layout()
    fig.savefig(out_dir / "eda.png", dpi=150)
    plt.close(fig)


def build_models():
    """Same base models and ensembles as the original project.

    SVC, KNN and Logistic Regression are scale-sensitive, so they get a
    StandardScaler in front. The scaler is fitted inside each CV fold.
    """
    return {
        "SVM (linear) + AdaBoost": make_pipeline(
            StandardScaler(),
            AdaBoostClassifier(estimator=SVC(kernel="linear"), n_estimators=150, random_state=SEED),
        ),
        "KNN (k=1) + Bagging": make_pipeline(
            StandardScaler(),
            BaggingClassifier(estimator=KNeighborsClassifier(n_neighbors=1),
                              n_estimators=300, random_state=SEED, n_jobs=-1),
        ),
        "Decision Tree + AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(random_state=SEED), n_estimators=50, random_state=SEED
        ),
        "Logistic Regression + AdaBoost": make_pipeline(
            StandardScaler(),
            AdaBoostClassifier(estimator=LogisticRegression(C=1.0, max_iter=1000),
                               n_estimators=150, random_state=SEED),
        ),
        "Random Forest + AdaBoost": AdaBoostClassifier(
            estimator=RandomForestClassifier(random_state=SEED, n_jobs=-1),
            n_estimators=150, random_state=SEED
        ),
    }


def evaluate(models, X_train, X_test, y_train, y_test, n_splits):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scoring = ["accuracy", "precision", "recall", "f1"]
    rows = []

    for name, model in models.items():
        print(f"\n=== {name} ===")

        # 1) Cross-validation on the training set only
        cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        cv_acc, cv_acc_std = cv_res["test_accuracy"].mean(), cv_res["test_accuracy"].std()
        print(f"{n_splits}-fold CV on training set: accuracy = {cv_acc:.4f} (+/- {cv_acc_std:.4f}), "
              f"precision = {cv_res['test_precision'].mean():.4f}, "
              f"recall = {cv_res['test_recall'].mean():.4f}, f1 = {cv_res['test_f1'].mean():.4f}")

        # 2) One final fit on the full training set, one score on the held-out test set
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0  # = sensitivity
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        print("Held-out test set:")
        print(classification_report(y_test, pred, digits=3))
        print(f"Sensitivity = {recall:.4f}, Specificity = {specificity:.4f}")

        rows.append({
            "model": name,
            "cv_accuracy_mean": cv_acc,
            "cv_accuracy_std": cv_acc_std,
            "cv_f1_mean": cv_res["test_f1"].mean(),
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall_sensitivity": recall,
            "test_specificity": specificity,
            "test_f1": f1,
        })

    return pd.DataFrame(rows)


def plot_results(results, out_dir):
    ordered = results.sort_values("test_accuracy")
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(y - 0.2, ordered["cv_accuracy_mean"] * 100, height=0.4, label="10-fold CV (training set)")
    ax.barh(y + 0.2, ordered["test_accuracy"] * 100, height=0.4, label="Held-out test set")
    for i, (cv_a, te_a) in enumerate(zip(ordered["cv_accuracy_mean"], ordered["test_accuracy"])):
        ax.text(cv_a * 100 + 0.5, i - 0.2, f"{cv_a * 100:.1f}%", va="center", fontsize=9)
        ax.text(te_a * 100 + 0.5, i + 0.2, f"{te_a * 100:.1f}%", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["model"])
    ax.set_xlim(0, 110)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Classifier accuracy (computed, no train/test leakage)")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "accuracy_comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="CAD detection model comparison")
    parser.add_argument("--data", default="heart.csv", help="path to heart.csv")
    parser.add_argument("--out", default="results", help="folder for tables and figures")
    parser.add_argument("--folds", type=int, default=10, help="number of CV folds")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="do not drop duplicate rows before splitting")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.data, args.keep_duplicates)
    plot_eda(data, out_dir)

    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    results = evaluate(build_models(), X_train, X_test, y_train, y_test, args.folds)

    results.to_csv(out_dir / "results.csv", index=False)
    plot_results(results, out_dir)

    print("\n=== Summary ===")
    summary = results[["model", "cv_accuracy_mean", "test_accuracy", "test_f1"]].copy()
    for col in ["cv_accuracy_mean", "test_accuracy", "test_f1"]:
        summary[col] = (summary[col] * 100).round(2)
    print(summary.to_string(index=False))
    print(f"\nSaved results.csv, accuracy_comparison.png and eda.png to {out_dir}/")


if __name__ == "__main__":
    main()
