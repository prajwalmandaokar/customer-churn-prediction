"""
train_model.py
--------------
Trains and compares three classifiers (Logistic Regression, Random Forest,
Gradient Boosting) on the churn dataset using a proper scikit-learn
Pipeline (so preprocessing is bundled with the model — no data leakage,
and the saved .pkl is ready to run on new raw-shaped data).

Outputs:
    models/churn_model.pkl        <- best pipeline (preprocessing + model)
    models/model_columns.json     <- the raw feature columns the model expects
    reports/metrics.json          <- accuracy / precision / recall / F1 / ROC-AUC per model
    reports/figures/confusion_matrix.png
    reports/figures/roc_curve.png
    reports/figures/feature_importance.png

Run:
    python src/train_model.py
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_preprocessing import CATEGORICAL_COLS, full_pipeline

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "customer_churn.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
METRICS_PATH = ROOT / "reports" / "metrics.json"

RANDOM_STATE = 42


def build_pipeline(model, feature_cols):
    cat_cols = [c for c in CATEGORICAL_COLS + ["TenureGroup"] if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning data...")
    df, X, y = full_pipeline(str(RAW_PATH))
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = {}
    fitted_pipelines = {}

    for name, model in candidates.items():
        print(f"Training {name}...")
        pipe = build_pipeline(model, feature_cols)
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        fitted_pipelines[name] = pipe
        print(f"  {metrics}")

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name} (ROC-AUC = {results[best_name]['roc_auc']})")

    # --- Save artifacts ---
    joblib.dump(best_pipeline, MODELS_DIR / "churn_model.pkl")
    with open(MODELS_DIR / "model_columns.json", "w") as f:
        json.dump({"feature_columns": feature_cols, "best_model": best_name}, f, indent=2)

    results["best_model"] = best_name
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # --- Confusion matrix ---
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        best_pipeline, X_test, y_test, display_labels=["No Churn", "Churn"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- ROC curves for all models ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in fitted_pipelines.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
    ax.set_title("ROC Curve Comparison")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    # --- Feature importance (best model, if tree-based; else coefficients) ---
    try:
        ohe = best_pipeline.named_steps["preprocess"].named_transformers_["cat"]
        cat_cols = [c for c in CATEGORICAL_COLS + ["TenureGroup"] if c in feature_cols]
        num_cols = [c for c in feature_cols if c not in cat_cols]
        feature_names = list(ohe.get_feature_names_out(cat_cols)) + num_cols

        model_step = best_pipeline.named_steps["model"]
        if hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_
        else:
            importances = np.abs(model_step.coef_[0])

        top_idx = np.argsort(importances)[-15:]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(np.array(feature_names)[top_idx], importances[top_idx], color="#4C72B0")
        ax.set_title(f"Top 15 Features — {best_name}")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Skipping feature importance plot: {e}")

    print("\nSaved:")
    print(f"  - {MODELS_DIR / 'churn_model.pkl'}")
    print(f"  - {MODELS_DIR / 'model_columns.json'}")
    print(f"  - {METRICS_PATH}")
    print(f"  - {FIGURES_DIR}/*.png")

    print("\nFull classification report (best model):")
    print(classification_report(y_test, best_pipeline.predict(X_test), target_names=["No Churn", "Churn"]))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
