"""
╔══════════════════════════════════════════════════════════════╗
║             MODEL TRAINING — train_model.py                  ║
║                                                              ║
║  Receives the clean DataFrame, target, task type, and        ║
║  pipeline review from main.py.                               ║
║  Auto-detects classification vs regression and trains         ║
║  the recommended models with evaluation metrics.             ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)

# ── Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ── Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────
CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVC": SVC(random_state=42),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
}

REGRESSION_MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42),
    "ElasticNet": ElasticNet(random_state=42),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
}


# ─────────────────────────────────────────────
# RESOLVE MODEL NAMES FROM AI REVIEW
# ─────────────────────────────────────────────
def resolve_models(review: dict, task_type: str) -> list:
    """
    Map the AI-recommended model names to our registry keys.
    Falls back to a sensible default set if no match found.
    """
    registry = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    ai_models = [m["model"] for m in review.get("final_models", [])]

    resolved = []
    for name in ai_models:
        # Try exact match first
        if name in registry:
            resolved.append(name)
            continue
        # Fuzzy match (case-insensitive partial)
        for key in registry:
            if name.lower().replace(" ", "") in key.lower().replace(" ", ""):
                resolved.append(key)
                break

    # Fallback: always train at least baseline + 2 models
    if not resolved:
        if task_type == "classification":
            resolved = ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"]
        else:
            resolved = ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"]

    return list(dict.fromkeys(resolved))  # dedupe preserving order


# ─────────────────────────────────────────────
# TRAIN & EVALUATE — CLASSIFICATION
# ─────────────────────────────────────────────
def train_classification(X_train, X_test, y_train, y_test, model_names: list) -> list:
    """Train classification models and return results list."""
    results = []
    for name in model_names:
        model = CLASSIFICATION_MODELS.get(name)
        if model is None:
            print(f"  [WARN] Skipping unknown model: {name}")
            continue

        print(f"\n  [TRAIN] Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        result = {
            "model": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "cv_mean_accuracy": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        results.append(result)

        print(f"     Accuracy : {acc:.4f}")
        print(f"     F1 Score : {f1:.4f}")
        print(f"     CV Mean  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return results


# ─────────────────────────────────────────────
# TRAIN & EVALUATE — REGRESSION
# ─────────────────────────────────────────────
def train_regression(X_train, X_test, y_train, y_test, model_names: list) -> list:
    """Train regression models and return results list."""
    results = []
    for name in model_names:
        model = REGRESSION_MODELS.get(name)
        if model is None:
            print(f"  [WARN] Skipping unknown model: {name}")
            continue

        print(f"\n  [TRAIN] Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        result = {
            "model": name,
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2_score": round(r2, 4),
            "cv_mean_r2": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
        }
        results.append(result)

        print(f"     R2 Score : {r2:.4f}")
        print(f"     RMSE     : {rmse:.4f}")
        print(f"     MAE      : {mae:.4f}")
        print(f"     CV Mean  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return results


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT — called from main.py
# ─────────────────────────────────────────────
def run_training(
    clean_df: pd.DataFrame,
    target_column: str,
    task_type: str,
    review: dict,
    project_dir: str,
) -> None:
    """
    Split data, resolve models from AI review, train, evaluate,
    and save results to project_dir.
    """
    # ── Validate target
    if target_column not in clean_df.columns:
        print(f"[ERROR] Target column '{target_column}' not found in clean dataset.")
        return

    # ── Drop any remaining nulls in target
    clean_df = clean_df.dropna(subset=[target_column])

    # ── Split features / target
    X = clean_df.drop(columns=[target_column])
    y = clean_df[target_column]

    # Drop any remaining non-numeric columns that weren't encoded
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        print(f"  [WARN] Dropping unencoded columns from features: {non_numeric}")
        X = X.drop(columns=non_numeric)

    # ── Handle remaining NaN in features
    if X.isnull().sum().sum() > 0:
        print("  [FIX] Filling remaining NaN in features with column median...")
        X = X.fillna(X.median())

    # ── Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n  [SPLIT] Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")

    # ── Resolve which models to train
    model_names = resolve_models(review, task_type)
    print(f"  [MODELS] Models to train: {model_names}")

    # ── Train
    if task_type == "classification":
        results = train_classification(X_train, X_test, y_train, y_test, model_names)
        # Find best model
        if results:
            best = max(results, key=lambda r: r["f1_score"])
            print(f"\n  >> BEST MODEL: {best['model']}  (F1 = {best['f1_score']:.4f})")
    else:
        results = train_regression(X_train, X_test, y_train, y_test, model_names)
        # Find best model
        if results:
            best = max(results, key=lambda r: r["r2_score"])
            print(f"\n  >> BEST MODEL: {best['model']}  (R2 = {best['r2_score']:.4f})")

    # ── Save training results
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task_type": task_type,
        "target_column": target_column,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "features_used": X.columns.tolist(),
        "models_trained": model_names,
        "results": results,
        "best_model": best["model"] if results else None,
    }

    results_path = os.path.join(project_dir, "training_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  [SAVED] Training results saved -> {results_path}")

    # ── Print summary table
    print(f"\n{'=' * 60}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 60}")
    if task_type == "classification":
        print(f"  {'Model':<35} {'Accuracy':>10} {'F1':>10} {'CV Mean':>10}")
        print(f"  {'_' * 65}")
        for r in sorted(results, key=lambda x: x["f1_score"], reverse=True):
            marker = " << BEST" if r["model"] == best["model"] else ""
            print(
                f"  {r['model']:<35} {r['accuracy']:>10.4f} "
                f"{r['f1_score']:>10.4f} {r['cv_mean_accuracy']:>10.4f}{marker}"
            )
    else:
        print(f"  {'Model':<35} {'R2':>10} {'RMSE':>10} {'CV Mean':>10}")
        print(f"  {'_' * 65}")
        for r in sorted(results, key=lambda x: x["r2_score"], reverse=True):
            marker = " << BEST" if r["model"] == best["model"] else ""
            print(
                f"  {r['model']:<35} {r['r2_score']:>10.4f} "
                f"{r['rmse']:>10.4f} {r['cv_mean_r2']:>10.4f}{marker}"
            )
    print(f"{'=' * 60}")
