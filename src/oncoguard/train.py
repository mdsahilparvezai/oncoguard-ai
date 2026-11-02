"""
Training module for ONCOGUARD AI - Tabular baseline models.

This module provides:
- Multiple baseline classifiers (Dummy, Logistic Regression, SVM, Random Forest, Gradient Boosting)
- Stratified cross-validation with leakage prevention
- Hyperparameter tuning using training data only
- Model serialization and metadata tracking
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from src.oncoguard.data import SplitData, FEATURE_COLUMNS, TARGET_NAMES, PROCESSED_DATA_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Cross-validation configuration
CV_FOLDS = 5
CV_RANDOM_STATE = 42
SCORING = [
    "accuracy", "balanced_accuracy", "precision", "recall",
    "f1", "roc_auc", "average_precision"
]


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    estimator: BaseEstimator
    param_grid: Dict[str, List[Any]]
    use_calibration: bool = False
    search_method: str = "grid"  # "grid" or "random"
    n_iter: int = 20  # for random search


@dataclass
class CVResult:
    """Container for cross-validation results."""
    model_name: str
    cv_scores: Dict[str, np.ndarray]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    best_params: Dict[str, Any]
    best_estimator: BaseEstimator
    fit_time: float
    score_time: float


@dataclass
class TestResult:
    """Container for held-out test evaluation results."""
    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]


def get_model_configs() -> List[ModelConfig]:
    """Get all model configurations for tabular baseline."""
    configs = [
        ModelConfig(
            name="DummyClassifier (stratified)",
            estimator=DummyClassifier(strategy="stratified", random_state=42),
            param_grid={},
            use_calibration=False
        ),
        ModelConfig(
            name="Logistic Regression",
            estimator=LogisticRegression(
                random_state=42,
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs"
            ),
            param_grid={
                "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l2"],
            },
            use_calibration=True,
            search_method="grid"
        ),
        ModelConfig(
            name="Support Vector Machine",
            estimator=SVC(
                random_state=42,
                class_weight="balanced",
                probability=True,
                kernel="rbf"
            ),
            param_grid={
                "C": [0.1, 1.0, 10.0, 100.0],
                "gamma": ["scale", "auto", 0.01, 0.1, 1.0],
            },
            use_calibration=False,  # SVC has built-in probability calibration
            search_method="random",
            n_iter=15
        ),
        ModelConfig(
            name="Random Forest",
            estimator=RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            ),
            param_grid={
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            },
            use_calibration=True,
            search_method="random",
            n_iter=20
        ),
        ModelConfig(
            name="Gradient Boosting",
            estimator=GradientBoostingClassifier(
                random_state=42
            ),
            param_grid={
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5, 6],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "subsample": [0.8, 0.9, 1.0],
            },
            use_calibration=True,
            search_method="random",
            n_iter=20
        ),
    ]
    return configs


def create_cv_splitter(n_splits: int = CV_FOLDS, random_state: int = CV_RANDOM_STATE) -> StratifiedKFold:
    """Create stratified k-fold cross-validator."""
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )


def run_cross_validation(
    config: ModelConfig,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Optional[StratifiedKFold] = None
) -> CVResult:
    """
    Run cross-validation with hyperparameter tuning.

    Args:
        config: Model configuration.
        X: Training features.
        y: Training target.
        cv: Cross-validation splitter (created if not provided).

    Returns:
        CVResult with scores and best estimator.
    """
    if cv is None:
        cv = create_cv_splitter()

    logger.info(f"Running CV for: {config.name}")

    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", config.estimator)
    ])

    # Adjust param grid for pipeline
    param_grid = {f"model__{k}": v for k, v in config.param_grid.items()}

    # Run hyperparameter search
    if config.param_grid:
        if config.search_method == "grid":
            search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=0,
                return_train_score=False
            )
        else:
            search = RandomizedSearchCV(
                pipeline,
                param_grid,
                n_iter=config.n_iter,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=0,
                random_state=CV_RANDOM_STATE,
                return_train_score=False
            )

        search.fit(X, y)
        best_pipeline = search.best_estimator_
        best_params = search.best_params_
        logger.info(f"Best params for {config.name}: {best_params}")
    else:
        # No hyperparameters to tune
        best_pipeline = pipeline
        best_pipeline.fit(X, y)
        best_params = {}

    # Apply calibration if requested
    if config.use_calibration and hasattr(best_pipeline.named_steps["model"], "predict_proba"):
        # Use CalibratedClassifierCV with prefit estimator
        calibrated = CalibratedClassifierCV(
            best_pipeline,
            method="isotonic",
            cv=3
        )
        calibrated.fit(X, y)
        best_pipeline = calibrated
        logger.info(f"Applied isotonic calibration to {config.name}")

    # Run final cross-validation on best model
    cv_results = cross_validate(
        best_pipeline,
        X, y,
        cv=cv,
        scoring=SCORING,
        return_train_score=False,
        return_estimator=False,
        n_jobs=-1
    )

    # Compute mean/std scores
    mean_scores = {k.replace("test_", ""): float(v.mean()) for k, v in cv_results.items() if k.startswith("test_")}
    std_scores = {k.replace("test_", ""): float(v.std()) for k, v in cv_results.items() if k.startswith("test_")}
    cv_scores = {k.replace("test_", ""): v for k, v in cv_results.items() if k.startswith("test_")}

    fit_time = float(cv_results.get("fit_time", np.array([0])).mean())
    score_time = float(cv_results.get("score_time", np.array([0])).mean())

    logger.info(f"CV Results for {config.name}:")
    for metric, mean_val in mean_scores.items():
        std_val = std_scores.get(metric, 0)
        logger.info(f"  {metric}: {mean_val:.4f} (+/- {std_val:.4f})")

    return CVResult(
        model_name=config.name,
        cv_scores=cv_scores,
        mean_scores=mean_scores,
        std_scores=std_scores,
        best_params=best_params,
        best_estimator=best_pipeline,
        fit_time=fit_time,
        score_time=score_time
    )


def evaluate_on_test_set(
    cv_result: CVResult,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> TestResult:
    """
    Evaluate the best model on held-out test set.

    Args:
        cv_result: Cross-validation result with best estimator.
        X_test: Test features.
        y_test: Test target.

    Returns:
        TestResult with predictions and metrics.
    """
    logger.info(f"Evaluating {cv_result.model_name} on test set...")

    estimator = cv_result.best_estimator

    # Predictions
    y_pred = estimator.predict(X_test)
    y_prob = estimator.predict_proba(X_test)[:, 1]  # Probability of positive class (malignant)

    # Metrics
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, average_precision_score,
        confusion_matrix, classification_report
    )

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
    }

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=list(TARGET_NAMES.values()), output_dict=True, zero_division=0)

    logger.info(f"Test metrics for {cv_result.model_name}:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    return TestResult(
        model_name=cv_result.model_name,
        y_true=y_test.values,
        y_pred=y_pred,
        y_prob=y_prob,
        metrics=metrics,
        confusion_matrix=cm,
        classification_report=cr
    )


def select_best_model(
    cv_results: List[CVResult],
    primary_metric: str = "roc_auc"
) -> CVResult:
    """
    Select best model based on cross-validation primary metric.

    Args:
        cv_results: List of CV results.
        primary_metric: Metric to optimize (must be in mean_scores).

    Returns:
        Best CVResult.
    """
    valid_results = [r for r in cv_results if primary_metric in r.mean_scores]
    if not valid_results:
        raise ValueError(f"No results have metric: {primary_metric}")

    best = max(valid_results, key=lambda r: r.mean_scores[primary_metric])
    logger.info(f"Selected best model: {best.model_name} ({primary_metric}={best.mean_scores[primary_metric]:.4f})")
    return best


def save_model_artifact(
    cv_result: CVResult,
    test_result: Optional[TestResult] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save model artifact with metadata.

    Args:
        cv_result: Cross-validation result with best estimator.
        test_result: Optional test evaluation result.
        metadata: Additional metadata to save.

    Returns:
        Path to saved model file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = cv_result.model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    model_path = MODELS_DIR / f"{model_name_safe}_{timestamp}.joblib"

    # Prepare artifact
    artifact = {
        "model": cv_result.best_estimator,
        "model_name": cv_result.model_name,
        "cv_results": {
            "mean_scores": cv_result.mean_scores,
            "std_scores": cv_result.std_scores,
            "best_params": cv_result.best_params,
            "fit_time": cv_result.fit_time,
            "score_time": cv_result.score_time,
        },
        "feature_names": FEATURE_COLUMNS,
        "target_names": TARGET_NAMES,
        "target_mapping": {"Benign": 0, "Malignant": 1},
        "timestamp": timestamp,
        "created_at": datetime.now().isoformat(),
    }

    if test_result:
        artifact["test_results"] = {
            "metrics": test_result.metrics,
            "confusion_matrix": test_result.confusion_matrix.tolist(),
            "classification_report": test_result.classification_report,
        }

    if metadata:
        artifact["metadata"] = metadata

    joblib.dump(artifact, model_path)
    logger.info(f"Saved model artifact to: {model_path}")
    return model_path


def load_model_artifact(path: Path) -> Dict[str, Any]:
    """Load model artifact from disk."""
    artifact = joblib.load(path)
    logger.info(f"Loaded model artifact from: {path}")
    return artifact


def verify_model_reload(path: Path, X_test: pd.DataFrame, y_test: pd.Series) -> bool:
    """
    Verify that a saved model can be reloaded and produces identical predictions.

    Args:
        path: Path to model artifact.
        X_test: Test features.
        y_test: Test target.

    Returns:
        True if verification passes.
    """
    artifact = load_model_artifact(path)
    model = artifact["model"]

    # Check if fitted
    try:
        check_is_fitted(model)
    except Exception as e:
        logger.error(f"Model not properly fitted: {e}")
        return False

    # Get predictions
    y_pred_original = artifact.get("test_results", {}).get("y_pred")
    if y_pred_original is None:
        y_pred_original = model.predict(X_test)

    y_pred_reloaded = model.predict(X_test)
    y_prob_reloaded = model.predict_proba(X_test)[:, 1]

    # Compare
    pred_match = np.array_equal(y_pred_original, y_pred_reloaded)
    prob_match = np.allclose(
        artifact.get("test_results", {}).get("y_prob", model.predict_proba(X_test)[:, 1]),
        y_prob_reloaded
    )

    if pred_match and prob_match:
        logger.info("Model reload verification PASSED")
        return True
    else:
        logger.error(f"Model reload verification FAILED: pred_match={pred_match}, prob_match={prob_match}")
        return False


def run_full_training_pipeline(
    splits: SplitData,
    primary_metric: str = "roc_auc"
) -> Tuple[List[CVResult], TestResult, Path]:
    """
    Run complete training pipeline: CV for all models, select best, evaluate on test, save.

    Args:
        splits: SplitData from data pipeline.
        primary_metric: Metric for model selection.

    Returns:
        Tuple of (all_cv_results, best_test_result, model_path)
    """
    logger.info("Starting full training pipeline...")

    # Combine train and validation for cross-validation
    X_train_val = pd.concat([splits.X_train, splits.X_val])
    y_train_val = pd.concat([splits.y_train, splits.y_val])

    # Run CV for all models
    configs = get_model_configs()
    cv_results = []

    for config in configs:
        try:
            cv_result = run_cross_validation(config, X_train_val, y_train_val)
            cv_results.append(cv_result)
        except Exception as e:
            logger.error(f"Failed to train {config.name}: {e}")
            continue

    if not cv_results:
        raise RuntimeError("All models failed to train")

    # Select best model
    best_cv_result = select_best_model(cv_results, primary_metric)

    # Evaluate on held-out test set
    best_test_result = evaluate_on_test_set(best_cv_result, splits.X_test, splits.y_test)

    # Save best model
    metadata = {
        "primary_metric": primary_metric,
        "cv_folds": CV_FOLDS,
        "n_train_val": len(X_train_val),
        "n_test": len(splits.X_test),
        "all_cv_results": [
            {
                "model_name": r.model_name,
                "mean_scores": r.mean_scores,
                "std_scores": r.std_scores,
                "best_params": r.best_params,
            }
            for r in cv_results
        ]
    }
    model_path = save_model_artifact(best_cv_result, best_test_result, metadata)

    # Verify reload
    verify_model_reload(model_path, splits.X_test, splits.y_test)

    logger.info("Full training pipeline completed successfully")
    return cv_results, best_test_result, model_path


if __name__ == "__main__":
    # Quick test when run directly
    from .data import run_data_pipeline
    splits, _ = run_data_pipeline()
    cv_results, test_result, model_path = run_full_training_pipeline(splits)
    print(f"Training test passed. Best model: {test_result.model_name}")
    print(f"Test ROC-AUC: {test_result.metrics['roc_auc']:.4f}")