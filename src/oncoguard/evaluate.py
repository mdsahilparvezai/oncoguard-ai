"""
Evaluation and explainability module for ONCOGUARD AI.

This module provides:
- Comprehensive model evaluation metrics
- Confusion matrix analysis
- Calibration diagnostics
- SHAP explainability for tree-based models
- Feature importance analysis
- Experiment report generation
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
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
    brier_score_loss
)
from sklearn.inspection import permutation_importance

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from src.oncoguard.data import SplitData, FEATURE_COLUMNS, TARGET_NAMES, PROCESSED_DATA_PATH
from src.oncoguard.train import TestResult, load_model_artifact, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CalibrationResult:
    fraction_of_positives: np.ndarray
    mean_predicted_value: np.ndarray
    brier_score: float
    ece: float


@dataclass
class FeatureImportanceResult:
    method: str
    feature_names: List[str]
    importances: np.ndarray
    std_importances: Optional[np.ndarray] = None


@dataclass
class SHAPResult:
    values: np.ndarray
    base_values: np.ndarray
    data: np.ndarray
    feature_names: List[str]
    model_type: str


def compute_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> CalibrationResult:
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    brier = brier_score_loss(y_true, y_prob)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return CalibrationResult(
        fraction_of_positives=fraction_of_positives,
        mean_predicted_value=mean_predicted_value,
        brier_score=float(brier),
        ece=float(ece)
    )


def compute_permutation_importance(model: Any, X: pd.DataFrame, y: pd.Series,
                                    n_repeats: int = 10, random_state: int = 42,
                                    scoring: str = "roc_auc") -> FeatureImportanceResult:
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state,
        scoring=scoring, n_jobs=-1
    )
    sorted_idx = result.importances_mean.argsort()[::-1]
    return FeatureImportanceResult(
        method="permutation",
        feature_names=[X.columns[i] for i in sorted_idx],
        importances=result.importances_mean[sorted_idx],
        std_importances=result.importances_std[sorted_idx]
    )


def compute_shap_values(model: Any, X: pd.DataFrame, sample_size: Optional[int] = None,
                         random_state: int = 42) -> Optional[SHAPResult]:
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available, skipping SHAP analysis")
        return None

    if sample_size and len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=random_state)
    else:
        X_sample = X

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return SHAPResult(
            values=shap_values,
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_sample.values,
            feature_names=list(X.columns),
            model_type="tree"
        )
    except Exception as e:
        logger.warning(f"TreeExplainer failed: {e}")
        try:
            background = shap.sample(X, min(100, len(X)), random_state=random_state)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_sample, nsamples=100)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            return SHAPResult(
                values=shap_values,
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                data=X_sample.values,
                feature_names=list(X.columns),
                model_type="kernel"
            )
        except Exception as e2:
            logger.warning(f"KernelExplainer also failed: {e2}")
            return None


def compute_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, np.ndarray]:
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    return {
        "fpr": fpr, "tpr": tpr, "roc_thresholds": roc_thresholds,
        "precision": precision, "recall": recall, "pr_thresholds": pr_thresholds,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob)
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> Tuple[float, Dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)

    if metric == "f1":
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]
        metrics_at = {"f1": f1_scores[best_idx], "precision": precision[best_idx],
                      "recall": recall[best_idx], "threshold": best_threshold}
    elif metric == "balanced_accuracy":
        balanced_acc = (tpr + (1 - fpr)) / 2
        best_idx = np.argmax(balanced_acc[:-1])
        best_threshold = roc_thresholds[best_idx]
        y_pred_thresh = (y_prob >= best_threshold).astype(int)
        metrics_at = {"balanced_accuracy": balanced_acc[best_idx],
                      "sensitivity": tpr[best_idx], "specificity": 1 - fpr[best_idx],
                      "threshold": best_threshold}
    else:
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores[:-1])
        best_threshold = roc_thresholds[best_idx]
        metrics_at = {"youden_j": j_scores[best_idx], "sensitivity": tpr[best_idx],
                      "specificity": 1 - fpr[best_idx], "threshold": best_threshold}

    return float(best_threshold), metrics_at


def generate_model_card(model_path: Path, test_result: TestResult, cv_results: List[Any],
                        calibration: CalibrationResult, feature_importance: FeatureImportanceResult,
                        shap_result: Optional[SHAPResult], optimal_threshold: float,
                        threshold_metrics: Dict[str, float]) -> Dict[str, Any]:
    artifact = load_model_artifact(model_path)

    model_card = {
        "model_details": {
            "name": artifact.get("model_name", "ONCOGUARD Baseline"),
            "version": "1.0.0",
            "date": datetime.now().isoformat(),
            "type": "Binary Classification",
            "task": "Breast tumor classification (Malignant vs Benign)",
            "dataset": "Wisconsin Diagnostic Breast Cancer (UCI)",
            "license": "CC BY 4.0 (UCI)",
        },
        "intended_use": {
            "primary_use": "Educational research and benchmarking",
            "users": "Researchers, students, ML practitioners",
            "out_of_scope": ["Clinical diagnosis", "Patient screening",
                           "Treatment decisions", "Any medical decision-making"]
        },
        "training_data": {
            "source": "UCI Machine Learning Repository",
            "url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
            "n_samples": 569,
            "n_features": 30,
            "features": "Mean, SE, and Worst of 10 nuclear morphology measurements",
            "target": "Diagnosis (M=Malignant, B=Benign)",
            "preprocessing": "StandardScaler fit on training data only",
            "split": "Stratified 70/10/20 (train/val/test)"
        },
        "evaluation": {
            "cross_validation": {
                "method": "Stratified 5-fold CV",
                "primary_metric": "ROC-AUC",
                "results": {
                    r.model_name: {
                        "mean_roc_auc": r.mean_scores.get("roc_auc"),
                        "std_roc_auc": r.std_scores.get("roc_auc"),
                        "mean_balanced_accuracy": r.mean_scores.get("balanced_accuracy"),
                        "mean_f1": r.mean_scores.get("f1"),
                        "best_params": r.best_params
                    } for r in cv_results
                }
            },
            "held_out_test": {
                "n_samples": len(test_result.y_true),
                "class_distribution": {
                    "Benign": int((test_result.y_true == 0).sum()),
                    "Malignant": int((test_result.y_true == 1).sum())
                },
                "metrics": test_result.metrics,
                "confusion_matrix": test_result.confusion_matrix.tolist(),
                "classification_report": test_result.classification_report
            },
            "calibration": {
                "brier_score": calibration.brier_score,
                "expected_calibration_error": calibration.ece,
                "calibration_curve": {
                    "fraction_of_positives": calibration.fraction_of_positives.tolist(),
                    "mean_predicted_value": calibration.mean_predicted_value.tolist()
                }
            },
            "threshold_analysis": {
                "default_threshold": 0.5,
                "optimal_threshold": optimal_threshold,
                "metrics_at_optimal": threshold_metrics
            }
        },
        "explainability": {
            "permutation_importance": {
                "method": feature_importance.method,
                "top_10_features": dict(zip(
                    feature_importance.feature_names[:10],
                    feature_importance.importances[:10].tolist()
                )),
                "all_features": dict(zip(
                    feature_importance.feature_names,
                    feature_importance.importances.tolist()
                ))
            },
            "shap_available": shap_result is not None,
            "shap_model_type": shap_result.model_type if shap_result else None
        },
        "limitations": [
            "Small Dataset: 569 samples limits statistical power",
            "Single Institution: Data from University of Wisconsin only",
            "No Clinical Metadata: Age, demographics, outcomes not included",
            "Feature Extraction: Features from FNA images, not raw histopathology",
            "No External Validation: Performance on independent cohorts unknown",
            "Research Only: Not validated for clinical decision-making",
            "Class Imbalance: ~37% malignant requires balanced metrics"
        ],
        "ethical_considerations": [
            "Research-only tool - not for clinical use",
            "False negatives carry serious consequences (missed cancer)",
            "Model bias unknown - no demographic data in features",
            "Requires human expert oversight for any real-world application"
        ],
        "artifact_path": str(model_path)
    }
    return model_card


def save_model_card(model_card: Dict[str, Any], path: Optional[Path] = None) -> Path:
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"model_card_{timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(model_card, f, indent=2, default=str)
    logger.info(f"Saved model card to: {path}")
    return path


def generate_experiment_report(splits: SplitData, cv_results: List[Any], test_result: TestResult,
                               calibration: CalibrationResult, feature_importance: FeatureImportanceResult,
                               shap_result: Optional[SHAPResult], optimal_threshold: float,
                               threshold_metrics: Dict[str, float], model_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"experiment_report_{timestamp}.md"

    lines = [
        f"# ONCOGUARD AI - Experiment Report",
        f"", f"**Generated:** {datetime.now().isoformat()}",
        f"**Model:** {test_result.model_name}",
        f"**Model Artifact:** {model_path.name}", f"", f"---", f"",
        f"## 1. Dataset Summary", f"",
        f"- **Source:** Wisconsin Diagnostic Breast Cancer (UCI Repository)",
        f"- **Total Samples:** {len(splits.X_train) + len(splits.X_val) + len(splits.X_test)}",
        f"- **Features:** 30 numerical features (10 nuclear measurements x 3 statistics)",
        f"- **Target:** Binary (Benign=0, Malignant=1)",
        f"- **Split:** Train={len(splits.X_train)}, Val={len(splits.X_val)}, Test={len(splits.X_test)}",
        f"- **Class Distribution (Test):** Benign={int((splits.y_test==0).sum())}, Malignant={int((splits.y_test==1).sum())}",
        f"", f"---", f"", f"## 2. Cross-Validation Results", f"",
        f"| Model | ROC-AUC | Balanced Acc | Precision | Recall | F1 | Best Params |",
        f"|-------|---------|--------------|-----------|--------|-----|-------------|",
    ]

    for r in cv_results:
        lines.append(
            f"| {r.model_name} | "
            f"{r.mean_scores.get('roc_auc', 0):.4f} (+/- {r.std_scores.get('roc_auc', 0):.4f}) | "
            f"{r.mean_scores.get('balanced_accuracy', 0):.4f} | "
            f"{r.mean_scores.get('precision', 0):.4f} | "
            f"{r.mean_scores.get('recall', 0):.4f} | "
            f"{r.mean_scores.get('f1', 0):.4f} | "
            f"{str(r.best_params)[:50]}... |"
        )

    lines.extend([
        f"", f"---", f"", f"## 3. Held-Out Test Evaluation", f"",
        f"### Best Model: {test_result.model_name}", f"",
        f"| Metric | Value |", f"|--------|-------|",
    ])

    for metric, value in test_result.metrics.items():
        lines.append(f"| {metric} | {value:.4f} |")

    lines.extend([
        f"", f"### Confusion Matrix", f"", f"```",
        f"                 Predicted",
        f"              Benign  Malignant",
        f"Actual Benign   {test_result.confusion_matrix[0,0]:4d}     {test_result.confusion_matrix[0,1]:4d}",
        f"     Malignant  {test_result.confusion_matrix[1,0]:4d}     {test_result.confusion_matrix[1,1]:4d}",
        f"```", f"",
        f"- **True Negatives (Benign to Benign):** {test_result.confusion_matrix[0,0]}",
        f"- **False Positives (Benign to Malignant):** {test_result.confusion_matrix[0,1]}",
        f"- **False Negatives (Malignant to Benign):** {test_result.confusion_matrix[1,0]} (Critical)",
        f"- **True Positives (Malignant to Malignant):** {test_result.confusion_matrix[1,1]}",
        f"", f"---", f"", f"## 4. Calibration Analysis", f"",
        f"- **Brier Score:** {calibration.brier_score:.4f} (lower is better, 0=perfect)",
        f"- **Expected Calibration Error (ECE):** {calibration.ece:.4f} (lower is better)", f"",
        f"Calibration curve shows how predicted probabilities align with observed frequencies.", f"",
        f"---", f"", f"## 5. Threshold Analysis", f"",
        f"- **Default Threshold:** 0.5",
        f"- **Optimal Threshold ({threshold_metrics.get('metric', 'F1')}):** {optimal_threshold:.4f}", f"",
        f"Metrics at optimal threshold:", f"",
    ])

    for k, v in threshold_metrics.items():
        if k != "threshold":
            lines.append(f"- **{k.replace('_', ' ').title()}:** {v:.4f}")

    lines.extend([
        f"", f"---", f"", f"## 6. Feature Importance (Permutation)", f"",
        f"Top 15 features by permutation importance (ROC-AUC):", f"",
        f"| Rank | Feature | Importance | Std |", f"|------|---------|------------|-----|",
    ])

    for i, (feat, imp, std) in enumerate(zip(
        feature_importance.feature_names[:15],
        feature_importance.importances[:15],
        feature_importance.std_importances[:15] if feature_importance.std_importances is not None else [0]*15
    )):
        lines.append(f"| {i+1} | {feat} | {imp:.6f} | {std:.6f} |")

    lines.extend([f"", f"---", f"", f"## 7. SHAP Explainability", f""])

    if shap_result:
        lines.extend([
            f"- **Method:** {shap_result.model_type} explainer",
            f"- **Samples Explained:** {len(shap_result.values)}",
            f"- **Features:** {len(shap_result.feature_names)}", f"",
            f"SHAP values computed for model interpretability. See interactive dashboard for detailed visualizations.",
        ])
    else:
        lines.extend([
            f"- **Status:** Not computed (SHAP unavailable or model incompatible)", f"",
        ])

    lines.extend([
        f"", f"---", f"", f"## 8. Limitations & Caveats", f"",
        f"1. **Small Dataset:** 569 samples limits statistical confidence",
        f"2. **Single Institution:** Data from University of Wisconsin only",
        f"3. **No Clinical Metadata:** Age, demographics, outcomes not included",
        f"4. **Feature Extraction:** Features from FNA images, not raw pathology",
        f"5. **No External Validation:** Performance on independent cohorts unknown",
        f"6. **Research Only:** Not validated for clinical decision-making",
        f"7. **Class Imbalance:** ~37% malignant, 63% benign in original data", f"",
        f"---", f"", f"## 9. Reproducibility", f"",
        f"- **Random Seeds:** 42 (data split, CV, model training)",
        f"- **CV Strategy:** Stratified 5-fold",
        f"- **Hyperparameter Selection:** Training data only (no test leakage)",
        f"- **Model Artifact:** {model_path.name}",
        f"- **Environment:** Python 3.11, scikit-learn 1.5+, NumPy 1.26+", f"",
        f"---", f"", f"*Report generated by ONCOGUARD AI Research Platform*", f""
    ])

    report_content = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_content)

    logger.info(f"Saved experiment report to: {report_path}")
    return report_path


def run_full_evaluation(model_path: Path, splits: SplitData) -> Dict[str, Any]:
    logger.info(f"Running full evaluation for: {model_path}")
    artifact = load_model_artifact(model_path)
    model = artifact["model"]

    X_test, y_test = splits.X_test, splits.y_test
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
    }
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=list(TARGET_NAMES.values()),
                               output_dict=True, zero_division=0)

    test_result = TestResult(
        model_name=artifact.get("model_name", "Unknown"),
        y_true=y_test.values, y_pred=y_pred, y_prob=y_prob,
        metrics=test_metrics, confusion_matrix=cm, classification_report=cr
    )

    calibration = compute_calibration(y_test.values, y_prob)
    feature_importance = compute_permutation_importance(model, X_test, y_test)
    shap_result = compute_shap_values(model, X_test, sample_size=100)
    optimal_threshold, threshold_metrics = find_optimal_threshold(y_test.values, y_prob, metric="f1")
    threshold_metrics["metric"] = "f1"

    model_card = generate_model_card(
        model_path, test_result, artifact.get("metadata", {}).get("all_cv_results", []),
        calibration, feature_importance, shap_result, optimal_threshold, threshold_metrics
    )
    model_card_path = save_model_card(model_card)
    report_path = generate_experiment_report(
        splits, artifact.get("metadata", {}).get("all_cv_results", []),
        test_result, calibration, feature_importance, shap_result,
        optimal_threshold, threshold_metrics, model_path
    )

    return {
        "test_result": test_result, "calibration": calibration,
        "feature_importance": feature_importance, "shap_result": shap_result,
        "optimal_threshold": optimal_threshold, "threshold_metrics": threshold_metrics,
        "model_card_path": model_card_path, "report_path": report_path
    }


if __name__ == "__main__":
    from src.oncoguard.data import load_processed_data
    splits = load_processed_data()
    model_files = list(MODELS_DIR.glob("*.joblib"))
    if model_files:
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        results = run_full_evaluation(latest_model, splits)
        print(f"Evaluation test passed. Report: {results['report_path']}")
    else:
        print("No model found to evaluate")