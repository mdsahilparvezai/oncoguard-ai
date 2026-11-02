# ONCOGUARD AI - Model Card

## Model Overview

| Property | Value |
|----------|-------|
| **Model Name** | ONCOGUARD Baseline Classifier |
| **Version** | 1.0.0 |
| **Date** | Auto-generated per training run |
| **Type** | Binary Classification |
| **Task** | Breast tumor classification (Malignant vs Benign) |
| **Framework** | scikit-learn 1.5+ |
| **License** | Educational/Research Use Only |

## Intended Use

### Primary Use
- Educational research and benchmarking on Wisconsin Diagnostic Breast Cancer dataset
- ML methodology demonstration (CV, leakage prevention, calibration, explainability)
- Teaching tool for binary classification, medical ML concepts

### Intended Users
- ML researchers and students
- Educators teaching medical AI/ML
- Practitioners learning explainable AI techniques

### Out of Scope (DO NOT USE FOR)
- ❌ Clinical diagnosis or screening
- ❌ Patient risk assessment
- ❌ Treatment decision support
- ❌ Any medical decision-making
- ❌ Deployment in healthcare settings
- ❌ Regulatory submission

## Training Data

| Property | Value |
|----------|-------|
| **Dataset** | Wisconsin Diagnostic Breast Cancer (UCI) |
| **Source** | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| **Total Samples** | 569 |
| **Training Samples** | 398 (70%) |
| **Validation Samples** | 57 (10%) |
| **Test Samples** | 114 (20%) |
| **Features** | 30 numerical (nuclear morphology measurements) |
| **Target** | Binary: Benign (0) vs Malignant (1) |
| **Class Distribution (Train)** | Benign: ~250, Malignant: ~148 |

### Preprocessing
- **ID Column:** Explicitly excluded (no patient identifiers in features)
- **Scaling:** StandardScaler fit on training data ONLY (leakage prevention)
- **Target Encoding:** M→1 (malignant/positive), B→0 (benign/negative)
- **Stratification:** All splits maintain class proportions

## Model Architecture

### Baseline Models Evaluated

| Model | Configuration | Calibration |
|-------|---------------|-------------|
| DummyClassifier | Stratified baseline | None |
| Logistic Regression | L2, C∈[0.001,100], balanced weights | Isotonic |
| SVM (RBF) | C∈[0.1,100], γ∈[scale,auto,0.01,1], balanced | Built-in Platt |
| Random Forest | n_estimators∈[100,500], max_depth∈[None,5-20], balanced | Isotonic |
| Gradient Boosting | n_estimators∈[100,300], lr∈[0.01,0.2], max_depth∈[3,6] | Isotonic |

### Hyperparameter Selection
- **Method:** Grid Search (Logistic) / Random Search (SVM, RF, GB)
- **Search Space:** Defined per model (see train.py)
- **CV Strategy:** Stratified 5-fold (seed=42)
- **Optimization Metric:** ROC-AUC
- **No Test Set Leakage:** Test set never used during tuning

### Best Model Selection
- **Criterion:** Highest mean CV ROC-AUC
- **Tie-breaking:** Lower std, then higher balanced accuracy
- **Final Evaluation:** Single evaluation on held-out test set

## Evaluation Results

*Results are populated after training. Example structure:*

### Cross-Validation (5-Fold Stratified)

| Model | ROC-AUC | Balanced Acc | Precision | Recall | F1 |
|-------|---------|--------------|-----------|--------|-----|
| Dummy | ~0.50 | ~0.50 | ~0.37 | ~0.37 | ~0.37 |
| Logistic Regression | **~0.98-0.99** | ~0.93 | ~0.95 | ~0.93 | ~0.94 |
| SVM | ~0.98-0.99 | ~0.93 | ~0.94 | ~0.93 | ~0.93 |
| Random Forest | ~0.97-0.99 | ~0.92 | ~0.94 | ~0.91 | ~0.92 |
| Gradient Boosting | ~0.98-0.99 | ~0.93 | ~0.95 | ~0.92 | ~0.93 |

### Held-Out Test Set (n=114)

*Metrics computed on untouched 20% test set*

| Metric | Value | 95% CI (approx) |
|--------|-------|-----------------|
| Accuracy | ~0.95 | ±0.04 |
| Balanced Accuracy | ~0.93 | ±0.05 |
| Precision | ~0.94 | ±0.05 |
| Recall (Sensitivity) | ~0.92 | ±0.06 |
| Specificity | ~0.94 | ±0.05 |
| F1 Score | ~0.93 | ±0.05 |
| ROC-AUC | ~0.98 | ±0.02 |
| Average Precision | ~0.97 | ±0.03 |

### Confusion Matrix (Test Set)

| | Predicted Benign | Predicted Malignant |
|---|---|---|
| **Actual Benign** | TN ~70 | FP ~4 |
| **Actual Malignant** | FN ~3 | TP ~37 |

*Exact values vary per training run*

### Calibration

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Brier Score | ~0.05-0.08 | Lower = better calibrated |
| Expected Calibration Error | ~0.02-0.05 | Lower = better calibrated |

Calibration curves show predicted probabilities align well with observed frequencies.

### Threshold Analysis

| Threshold | Sensitivity | Specificity | F1 | Use Case |
|-----------|-------------|-------------|-----|----------|
| 0.5 (default) | ~0.92 | ~0.94 | ~0.93 | Balanced |
| Optimal (F1) | ~0.93 | ~0.93 | ~0.94 | General |
| Optimal (Bal Acc) | ~0.94 | ~0.93 | ~0.93 | Equal weight |
| High Sens (0.3) | ~0.98 | ~0.80 | ~0.88 | Screening |
| High Spec (0.7) | ~0.85 | ~0.98 | ~0.90 | Confirmation |

## Explainability

### Permutation Importance (Test Set, ROC-AUC)

Top predictive features typically:
1. `concave_points_worst`
2. `perimeter_worst`
3. `concave_points_mean`
4. `radius_worst`
5. `perimeter_mean`
6. `area_worst`
7. `concavity_worst`
8. `compactness_worst`
9. `concave_points_se`
10. `radius_mean`

### SHAP Analysis
- **Method:** TreeExplainer (RF, GB) / KernelExplainer (LR, SVM)
- **Scope:** Local (per prediction) and global (feature importance)
- **Interpretation:** Positive SHAP → pushes toward malignant; negative → toward benign

## Limitations

### Data Limitations
1. **Small Dataset** — 569 samples limits statistical power
2. **Single Institution** — University of Wisconsin only
3. **No Demographics** — Age, race, gender, menopausal status unknown
4. **No Outcomes** — Treatment response, survival, recurrence unavailable
5. **Feature Abstraction** — Derived from FNA, not whole-slide histopathology

### Methodological Limitations
1. **No External Validation** — Independent cohort performance unknown
2. **Benchmark Overfitting Risk** — Models tuned for UCI benchmark metrics
3. **Static Features** — No temporal/longitudinal information
4. **Class Imbalance** — 37% malignant requires balanced metrics

### Clinical Limitations
1. **Not Clinically Validated** — No FDA clearance, no clinical trials
2. **False Negative Risk** — Missing cancer has severe consequences
3. **Population Bias** — Unknown generalizability to diverse populations
4. **Feature Gap** — Clinical practice uses imaging, genomics, history — not just 30 nuclear features
5. **Threshold Uncertainty** — Optimal threshold depends on clinical context (screening vs diagnostic)

## Ethical Considerations

| Risk | Mitigation |
|------|------------|
| False negatives (missed cancer) | Explicit warnings; high-sensitivity thresholds for screening context |
| Model bias | Documented unknown; requires diverse validation before any clinical use |
| Automation bias | Research-only disclaimer; human expert oversight required |
| Privacy | No PHI in features; local processing by default |
| Misuse | Clear out-of-scope declarations; license restrictions |

## Fairness & Bias

**Unknown** — Dataset lacks demographic variables (age, race, ethnicity, socioeconomic status). Cannot assess:
- Performance across age groups
- Racial/ethnic disparities
- Gender-specific performance (though breast cancer is predominantly female)
- Socioeconomic bias

**Required before clinical consideration:** External validation on diverse, representative cohorts with demographic metadata.

## Robustness

| Aspect | Status |
|--------|--------|
| Adversarial robustness | Not evaluated |
| Distribution shift | Not evaluated |
| Missing feature handling | Not applicable (no missing values in benchmark) |
| Out-of-distribution detection | Not implemented |

## Monitoring & Maintenance

| Aspect | Approach |
|--------|----------|
| Model versioning | Timestamped joblib artifacts with full metadata |
| Data drift detection | Not implemented (benchmark dataset static) |
| Performance monitoring | Re-run evaluation pipeline |
| Retraining trigger | Manual (new data, new requirements) |

## Artifact Information

| Artifact | Format | Contents |
|----------|--------|----------|
| Model | `.joblib` | Fitted pipeline, CV results, metadata |
| Model Card | `.json` | This document (structured) |
| Experiment Report | `.md` | Human-readable report |
| Scaler | `.joblib` | Fitted StandardScaler |
| Splits | `.parquet` | Train/val/test data |

## Reproducibility

- **Random Seeds:** 42 (all stochastic operations)
- **Environment:** Python 3.11, scikit-learn 1.5+, NumPy 1.26+
- **Dependencies:** Pinned in `requirements.txt` and `pyproject.toml`
- **Verification:** `verify_model_reload()` confirms identical predictions

## Contact & References

- **Platform:** ONCOGUARD AI Research Platform
- **Dataset:** UCI ML Repository (Dua & Graff, 2019)
- **Original Features:** Street et al., 1993 (Nuclear feature extraction)

---

*This model card is auto-generated by the ONCOGUARD AI evaluation pipeline. Values shown are typical ranges; exact values depend on training run.*