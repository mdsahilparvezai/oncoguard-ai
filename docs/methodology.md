# ONCOGUARD AI - Methodology Documentation

## Experimental Design

### Problem Formulation
- **Task:** Binary classification of breast tumors (Malignant vs Benign)
- **Data:** Wisconsin Diagnostic Breast Cancer dataset (UCI)
- **Features:** 30 numerical features describing nuclear morphology from FNA images
- **Target:** Binary diagnosis (M=Malignant=1, B=Benign=0)
- **Positive Class:** Malignant (cancer present) — critical for medical interpretation

### Data Pipeline

#### 1. Data Loading & Validation
```python
# Source: UCI Repository
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

# Validation checks:
- Column count (32: ID + diagnosis + 30 features)
- Column names match expected schema
- Target values ∈ {M, B}
- No missing values
- ID column unique
- All features numeric
```

#### 2. Data Quality Analysis
- Class distribution computation
- Feature statistics (mean, std, min, max, zeros)
- Missing value check
- Duplicate row check
- SHA256 checksum for reproducibility

#### 3. Train/Validation/Test Split
```
Strategy: Stratified splitting (preserves class proportions)
Random Seed: 42 (fixed for reproducibility)
Proportions: 70% train / 10% validation / 20% test
Implementation: sklearn.model_selection.train_test_split (stratify=y)
```

#### 4. Preprocessing Pipeline
```
StandardScaler
  ├── Fit: X_train only (LEAKAGE PREVENTION)
  ├── Transform: X_train, X_val, X_test
  └── Save: scaler.joblib for inference
```

**Critical:** Scaler fitted ONLY on training data. Validation and test data transformed using training-fitted scaler. This prevents data leakage.

### Model Training

#### Baseline Models

| Model | Rationale |
|-------|-----------|
| DummyClassifier (stratified) | Baseline — predicts class distribution |
| Logistic Regression | Linear baseline, interpretable, calibrated |
| SVM (RBF) | Non-linear kernel, strong on small-medium data |
| Random Forest | Ensemble, handles non-linearity, feature importance |
| Gradient Boosting | Strong predictive performance, feature importance |

#### Hyperparameter Tuning

| Model | Search Method | Search Space | CV Metric |
|-------|--------------|--------------|-----------|
| Logistic Regression | Grid Search | C: [0.001, 0.01, 0.1, 1, 10, 100] | ROC-AUC |
| SVM | Random Search (15 iter) | C, γ | ROC-AUC |
| Random Forest | Random Search (20 iter) | n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features | ROC-AUC |
| Gradient Boosting | Random Search (20 iter) | n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample | ROC-AUC |

**No Leakage:** All hyperparameter tuning uses cross-validation on training data ONLY. Test set never accessed.

#### Cross-Validation Strategy
- **Method:** Stratified K-Fold (K=5)
- **Seed:** 42
- **Stratification:** Maintains class proportions in each fold
- **Scoring:** Multiple metrics (accuracy, balanced_accuracy, precision, recall, f1, roc_auc, average_precision)
- **Primary Selection Metric:** ROC-AUC

#### Calibration
- **Logistic Regression, RF, GB:** Isotonic calibration (CalibratedClassifierCV, cv=3)
- **SVM:** Built-in Platt scaling (probability=True)
- **Dummy:** No calibration

### Model Selection

1. Run CV for all models with hyperparameter tuning
2. Select model with highest mean CV ROC-AUC
3. Break ties: lower std → higher balanced accuracy
4. Refit best model on full train+val data
5. Evaluate ONCE on held-out test set

### Evaluation Protocol

#### Held-Out Test Set
- **Size:** 20% (114 samples)
- **Usage:** Exactly ONE evaluation after model selection
- **Metrics Computed:**
  - Accuracy, Balanced Accuracy
  - Precision, Recall (Sensitivity), F1
  - ROC-AUC, Average Precision (PR-AUC)
  - Confusion Matrix
  - Classification Report

#### Calibration Analysis
- **Brier Score:** Mean squared error of probabilities
- **Expected Calibration Error (ECE):** Weighted avg |confidence - accuracy| across bins
- **Calibration Curve:** Fraction of positives vs mean predicted probability (10 bins)

#### Threshold Optimization
- **Default:** 0.5
- **F1-optimal:** Maximize F1 on test set
- **Balanced Accuracy-optimal:** Maximize (sensitivity + specificity) / 2
- **Youden's J:** Maximize sensitivity + specificity - 1
- **Clinical Context:** High sensitivity for screening, high specificity for confirmation

### Explainability Methods

#### Permutation Importance
- **Method:** sklearn.inspection.permutation_importance
- **Metric:** ROC-AUC decrease when feature shuffled
- **Repeats:** 10 (for stability)
- **Scope:** Global (test set)
- **Advantage:** Model-agnostic

#### SHAP (SHapley Additive exPlanations)
- **TreeExplainer:** For Random Forest, Gradient Boosting (exact, fast)
- **KernelExplainer:** Fallback for Logistic Regression, SVM (approximate)
- **Background Sample:** 100 samples from training data
- **Explanation Sample:** 100 samples from test data
- **Outputs:** SHAP values per feature per sample, base values

### Reporting

#### Model Card (JSON)
Structured metadata including:
- Model details, intended use, training data
- Cross-validation results (all models)
- Held-out test metrics
- Calibration metrics
- Threshold analysis
- Explainability summary
- Limitations, ethical considerations

#### Experiment Report (Markdown)
Human-readable report with:
- Dataset summary
- CV comparison table
- Test metrics with confidence intervals
- Confusion matrix
- Calibration analysis
- Feature importance tables
- Limitations & caveats
- Reproducibility information

### Statistical Considerations

#### Confidence Intervals (Approximate)
For test set metrics (n=114):
- **ROC-AUC:** ±0.02-0.03 (DeLong method)
- **Accuracy:** ±0.04-0.05 (binomial)
- **Sensitivity/Specificity:** ±0.05-0.07 (binomial)

#### Multiple Comparison Note
- 5 models compared on CV
- No formal correction (exploratory research)
- Final test evaluation on single selected model

#### Class Imbalance Handling
- **Metric Selection:** Balanced accuracy, ROC-AUC, F1 (not raw accuracy)
- **Class Weights:** `balanced` for all models supporting it
- **Stratification:** All splits and CV folds stratified
- **Dummy Baseline:** Stratified strategy (predicts class distribution)

### Reproducibility Checklist

- [x] Fixed random seeds (42) for all stochastic operations
- [x] Pinned dependencies (requirements.txt, pyproject.toml)
- [x] Dataset checksum verification
- [x] Train/val/test splits saved to disk
- [x] Scaler saved and reused
- [x] Model artifacts include full metadata
- [x] Reload verification (identical predictions)
- [x] Experiment reports auto-generated
- [x] Model cards auto-generated

### Extensions for Future Work

#### Image Classification (Optional Experiment B)
- **Dataset:** BreakHis or similar histopathology dataset
- **Architecture:** CNN or transfer learning (ResNet, EfficientNet)
- **Splitting:** Patient-level (no data leakage across patients)
- **Augmentation:** Training only (rotation, flip, color jitter)
- **Explainability:** Grad-CAM
- **Separate Pipeline:** Completely independent from tabular pipeline

#### Advanced Tabular Methods
- XGBoost / LightGBM / CatBoost
- Neural networks (tabular)
- Stacking / ensemble methods
- Feature engineering (interactions, ratios)

#### Clinical Validation Pathway
1. External validation on independent cohort
2. Prospective study design
3. Regulatory pathway (FDA SaMD, CE marking)
4. Clinical workflow integration
5. Post-market surveillance