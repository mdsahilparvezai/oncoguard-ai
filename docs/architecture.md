# ONCOGUARD AI - Architecture Documentation

## System Overview

ONCOGUARD AI is a modular, reproducible research platform with two primary components:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ONCOGUARD AI Platform                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────────────┐   │
│  │  Research Laboratory│    │   Intelligent Assistant     │   │
│  ├─────────────────────┤    ├─────────────────────────────┤   │
│  │ • Data Pipeline     │    │ • LLM Provider Interface    │   │
│  │ • Model Training    │    │ • Offline Provider          │   │
│  │ • Evaluation        │    │ • Document Retriever (RAG)  │   │
│  │ • Explainability    │    │ • Tool Registry             │   │
│  │ • Reporting         │    │ • Safety Guardrails         │   │
│  └──────────┬──────────┘    └──────────────┬──────────────┘   │
│             │                              │                  │
│             └──────────────┬───────────────┘                  │
│                            ▼                                  │
│                 ┌─────────────────────┐                       │
│                 │   Streamlit Dashboard│                       │
│                 │  (Unified Interface) │                       │
│                 └─────────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Module Architecture

### Core Modules (`src/oncoguard/`)

| Module | Responsibility | Key Classes/Functions |
|--------|---------------|----------------------|
| `data.py` | Dataset loading, validation, splitting, preprocessing | `load_raw_data`, `create_splits`, `SplitData`, `DatasetInfo` |
| `train.py` | Model training, cross-validation, hyperparameter tuning | `run_cross_validation`, `run_full_training_pipeline`, `ModelConfig`, `CVResult` |
| `evaluate.py` | Model evaluation, calibration, SHAP, report generation | `run_full_evaluation`, `compute_shap_values`, `generate_model_card` |
| `assistant.py` | Conversational AI with LLM/offline modes, tools, safety | `OncoguardAssistant`, `LLMProvider`, `ToolRegistry`, `SafetyGuardrails` |
| `app.py` | Streamlit dashboard with all research pages | `main()`, page functions |

### Data Flow

```
UCI Repository
      │
      ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw Data   │────▶│  Validation &    │────▶│  Train/Val/Test │
│  (CSV)      │     │  Quality Checks  │     │  Splits (70/10/20)│
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                    ┌──────────────────────────────────┘
                    ▼
         ┌─────────────────────┐
         │  StandardScaler     │
         │  (fit on train only)│
         └──────────┬──────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ Logistic│ │   SVM   │ │Random   │
   │Regression│ │         │ │Forest   │
   └────┬────┘ └────┬────┘ └────┬────┘
        │           │           │
        └───────────┼───────────┘
                    ▼
         ┌─────────────────────┐
         │  Stratified 5-Fold  │
         │  Cross-Validation   │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Best Model Selection│
         │  (by CV ROC-AUC)    │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Held-Out Test Eval │
         │  (20% untouched)    │
         └──────────┬──────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │Metrics  │ │Calibr-  │ │SHAP/    │
   │Report   │ │ation    │ │Perm Imp │
   └────┬────┘ └────┬────┘ └────┬────┘
        │           │           │
        └───────────┼───────────┘
                    ▼
         ┌─────────────────────┐
         │  Model Card (JSON)  │
         │  Experiment Report  │
         └─────────────────────┘
```

## Key Design Principles

### 1. Leakage Prevention
- **Scaler fit only on training data** — `StandardScaler` fitted on `X_train`, applied to val/test
- **Hyperparameter tuning on training data only** — CV never sees test set
- **Model selection by CV metric** — Test set used exactly once for final evaluation
- **ID column explicitly excluded** — No patient identifiers in features

### 2. Reproducibility
- Fixed random seeds (42) for all stochastic operations
- Checksums for dataset verification
- Model artifacts include full metadata (params, CV results, timestamps)
- Reload verification ensures identical predictions

### 3. Modularity
- Each module has single responsibility
- Clear interfaces between modules (dataclasses for data transfer)
- LLM provider abstraction for easy swapping
- Tool registry for assistant capabilities

### 4. Safety by Design
- Medical disclaimer on every health-related response
- Urgent symptom detection → emergency referral
- Diagnosis request refusal
- Personal medical query guardrails
- Offline mode as safe fallback

## Assistant Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│ Safety Guardrails   │──▶ Urgent? ──▶ Emergency Response
│ (urgent, personal,  │
│  diagnosis checks)  │
└──────────┬──────────┘
           │ Safe
           ▼
┌─────────────────────┐
│ Document Retriever  │──▶ Relevant project docs
│ (RAG - keyword)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Tool Registry       │──▶ Execute read-only tools if needed
│ (dataset, metrics,  │
│  features, etc.)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ LLM Provider        │──▶ Generate response
│ (OpenAI / Offline)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Post-Processing     │
│ • Confidence assess │
│ • Source extraction │
│ • Disclaimer check  │
│ • Follow-up suggest │
└──────────┬──────────┘
           │
           ▼
      Response
```

## Dashboard Pages

| Page | Module | Purpose |
|------|--------|---------|
| Home | `app.py` | Overview, quick actions, status |
| Dataset Explorer | `app.py` | Feature distributions, correlations, samples |
| Model Laboratory | `app.py` | CV results, test metrics, confusion matrix |
| Prediction Sandbox | `app.py` | Interactive prediction with research disclaimer |
| Explainability Lab | `app.py` | Permutation importance, SHAP plots |
| Reports | `app.py` | Model cards, experiment reports |
| AI Assistant | `app.py` | Conversational interface |
| About | `app.py` | Methodology, limitations, privacy, docs |

## Storage Structure

```
project_root/
├── data/
│   ├── raw/wdbc.data              # Original UCI CSV
│   └── processed/
│       ├── X_train.parquet        # Scaled training features
│       ├── X_val.parquet          # Scaled validation features
│       ├── X_test.parquet         # Scaled test features
│       ├── y_train.parquet        # Training targets
│       ├── y_val.parquet          # Validation targets
│       ├── y_test.parquet         # Test targets
│       ├── scaler.joblib          # Fitted StandardScaler
│       └── metadata.json          # Split metadata
├── models/
│   └── {model_name}_{timestamp}.joblib  # Model artifacts
├── reports/
│   ├── model_card_{timestamp}.json      # Model cards
│   └── experiment_report_{timestamp}.md # Markdown reports
└── docs/
    ├── architecture.md
    ├── data_card.md
    ├── model_card.md
    ├── methodology.md
    ├── assistant_behavior.md
    ├── privacy.md
    └── troubleshooting.md
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY` — Enable online assistant mode
- `OPENAI_MODEL` — Model selection (default: gpt-4o-mini)
- `OPENAI_BASE_URL` — Custom endpoint (optional)

### Random Seeds
- Data splitting: 42
- Cross-validation: 42
- Model training: 42
- SHAP sampling: 42
- Permutation importance: 42

## Extensibility

### Adding New Models
1. Add `ModelConfig` to `get_model_configs()` in `train.py`
2. Ensure estimator supports `predict_proba` for calibration
3. Run training pipeline

### Adding New Explainability Methods
1. Add function to `evaluate.py`
2. Register in `Explainability Lab` page in `app.py`
3. Update model card generation if needed

### Adding Assistant Tools
1. Add function to `ToolRegistry._register_default_tools()` in `assistant.py`
2. Register with `self.register(name, func)`
3. Tool will be available for function calling (future enhancement)

### Adding LLM Providers
1. Implement `LLMProvider` abstract base class
2. Add to provider selection logic in `OncoguardAssistant.__init__()`
3. Configure via environment variables