# ONCOGUARD AI

**Explainable Breast Cancer Research Platform + Intelligent AI Assistant**

An educational research platform for breast tumor classification using the Wisconsin Diagnostic Breast Cancer dataset (UCI), featuring multiple ML baselines, comprehensive evaluation, SHAP explainability, and an integrated conversational AI assistant.

## 🎯 Overview

ONCOGUARD AI consists of two complementary components:

| Component | Description |
|-----------|-------------|
| **Research Laboratory** | Complete ML pipeline: data validation, baseline models, cross-validation, held-out test evaluation, explainability, calibration analysis |
| **Intelligent Assistant** | Conversational AI that explains the project, ML concepts, results, and provides general health education (with safety guardrails) |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- NVIDIA GPU (optional, for PyTorch)
- Windows / Linux / macOS

### Installation

```bash
# Clone and enter project
cd oncoguard-ai

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Optional: Configure LLM for AI Assistant
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run the Pipeline

```bash
# 1. Data pipeline (downloads UCI dataset, validates, splits)
python -m src.oncoguard.data

# 2. Train all baseline models
python -m src.oncoguard.train

# 3. Evaluate best model (generates reports, SHAP, calibration)
python -m src.oncoguard.evaluate

# 4. Launch interactive dashboard
streamlit run src/oncoguard/app.py
```

### Run Tests

```bash
pytest tests/ -v
```

## 📊 Features

### Research Laboratory
- **Dataset:** Wisconsin Diagnostic Breast Cancer (569 samples, 30 features, UCI)
- **Models:** Dummy, Logistic Regression, SVM, Random Forest, Gradient Boosting
- **Validation:** Stratified 5-fold CV, leakage-resistant preprocessing
- **Metrics:** Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision
- **Calibration:** Brier Score, Expected Calibration Error (ECE), calibration curves
- **Explainability:** Permutation importance, SHAP (TreeExplainer/KernelExplainer)
- **Threshold Analysis:** Optimal threshold selection (F1, Balanced Accuracy, Youden's J)
- **Artifacts:** Model cards (JSON), experiment reports (Markdown), joblib models

### Intelligent Assistant
- **Online Mode:** OpenAI GPT integration (configurable)
- **Offline Mode:** Predefined responses for project queries, ML concepts, health education
- **Safety:** Medical disclaimer, urgent symptom detection, diagnosis refusal, personal query guardrails
- **RAG:** Retrieval from project documentation (README, data card, model card, methodology)
- **Tools:** Dataset info, model card, metrics, feature definitions, limitation lookup
- **Languages:** English, Hindi, Hinglish

## 📁 Project Structure

```
oncoguard-ai/
├── data/
│   ├── raw/           # Downloaded UCI dataset
│   └── processed/     # Train/val/test splits (parquet)
├── models/            # Saved model artifacts (.joblib)
├── reports/           # Model cards, experiment reports
├── src/oncoguard/
│   ├── __init__.py
│   ├── data.py        # Data loading, validation, splitting
│   ├── train.py       # Model training, CV, hyperparameter tuning
│   ├── evaluate.py    # Evaluation, calibration, SHAP, reports
│   ├── assistant.py   # AI Assistant (LLM + offline + tools)
│   └── app.py         # Streamlit dashboard
├── tests/
│   └── test_baseline.py
├── docs/              # Documentation
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## ⚠️ Important Disclaimers

> **RESEARCH ONLY** — This platform is for educational and research purposes only.
> The models are benchmarks trained on a specific dataset (Wisconsin Diagnostic Breast Cancer, UCI)
> and are **NOT validated for clinical diagnosis or medical decision-making**.
> False negatives (missed cancer) carry serious consequences.
> Always consult qualified healthcare professionals for medical concerns.

### Key Limitations
- Small dataset (569 samples) — limited statistical power
- Single institution (University of Wisconsin) — limited generalizability
- No clinical metadata (age, demographics, outcomes)
- Features from FNA images, not raw histopathology
- No external validation on independent cohorts
- Class imbalance (~37% malignant) requires balanced metrics

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
# LLM Provider (optional)
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional: for compatible APIs

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=false
```

### Streamlit Secrets (`.streamlit/secrets.toml`)

```toml
OPENAI_API_KEY = "your_key_here"
OPENAI_MODEL = "gpt-4o-mini"
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| `docs/architecture.md` | System architecture and data flow |
| `docs/data_card.md` | Dataset documentation (schema, source, licensing) |
| `docs/model_card.md` | Model documentation (training, evaluation, limitations) |
| `docs/methodology.md` | Detailed methodology and experimental design |
| `docs/assistant_behavior.md` | Assistant behavior specification and safety rules |
| `docs/privacy.md` | Privacy and security documentation |
| `docs/troubleshooting.md` | Common issues and solutions |

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/oncoguard --cov-report=html

# Run specific test class
pytest tests/test_baseline.py::TestDataSchema -v
```

## 📄 License

This project is for educational/research use. The Wisconsin Diagnostic Breast Cancer dataset is from the UCI Machine Learning Repository (CC BY 4.0). Code is provided as-is for educational purposes.

## 🤝 Contributing

This is a personal research project. For questions or issues, please refer to the documentation in `docs/`.

---

**Built with:** Python, scikit-learn, PyTorch, SHAP, Plotly, Streamlit