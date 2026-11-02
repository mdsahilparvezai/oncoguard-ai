"""
ONCOGUARD AI Assistant - Intelligent conversational assistant for the research platform.

This module provides:
- Modular LLM provider interface (supports multiple providers)
- Project-aware retrieval from local documentation
- Research-result awareness (fetches actual experiment outputs)
- General health-information behavior with safety guardrails
- Source transparency and citation handling
- Conversation management with context
- Offline mode when no LLM provider is configured
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import httpx

from src.oncoguard.data import (
    FEATURE_COLUMNS, TARGET_NAMES, TARGET_MAPPING, get_feature_definitions,
    PROCESSED_DATA_PATH, load_processed_data
)
from src.oncoguard.evaluate import load_model_artifact, MODELS_DIR, REPORTS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_CONTEXT_MESSAGES = 10
MAX_MESSAGE_LENGTH = 4000
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1500
REQUEST_TIMEOUT = 30.0

# Safety patterns
URGENT_SYMPTOMS = [
    r"chest pain", r"difficulty breathing", r"shortness of breath",
    r"severe pain", r"uncontrolled bleeding", r"loss of consciousness",
    r"stroke", r"heart attack", r"suicidal", r"self.harm"
]

MEDICAL_DISCLAIMER = (
    "⚠️ **Important Medical Disclaimer:** I am an AI research assistant, not a healthcare professional. "
    "My responses are for educational and informational purposes only and should never replace "
    "professional medical advice, diagnosis, or treatment. If you have health concerns, "
    "please consult a qualified healthcare provider. For urgent symptoms, seek immediate medical attention."
)

# Project documentation paths
DOCS_DIR = Path(__file__).parent.parent.parent / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_DOCS = {
    "readme": DOCS_DIR / "README.md",
    "architecture": DOCS_DIR / "architecture.md",
    "data_card": DOCS_DIR / "data_card.md",
    "model_card": DOCS_DIR / "model_card.md",
    "methodology": DOCS_DIR / "methodology.md",
    "assistant_behavior": DOCS_DIR / "assistant_behavior.md",
    "privacy": DOCS_DIR / "privacy.md",
    "troubleshooting": DOCS_DIR / "troubleshooting.md",
}


class AssistantMode(Enum):
    """Assistant operating modes."""
    ONLINE = "online"      # LLM provider available
    OFFLINE = "offline"    # No LLM provider, local only


@dataclass
class Message:
    """Conversation message."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class AssistantResponse:
    """Structured assistant response."""
    content: str
    sources: List[Dict[str, str]] = field(default_factory=list)
    confidence: str = "medium"  # "high", "medium", "low", "uncertain"
    mode: AssistantMode = AssistantMode.ONLINE
    disclaimer_shown: bool = False
    suggested_followups: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs
    ) -> str:
        """Generate completion from messages."""
        pass

    @abstractmethod
    def check_connection(self) -> bool:
        """Check if provider is available."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider - synchronous version."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        timeout: float = REQUEST_TIMEOUT
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    @property
    def name(self) -> str:
        return "OpenAI"

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout
            )
        return self._client

    def check_connection(self) -> bool:
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            response = client.get("/models")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"OpenAI connection check failed: {e}")
            return False

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs
    ) -> str:
        if not self.api_key:
            raise RuntimeError("OpenAI API key not configured")

        client = self._get_client()
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        response = client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def close(self):
        if self._client:
            self._client.close()
            self._client = None


class OfflineProvider(LLMProvider):
    """Offline provider - returns predefined responses."""

    def __init__(self):
        self._model = "offline-local"

    @property
    def name(self) -> str:
        return "Offline (Local)"

    @property
    def model(self) -> str:
        return self._model

    def check_connection(self) -> bool:
        return True  # Always available

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs
    ) -> str:
        # Extract last user message
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return self._handle_offline_query(user_msg)

    def _handle_offline_query(self, query: str) -> str:
        """Handle queries in offline mode with predefined responses."""
        query_lower = query.lower()

        # Project-specific queries
        if any(kw in query_lower for kw in ["project", "oncoguard", "what is this", "overview"]):
            return self._project_overview()

        if any(kw in query_lower for kw in ["dataset", "data", "wisconsin", "uci"]):
            return self._dataset_info()

        if any(kw in query_lower for kw in ["model", "algorithm", "classifier", "training"]):
            return self._model_info()

        if any(kw in query_lower for kw in ["feature", "features", "explain feature", "what is"]):
            return self._feature_info(query)

        if any(kw in query_lower for kw in ["metric", "accuracy", "auc", "roc", "performance", "result"]):
            return self._metrics_info()

        if any(kw in query_lower for kw in ["explain", "shap", "interpret", "importance"]):
            return self._explainability_info()

        if any(kw in query_lower for kw in ["limitation", "caveat", "weakness"]):
            return self._limitations_info()

        if any(kw in query_lower for kw in ["how to run", "install", "setup", "requirements"]):
            return self._setup_info()

        # General ML/educational queries
        if any(kw in query_lower for kw in ["what is machine learning", "ml ", "supervised", "classification"]):
            return self._ml_basics(query)

        if any(kw in query_lower for kw in ["python", "code", "script", "pandas", "sklearn"]):
            return self._coding_help(query)

        # Health queries - safe general info only
        if any(kw in query_lower for kw in ["breast cancer", "symptom", "risk factor", "screening", "mammogram"]):
            return self._health_info(query)

        # ROC-AUC and metrics
        if any(kw in query_lower for kw in ["roc-auc", "roc auc", "auc score", "what is auc"]):
            return self._roc_auc_explanation()

        # Default
        return self._default_response()

    def _project_overview(self) -> str:
        return (
            "**ONCOGUARD AI** is an educational research platform for breast tumor classification.\n\n"
            "**Components:**\n"
            "1. **Research Laboratory** - ML pipeline using Wisconsin Diagnostic Breast Cancer dataset (UCI)\n"
            "2. **Intelligent Assistant** - This conversational interface\n\n"
            "**Current Status:** Tabular baseline models (Logistic Regression, SVM, Random Forest, Gradient Boosting) "
            "with stratified 5-fold CV and held-out test evaluation.\n\n"
            "**Key Features:**\n"
            "- Leakage-resistant preprocessing\n"
            "- SHAP explainability\n"
            "- Calibration analysis\n"
            "- Model cards and experiment reports\n"
            "- Research-only disclaimer (not for clinical use)\n\n"
            "Ask me about: dataset, models, features, metrics, limitations, or how to run the project."
        )

    def _dataset_info(self) -> str:
        return (
            "**Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)**\n\n"
            "- **Source:** UCI Machine Learning Repository\n"
            "- **URL:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic\n"
            "- **Samples:** 569\n"
            "- **Features:** 30 numerical (10 nuclear measurements x 3 statistics: mean, SE, worst)\n"
            "- **Target:** Binary diagnosis (M=Malignant, B=Benign)\n"
            "- **Class Distribution:** ~37% Malignant, ~63% Benign\n\n"
            "**Features describe:** Radius, texture, perimeter, area, smoothness, "
            "compactness, concavity, concave points, symmetry, fractal dimension\n\n"
            "**Important:** Features are extracted from digitized fine-needle aspirate (FNA) images, "
            "not raw histopathology slides. This is a benchmark dataset for ML research, not a clinical tool."
        )

    def _model_info(self) -> str:
        return (
            "**Tabular Baseline Models**\n\n"
            "1. **DummyClassifier (stratified)** - Baseline\n"
            "2. **Logistic Regression** - L2 regularized, balanced class weights\n"
            "3. **Support Vector Machine** - RBF kernel, probability calibration\n"
            "4. **Random Forest** - Balanced class weights, permutation importance\n"
            "5. **Gradient Boosting** - Default sklearn implementation\n\n"
            "**Methodology:**\n"
            "- Stratified 5-fold cross-validation\n"
            "- Hyperparameter tuning on training data only (no test leakage)\n"
            "- Primary metric: ROC-AUC\n"
            "- Final evaluation on held-out 20% test set\n"
            "- Models saved with joblib, reload-verified\n\n"
            "The best model is selected by CV ROC-AUC, then evaluated on the untouched test set."
        )

    def _feature_info(self, query: str) -> str:
        definitions = get_feature_definitions()

        # Try to match specific feature
        query_lower = query.lower()
        for feat, desc in definitions.items():
            if feat.lower() in query_lower:
                return f"**{feat}**: {desc}"

        # Return top features by importance (generic)
        top_features = [
            "concave_points_worst", "perimeter_worst", "concave_points_mean",
            "radius_worst", "perimeter_mean", "area_worst"
        ]

        lines = ["**Feature Definitions (Top Predictive Features):**\n"]
        for feat in top_features:
            if feat in definitions:
                lines.append(f"- **{feat}**: {definitions[feat]}")

        lines.append(f"\n*Total: {len(definitions)} features. Ask about a specific feature for details.*")
        return "\n".join(lines)

    def _metrics_info(self) -> str:
        return (
            "**Evaluation Metrics**\n\n"
            "**Primary:** ROC-AUC (area under ROC curve)\n"
            "**Secondary:**\n"
            "- Balanced Accuracy (accounts for class imbalance)\n"
            "- Precision (PPV) / Recall (Sensitivity)\n"
            "- F1 Score (harmonic mean)\n"
            "- Average Precision (area under PR curve)\n\n"
            "**Clinical Context:**\n"
            "- **Sensitivity (Recall):** Critical - missing cancer (false negative) is serious\n"
            "- **Specificity:** Important - false positives cause unnecessary procedures\n"
            "- **Calibration:** Brier score & Expected Calibration Error (ECE)\n\n"
            "Thresholds can be optimized for different clinical priorities "
            "(e.g., high sensitivity for screening, high specificity for confirmation)."
        )

    def _explainability_info(self) -> str:
        return (
            "**Explainability Methods**\n\n"
            "1. **Permutation Importance** - Model-agnostic, measures feature contribution to ROC-AUC\n"
            "2. **SHAP (SHapley Additive exPlanations)** - Local & global explanations\n"
            "   - TreeExplainer for Random Forest/Gradient Boosting\n"
            "   - KernelExplainer fallback for other models\n\n"
            "**Key Insights:**\n"
            "- `concave_points_worst` and `perimeter_worst` typically most important\n"
            "- Features represent nuclear morphology from FNA images\n"
            "- SHAP values show direction and magnitude of feature influence\n\n"
            "Access via the Explainability Laboratory in the dashboard."
        )

    def _limitations_info(self) -> str:
        return (
            "**Key Limitations**\n\n"
            "1. **Small Dataset:** 569 samples limits statistical confidence\n"
            "2. **Single Institution:** University of Wisconsin only - limited generalizability\n"
            "3. **No Clinical Metadata:** Age, demographics, treatment outcomes unknown\n"
            "4. **Feature Source:** Extracted from FNA images, not raw histopathology\n"
            "5. **No External Validation:** Independent cohort performance unknown\n"
            "6. **Research Only:** Not validated for clinical decision-making\n"
            "7. **Class Imbalance:** ~37% malignant requires balanced metrics\n\n"
            "**Ethical:** False negatives (missed cancer) carry serious consequences. "
            "Model must never be used for clinical diagnosis without rigorous validation "
            "and regulatory approval."
        )

    def _setup_info(self) -> str:
        return (
            "**Setup Instructions**\n\n"
            "```bash\n"
            "# Clone and enter project\n"
            "cd oncoguard-ai\n\n"
            "# Create virtual environment\n"
            "python -m venv .venv\n"
            ".venv\\Scripts\\activate  # Windows\n\n"
            "# Install dependencies\n"
            "pip install -r requirements.txt\n\n"
            "# Run data pipeline\n"
            "python -m src.oncoguard.data\n\n"
            "# Train models\n"
            "python -m src.oncoguard.train\n\n"
            "# Evaluate\n"
            "python -m src.oncoguard.evaluate\n\n"
            "# Launch dashboard\n"
            "streamlit run src/oncoguard/app.py\n"
            "```\n\n"
            "**Optional - LLM Assistant:**\n"
            "Set `OPENAI_API_KEY` environment variable for AI-powered responses.\n"
            "Without it, the assistant runs in offline mode with predefined responses."
        )

    def _ml_basics(self, query: str) -> str:
        if "supervised" in query.lower():
            return (
                "**Supervised Learning** learns from labeled examples (input→output pairs).\n"
                "Classification predicts discrete categories (e.g., Benign/Malignant).\n"
                "Regression predicts continuous values."
            )
        if "classification" in query.lower():
            return (
                "**Classification** assigns inputs to discrete classes.\n"
                "Binary: 2 classes (e.g., cancer/no cancer).\n"
                "Multi-class: >2 classes.\n"
                "Metrics: Accuracy, Precision, Recall, F1, ROC-AUC."
            )
        return (
            "**Machine Learning** enables computers to learn patterns from data.\n"
            "Types: Supervised (labeled), Unsupervised (unlabeled), Reinforcement (rewards).\n"
            "ONCOGUARD uses supervised binary classification."
        )

    def _coding_help(self, query: str) -> str:
        return (
            "**Coding Help** - I can explain concepts but can't execute code.\n\n"
            "Common patterns in this project:\n"
            "- `sklearn.pipeline.Pipeline` for leakage-resistant preprocessing\n"
            "- `StratifiedKFold` for balanced CV splits\n"
            "- `permutation_importance` for model-agnostic feature importance\n"
            "- `shap.TreeExplainer` for tree model explanations\n"
            "- `joblib` for model serialization\n\n"
            "Ask specific questions about any part of the codebase!"
        )

    def _roc_auc_explanation(self) -> str:
        return (
            "**ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**\n\n"
            "**What it measures:** The ability of a classifier to distinguish between classes.\n"
            "- AUC = 1.0: Perfect classification\n"
            "- AUC = 0.5: Random guessing\n"
            "- AUC < 0.5: Worse than random (invert predictions)\n\n"
            "**How it works:**\n"
            "1. Vary the classification threshold from 0 to 1\n"
            "2. Plot True Positive Rate (Sensitivity) vs False Positive Rate (1-Specificity)\n"
            "3. Area under this curve = AUC\n\n"
            "**Why it's preferred for imbalanced data:**\n"
            "- Accuracy can be misleading (e.g., 95% accuracy by always predicting majority class)\n"
            "- ROC-AUC evaluates ranking ability across ALL thresholds\n"
            "- Insensitive to class distribution changes\n\n"
            "**Our results:** SVM achieved **0.9954 ROC-AUC** on the held-out test set."
        )

    def _health_info(self, query: str) -> str:
        return (
            "**General Breast Cancer Information**\n\n"
            "**Risk Factors:** Age, family history, genetic mutations (BRCA1/2), "
            "hormone exposure, dense breast tissue, prior radiation.\n\n"
            "**Screening:** Mammograms recommended starting age 40-50 (varies by guideline). "
            "Clinical breast exams and self-awareness.\n\n"
            "**Symptoms to Discuss with Doctor:**\n"
            "- New lump or thickening\n"
            "- Change in size/shape/appearance\n"
            "- Skin changes (dimpling, redness)\n"
            "- Nipple changes or discharge\n\n"
            f"{MEDICAL_DISCLAIMER}"
        )

    def _default_response(self) -> str:
        return (
            "I'm the ONCOGUARD AI Assistant (offline mode). I can help with:\n\n"
            "- **Project info:** Overview, dataset, models, features, metrics\n"
            "- **ML concepts:** Classification, metrics, explainability\n"
            "- **Code help:** Pipeline, CV, SHAP, serialization\n"
            "- **General health info:** Risk factors, screening (educational only)\n\n"
            "Try asking: \"What dataset does this use?\" or \"How does the model work?\" "
            "or \"Explain permutation importance.\" or \"What is ROC-AUC?\"\n\n"
            f"{MEDICAL_DISCLAIMER}"
        )


class DocumentRetriever:
    """Retrieves relevant project documentation for RAG."""

    def __init__(self, docs_dir: Path = DOCS_DIR):
        self.docs_dir = docs_dir
        self._cache: Dict[str, str] = {}

    def load_document(self, key: str) -> Optional[str]:
        """Load a project document by key."""
        if key in self._cache:
            return self._cache[key]

        path = PROJECT_DOCS.get(key)
        if path and path.exists():
            content = path.read_text(encoding="utf-8")
            self._cache[key] = content
            return content
        return None

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, str]]:
        """Simple keyword search across documents."""
        query_terms = set(query.lower().split())
        results = []

        for key, path in PROJECT_DOCS.items():
            if path.exists():
                content = self.load_document(key) or ""
                # Simple scoring: count query term matches
                score = sum(1 for term in query_terms if term in content.lower())
                if score > 0:
                    results.append((key, content[:2000]))  # Truncate for context

        results.sort(key=lambda x: -1)  # Already sorted by score implicitly
        return results[:top_k]

    def get_relevant_context(self, query: str, max_chars: int = 4000) -> str:
        """Get relevant document context for a query."""
        results = self.search(query, top_k=3)
        if not results:
            return ""

        context_parts = []
        for key, content in results:
            context_parts.append(f"--- {key.upper()} ---\n{content}")

        combined = "\n\n".join(context_parts)
        return combined[:max_chars]


class ToolRegistry:
    """Registry of read-only tools the assistant can invoke."""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register built-in read-only tools."""
        self.register("get_dataset_info", self._get_dataset_info)
        self.register("get_model_card", self._get_model_card)
        self.register("get_evaluation_metrics", self._get_evaluation_metrics)
        self.register("get_feature_definitions", self._get_feature_definitions)
        self.register("explain_metric", self._explain_metric)
        self.register("get_limitation", self._get_limitation)
        self.register("get_setup_instructions", self._get_setup_instructions)

    def register(self, name: str, func: Callable):
        """Register a tool function."""
        self.tools[name] = func

    def get_tool(self, name: str) -> Optional[Callable]:
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

    # Tool implementations
    def _get_dataset_info(self, **kwargs) -> str:
        return OfflineProvider()._dataset_info()

    def _get_model_card(self, **kwargs) -> str:
        model_cards = list(REPORTS_DIR.glob("model_card_*.json"))
        if model_cards:
            latest = max(model_cards, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                card = json.load(f)
            return json.dumps(card, indent=2)[:3000]
        return "No model card found. Run evaluation first."

    def _get_evaluation_metrics(self, **kwargs) -> str:
        model_cards = list(REPORTS_DIR.glob("model_card_*.json"))
        if model_cards:
            latest = max(model_cards, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                card = json.load(f)
            eval_data = card.get("evaluation", {})
            return json.dumps(eval_data, indent=2)[:3000]
        return "No evaluation metrics found. Run evaluation first."

    def _get_feature_definitions(self, **kwargs) -> str:
        defs = get_feature_definitions()
        return json.dumps(defs, indent=2)

    def _explain_metric(self, metric: str = "", **kwargs) -> str:
        explanations = {
            "roc_auc": "ROC-AUC measures the ability to rank positive samples higher than negative ones. 0.5 = random, 1.0 = perfect.",
            "balanced_accuracy": "Average of sensitivity and specificity. Better than accuracy for imbalanced data.",
            "precision": "Of positive predictions, how many are actually positive (PPV).",
            "recall": "Of actual positives, how many were correctly identified (Sensitivity).",
            "f1": "Harmonic mean of precision and recall.",
            "brier_score": "Mean squared error of predicted probabilities. Lower = better calibrated.",
            "ece": "Expected Calibration Error - average difference between confidence and accuracy across bins.",
        }
        metric_lower = metric.lower()
        for key, exp in explanations.items():
            if key in metric_lower:
                return f"**{key.upper()}**: {exp}"
        return "Available metrics: " + ", ".join(explanations.keys())

    def _get_limitation(self, **kwargs) -> str:
        return OfflineProvider()._limitations_info()

    def _get_setup_instructions(self, **kwargs) -> str:
        return OfflineProvider()._setup_info()


class ConversationManager:
    """Manages conversation history and context."""

    def __init__(self, max_messages: int = MAX_CONTEXT_MESSAGES):
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.session_id = str(uuid.uuid4())[:8]

    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to history."""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        self._trim()

    def _trim(self):
        """Trim history to max messages."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_context(self, include_system: bool = True) -> List[Dict[str, str]]:
        """Get messages formatted for LLM context."""
        context = []
        for msg in self.messages:
            if msg.role == "system" and not include_system:
                continue
            context.append({"role": msg.role, "content": msg.content})
        return context

    def clear(self):
        """Clear conversation history."""
        self.messages = []

    def export(self) -> List[Dict[str, Any]]:
        """Export conversation for saving."""
        return [m.to_dict() for m in self.messages]

    def import_messages(self, messages: List[Dict[str, Any]]):
        """Import conversation history."""
        self.messages = [Message.from_dict(m) for m in messages]


class SafetyGuardrails:
    """Safety checks for medical queries."""

    @staticmethod
    def check_urgent_symptoms(text: str) -> bool:
        """Check if text mentions urgent symptoms requiring immediate care."""
        text_lower = text.lower()
        for pattern in URGENT_SYMPTOMS:
            if re.search(pattern, text_lower):
                return True
        return False

    @staticmethod
    def check_personal_medical_query(text: str) -> bool:
        """Check if user is asking about personal medical situation."""
        personal_patterns = [
            r"i have", r"my ", r"i'm ", r"i am ", r"i feel", r"i noticed",
            r"should i worry", r"am i at risk", r"do i have", r"my diagnosis"
        ]
        text_lower = text.lower()
        for pattern in personal_patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    @staticmethod
    def check_diagnosis_request(text: str) -> bool:
        """Check if user is asking for diagnosis."""
        diag_patterns = [
            r"diagnose", r"do i have cancer", r"is it cancer", r"malignant or benign",
            r"interpret.*result", r"what does.*mean for me", r"my result"
        ]
        text_lower = text.lower()
        for pattern in diag_patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    @staticmethod
    def format_safety_response(trigger: str) -> str:
        """Format appropriate safety response."""
        if trigger == "urgent":
            return (
                "🚨 **URGENT: If you are experiencing a medical emergency, "
                "please call emergency services (911 in US) or go to the nearest emergency department immediately.**\n\n"
                f"{MEDICAL_DISCLAIMER}"
            )
        elif trigger == "personal":
            return (
                "I understand you're asking about a personal health concern. "
                "I cannot provide personal medical assessments or diagnoses. "
                "Please consult a qualified healthcare professional who can evaluate your specific situation.\n\n"
                f"{MEDICAL_DISCLAIMER}"
            )
        elif trigger == "diagnosis":
            return (
                "I cannot interpret research model outputs as clinical diagnoses. "
                "The models in this platform are research benchmarks trained on a specific dataset "
                "and are not validated for clinical use. Please discuss any medical concerns with your doctor.\n\n"
                f"{MEDICAL_DISCLAIMER}"
            )
        return MEDICAL_DISCLAIMER


class OncoguardAssistant:
    """
    Main assistant class integrating all components.

    Provides both online (LLM-powered) and offline modes.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        mode: Optional[AssistantMode] = None,
        system_prompt: Optional[str] = None
    ):
        # Determine mode
        if provider is None:
            # Try to create OpenAI provider
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                temp_provider = OpenAIProvider(api_key=api_key)
                if temp_provider.check_connection():
                    provider = temp_provider
                    self.mode = AssistantMode.ONLINE
                else:
                    provider = OfflineProvider()
                    self.mode = AssistantMode.OFFLINE
            else:
                provider = OfflineProvider()
                self.mode = AssistantMode.OFFLINE
        else:
            self.mode = mode or AssistantMode.ONLINE

        self.provider = provider
        self.conversation = ConversationManager()
        self.retriever = DocumentRetriever()
        self.tools = ToolRegistry()
        self.safety = SafetyGuardrails()

        # System prompt
        self.system_prompt = system_prompt or self._default_system_prompt()

        # Initialize conversation with system prompt
        self.conversation.add_message("system", self.system_prompt)

        logger.info(f"Assistant initialized in {self.mode.value} mode with {self.provider.name}")

    def _default_system_prompt(self) -> str:
        return (
            "You are the ONCOGUARD AI Assistant, an intelligent conversational agent for a "
            "breast cancer research platform. Your role is to help users understand:\n\n"
            "1. **The ONCOGUARD AI project** - architecture, dataset, models, methodology\n"
            "2. **Machine learning concepts** - classification, evaluation, explainability\n"
            "3. **General educational health information** - risk factors, screening (with disclaimers)\n"
            "4. **Code and technical questions** - Python, scikit-learn, PyTorch, etc.\n\n"
            "**Critical Rules:**\n"
            "- NEVER diagnose or provide personal medical advice\n"
            "- NEVER interpret research models as clinical results\n"
            "- ALWAYS distinguish verified project results from general explanations\n"
            "- ALWAYS show medical disclaimer for health-related content\n"
            "- Acknowledge uncertainty - say \"I don't know\" when appropriate\n"
            "- Cite sources when referencing project documents or external information\n"
            "- In offline mode, clearly indicate limitations\n\n"
            "**Tone:** Professional, educational, precise, honest about limitations.\n"
            "**Languages:** English and Hindi (including Hinglish)."
        )

    def chat(self, user_message: str) -> AssistantResponse:
        """
        Process a user message and generate a response.

        Args:
            user_message: User's input message.

        Returns:
            AssistantResponse with content and metadata.
        """
        # Safety checks
        if self.safety.check_urgent_symptoms(user_message):
            return AssistantResponse(
                content=self.safety.format_safety_response("urgent"),
                confidence="high",
                mode=self.mode,
                disclaimer_shown=True,
                metadata={"safety_trigger": "urgent_symptoms"}
            )

        if self.safety.check_diagnosis_request(user_message):
            return AssistantResponse(
                content=self.safety.format_safety_response("diagnosis"),
                confidence="high",
                mode=self.mode,
                disclaimer_shown=True,
                metadata={"safety_trigger": "diagnosis_request"}
            )

        if self.safety.check_personal_medical_query(user_message):
            response_content = self.safety.format_safety_response("personal")
            return AssistantResponse(
                content=response_content,
                confidence="medium",
                mode=self.mode,
                disclaimer_shown=True,
                metadata={"safety_trigger": "personal_medical"}
            )

        # Add user message to history
        self.conversation.add_message("user", user_message)

        # Retrieve relevant project context
        doc_context = self.retriever.get_relevant_context(user_message)

        # Build messages for LLM
        messages = self._build_messages(user_message, doc_context)

        # Generate response
        try:
            content = self.provider.complete(messages)

            # Add assistant response to history
            self.conversation.add_message("assistant", content)

            # Determine confidence and sources
            confidence = self._assess_confidence(user_message, content)
            sources = self._extract_sources(doc_context)

            # Generate follow-up suggestions
            followups = self._suggest_followups(user_message)

            return AssistantResponse(
                content=content,
                sources=sources,
                confidence=confidence,
                mode=self.mode,
                disclaimer_shown=self._needs_disclaimer(user_message, content),
                suggested_followups=followups,
                metadata={"doc_context_used": bool(doc_context)}
            )

        except Exception as e:
            logger.error(f"Assistant error: {e}")
            # Fallback to offline provider
            try:
                offline_provider = OfflineProvider()
                content = offline_provider.complete(messages)
                self.conversation.add_message("assistant", content)
                return AssistantResponse(
                    content=content,
                    confidence="medium",
                    mode=AssistantMode.OFFLINE,
                    disclaimer_shown=self._needs_disclaimer(user_message, content),
                    suggested_followups=self._suggest_followups(user_message),
                    metadata={"error": str(e), "fallback": True}
                )
            except Exception:
                return AssistantResponse(
                    content=f"I encountered an error: {str(e)}. Please try rephrasing your question.",
                    confidence="low",
                    mode=self.mode,
                    metadata={"error": str(e)}
                )

    def _build_messages(self, user_message: str, doc_context: str) -> List[Dict[str, str]]:
        """Build message list for LLM including context."""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add document context if available
        if doc_context:
            messages.append({
                "role": "system",
                "content": f"Relevant project documentation:\n{doc_context}"
            })

        # Add conversation history
        for msg in self.conversation.get_context(include_system=False):
            messages.append(msg)

        # Add current user message
        if messages[-1]["role"] != "user" or messages[-1]["content"] != user_message:
            messages.append({"role": "user", "content": user_message})

        return messages

    def _assess_confidence(self, query: str, response: str) -> str:
        """Assess confidence level of response."""
        query_lower = query.lower()

        # High confidence: factual project queries with doc support
        if any(kw in query_lower for kw in ["dataset", "feature", "model", "metric", "accuracy", "auc"]):
            return "high"

        # High confidence: well-defined ML concepts
        if any(kw in query_lower for kw in ["what is", "define", "explain", "how does"]):
            if any(kw in query_lower for kw in ["roc", "auc", "precision", "recall", "f1", "shap", "permutation"]):
                return "high"

        # Medium: general ML/health education
        if any(kw in query_lower for kw in ["machine learning", "classification", "breast cancer", "screening"]):
            return "medium"

        # Low: speculative or outside scope
        return "low"

    def _extract_sources(self, doc_context: str) -> List[Dict[str, str]]:
        """Extract source citations from doc context."""
        sources = []
        if doc_context:
            for key in PROJECT_DOCS.keys():
                if key.upper() in doc_context:
                    sources.append({
                        "type": "project_document",
                        "name": key.replace("_", " ").title(),
                        "path": str(PROJECT_DOCS[key])
                    })
        return sources

    def _needs_disclaimer(self, query: str, response: str) -> bool:
        """Check if medical disclaimer should be shown."""
        health_keywords = [
            "breast cancer", "cancer", "tumor", "malignant", "benign",
            "screening", "mammogram", "biopsy", "diagnosis", "symptom",
            "risk factor", "treatment", "oncology", "oncologist"
        ]
        query_lower = query.lower()
        response_lower = response.lower()
        return any(kw in query_lower or kw in response_lower for kw in health_keywords)

    def _suggest_followups(self, query: str) -> List[str]:
        """Suggest relevant follow-up questions."""
        query_lower = query.lower()

        if "dataset" in query_lower or "data" in query_lower:
            return [
                "What are the feature definitions?",
                "How was the data split?",
                "What preprocessing was applied?"
            ]
        if "model" in query_lower or "algorithm" in query_lower:
            return [
                "Which model performed best?",
                "How was hyperparameter tuning done?",
                "Show me the confusion matrix"
            ]
        if "feature" in query_lower or "importance" in query_lower:
            return [
                "Explain SHAP values",
                "What is permutation importance?",
                "Which features are most predictive?"
            ]
        if "metric" in query_lower or "accuracy" in query_lower or "auc" in query_lower:
            return [
                "Why use balanced accuracy?",
                "Explain ROC-AUC vs PR-AUC",
                "What threshold was chosen?"
            ]
        if "limitation" in query_lower:
            return [
                "What about external validation?",
                "How could this be improved?",
                "What clinical steps would be needed?"
            ]

        return [
            "What dataset does this use?",
            "Which model performed best?",
            "Explain the evaluation metrics",
            "What are the limitations?"
        ]

    def reset_conversation(self):
        """Start a new conversation."""
        self.conversation.clear()
        self.conversation.add_message("system", self.system_prompt)

    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation.export()

    def set_history(self, history: List[Dict[str, Any]]):
        """Load conversation history."""
        self.conversation.import_messages(history)

    def get_suggested_starters(self) -> List[str]:
        """Get suggested starter questions."""
        return [
            "What is the ONCOGUARD AI project?",
            "What dataset does this use?",
            "Which model performed best?",
            "Explain the evaluation metrics",
            "What are the key features?",
            "How does SHAP explainability work?",
            "What are the limitations?",
            "How do I run this project?",
            "What is ROC-AUC?",
            "Explain permutation importance"
        ]


# Convenience function for simple usage
def create_assistant(
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> OncoguardAssistant:
    """Create an assistant instance with optional API key."""
    provider = None
    if api_key:
        provider = OpenAIProvider(api_key=api_key, model=model)
    return OncoguardAssistant(provider=provider)


if __name__ == "__main__":
    # Quick test
    assistant = OncoguardAssistant()
    print(f"Mode: {assistant.mode.value}")
    print(f"Provider: {assistant.provider.name}")

    # Test offline responses
    import asyncio
    # We'll just test the offline provider directly
    offline = OfflineProvider()
    test_queries = [
        "What dataset does this use?",
        "What are the limitations?",
        "What is ROC-AUC?",
        "Which model performed best?"
    ]
    for q in test_queries:
        print(f"\nQ: {q}")
        print(f"A: {offline.complete([{'role': 'user', 'content': q}])[:200]}...")