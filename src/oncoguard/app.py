"""
ONCOGUARD AI - Streamlit Research Dashboard

Main application entry point with all research platform pages.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path so src.oncoguard imports work
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.oncoguard.data import (
    run_data_pipeline,
    load_processed_data,
    SplitData,
    FEATURE_COLUMNS,
    TARGET_NAMES,
    get_feature_definitions,
    DatasetInfo,
)
from src.oncoguard.train import (
    run_full_training_pipeline,
    load_model_artifact,
    MODELS_DIR,
    CVResult,
    TestResult,
)
from src.oncoguard.evaluate import (
    run_full_evaluation,
    compute_calibration,
    compute_permutation_importance,
    compute_shap_values,
    find_optimal_threshold,
    generate_model_card,
    REPORTS_DIR,
    CalibrationResult,
    FeatureImportanceResult,
    SHAPResult,
)
from src.oncoguard.assistant import (
    OncoguardAssistant,
    AssistantMode,
    AssistantResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="ONCOGUARD AI - Breast Cancer Research Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a4a6a;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .warning-banner {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .research-notice {
        background: #e7f3ff;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background: #f8f9fa;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: #fff;
        border-bottom: 3px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Utility Functions
# ============================================================================

@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> tuple[SplitData, DatasetInfo]:
    """Load or create dataset splits."""
    try:
        splits = load_processed_data()
        # Create minimal info for display
        info = DatasetInfo(
            n_samples=len(splits.X_train) + len(splits.X_val) + len(splits.X_test),
            n_features=len(FEATURE_COLUMNS),
            class_distribution={
                0: int((splits.y_train == 0).sum() + (splits.y_val == 0).sum() + (splits.y_test == 0).sum()),
                1: int((splits.y_train == 1).sum() + (splits.y_val == 1).sum() + (splits.y_test == 1).sum()),
            },
            missing_values=0,
            duplicate_rows=0,
            feature_stats=splits.X_train.describe().T,
            target_mapping={0: "Benign", 1: "Malignant"},
            checksum="cached"
        )
        return splits, info
    except Exception as e:
        logger.warning(f"Could not load processed data: {e}")
        return None, None


@st.cache_data(show_spinner="Loading model...")
def load_latest_model() -> tuple[Path, Dict[str, Any]]:
    """Load the most recent model artifact."""
    model_files = list(MODELS_DIR.glob("*.joblib"))
    if not model_files:
        return None, None
    latest = max(model_files, key=lambda p: p.stat().st_mtime)
    artifact = load_model_artifact(latest)
    return latest, artifact


@st.cache_data(show_spinner="Loading evaluation...")
def load_latest_evaluation() -> Optional[Dict[str, Any]]:
    """Load the most recent evaluation results."""
    report_files = list(REPORTS_DIR.glob("model_card_*.json"))
    if not report_files:
        return None
    latest = max(report_files, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        return json.load(f)


def get_available_models() -> List[Path]:
    """Get list of available model artifacts."""
    return sorted(MODELS_DIR.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)


def get_available_reports() -> List[Path]:
    """Get list of available reports."""
    return sorted(REPORTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def display_research_notice():
    """Display the research-only disclaimer."""
    st.markdown("""
    <div class="research-notice">
        <strong>🔬 Research Only Notice:</strong> This platform is for educational and research purposes only.
        The models are benchmarks trained on a specific dataset (Wisconsin Diagnostic Breast Cancer, UCI)
        and are <strong>not validated for clinical diagnosis or medical decision-making</strong>.
        False negatives (missed cancer) carry serious consequences. Always consult qualified healthcare professionals.
    </div>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str = None):
    """Display a metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div style="color: #28a745; font-size: 0.85rem;">{delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Page Components
# ============================================================================

def page_home():
    """Home dashboard page."""
    st.markdown('<div class="main-header">ONCOGUARD AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explainable Breast Cancer Research Platform + Intelligent AI Assistant</div>', unsafe_allow_html=True)

    display_research_notice()

    # Quick stats
    splits, info = load_data()
    model_path, artifact = load_latest_model()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        n_total = info.n_samples if info else "—"
        metric_card("Total Samples", str(n_total))

    with col2:
        n_features = info.n_features if info else "—"
        metric_card("Features", str(n_features))

    with col3:
        if artifact:
            test_auc = artifact.get("test_results", {}).get("metrics", {}).get("roc_auc", "—")
            metric_card("Best Test ROC-AUC", f"{test_auc:.4f}" if isinstance(test_auc, float) else str(test_auc))
        else:
            metric_card("Best Test ROC-AUC", "—")

    with col4:
        if artifact:
            model_name = artifact.get("model_name", "—")
            metric_card("Best Model", model_name)
        else:
            metric_card("Best Model", "—")

    st.markdown("---")

    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔬 Run Full Pipeline", use_container_width=True, type="primary"):
            with st.spinner("Running data pipeline..."):
                try:
                    splits, info = run_data_pipeline()
                    st.success("Data pipeline complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if st.button("🏋️ Train Models", use_container_width=True):
            with st.spinner("Training models..."):
                try:
                    splits, _ = load_data()
                    if splits:
                        cv_results, test_result, model_path = run_full_training_pipeline(splits)
                        st.success(f"Training complete! Best: {test_result.model_name}")
                        st.rerun()
                    else:
                        st.error("Run data pipeline first")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col3:
        if st.button("📊 Evaluate Best Model", use_container_width=True):
            with st.spinner("Evaluating..."):
                try:
                    splits, _ = load_data()
                    model_path, _ = load_latest_model()
                    if splits and model_path:
                        results = run_full_evaluation(model_path, splits)
                        st.success("Evaluation complete!")
                        st.rerun()
                    else:
                        st.error("Train models first")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Recent activity
    st.markdown("---")
    st.subheader("📋 Recent Activity")

    models = get_available_models()
    reports = get_available_reports()

    if models:
        st.write("**Recent Models:**")
        for m in models[:5]:
            st.write(f"- {m.name} ({m.stat().st_size / 1024:.1f} KB)")

    if reports:
        st.write("**Recent Reports:**")
        for r in reports[:5]:
            st.write(f"- {r.name}")

    if not models and not reports:
        st.info("No models or reports yet. Run the pipeline to get started!")


def page_dataset_explorer():
    """Dataset explorer page."""
    st.markdown('<div class="main-header">📊 Dataset Explorer</div>', unsafe_allow_html=True)
    display_research_notice()

    splits, info = load_data()

    if splits is None:
        st.warning("No processed data found. Run the data pipeline first.")
        if st.button("Run Data Pipeline"):
            with st.spinner("Processing..."):
                try:
                    splits, info = run_data_pipeline()
                    st.success("Done!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        return

    # Overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Dataset Overview")
        st.write(f"**Source:** Wisconsin Diagnostic Breast Cancer (UCI Repository)")
        st.write(f"**Total Samples:** {info.n_samples}")
        st.write(f"**Features:** {info.n_features} numerical features")
        st.write(f"**Target:** Binary (Benign=0, Malignant=1)")
        st.write(f"**Class Distribution:** Benign={info.class_distribution[0]}, Malignant={info.class_distribution[1]}")
        st.write(f"**Missing Values:** {info.missing_values}")
        st.write(f"**Duplicate Rows:** {info.duplicate_rows}")

    with col2:
        # Class distribution pie chart
        fig = go.Figure(data=[go.Pie(
            labels=["Benign", "Malignant"],
            values=[info.class_distribution[0], info.class_distribution[1]],
            hole=0.4,
            marker_colors=["#2ecc71", "#e74c3c"]
        )])
        fig.update_layout(title="Class Distribution", height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # Feature distributions
    st.subheader("Feature Distributions")

    # Combine all splits for full view
    X_all = pd.concat([splits.X_train, splits.X_val, splits.X_test])
    y_all = pd.concat([splits.y_train, splits.y_val, splits.y_test])

    # Feature selector
    feature_groups = {}
    for feat in FEATURE_COLUMNS:
        base = "_".join(feat.split("_")[:-1])
        if base not in feature_groups:
            feature_groups[base] = []
        feature_groups[base].append(feat)

    selected_group = st.selectbox("Feature Group", list(feature_groups.keys()))
    selected_features = feature_groups[selected_group]

    # Plot distributions by class
    fig = make_subplots(rows=1, cols=len(selected_features), subplot_titles=selected_features)

    for i, feat in enumerate(selected_features):
        benign_vals = X_all[y_all == 0][feat]
        malignant_vals = X_all[y_all == 1][feat]

        fig.add_trace(go.Histogram(
            x=benign_vals, name="Benign", opacity=0.7,
            marker_color="#2ecc71", legendgroup="Benign", showlegend=(i==0)
        ), row=1, col=i+1)

        fig.add_trace(go.Histogram(
            x=malignant_vals, name="Malignant", opacity=0.7,
            marker_color="#e74c3c", legendgroup="Malignant", showlegend=(i==0)
        ), row=1, col=i+1)

    fig.update_layout(height=400, barmode="overlay", title=f"{selected_group} Features by Class")
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")

    # Compute correlations with target
    corr_with_target = X_all.copy()
    corr_with_target["target"] = y_all
    corr_matrix = corr_with_target.corr()

    # Show top correlated with target
    target_corrs = corr_matrix["target"].drop("target").abs().sort_values(ascending=False)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.loc[target_corrs.index[:15], target_corrs.index[:15]].values,
        x=target_corrs.index[:15],
        y=target_corrs.index[:15],
        colorscale="RdBu",
        zmid=0,
        text=corr_matrix.loc[target_corrs.index[:15], target_corrs.index[:15]].round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    fig.update_layout(title="Top 15 Features by Correlation with Target", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.subheader("Sample Data")
    display_df = pd.concat([X_all.head(10), y_all.head(10).to_frame("target")], axis=1)
    display_df["target"] = display_df["target"].map(TARGET_NAMES)
    st.dataframe(display_df, use_container_width=True)


def page_model_lab():
    """Tabular model laboratory page."""
    st.markdown('<div class="main-header">🏋️ Tabular Model Laboratory</div>', unsafe_allow_html=True)
    display_research_notice()

    splits, _ = load_data()
    model_path, artifact = load_latest_model()

    if splits is None:
        st.warning("No data available. Run data pipeline first.")
        return

    if artifact is None:
        st.warning("No trained models found. Train models first.")
        return

    # Model selector
    models = get_available_models()
    model_options = {m.name: m for m in models}
    selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
    selected_model_path = model_options[selected_model_name]
    selected_artifact = load_model_artifact(selected_model_path)

    # Model info
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Model", selected_artifact.get("model_name", "Unknown"))
    with col2:
        cv_auc = selected_artifact.get("cv_results", {}).get("mean_scores", {}).get("roc_auc", 0)
        metric_card("CV ROC-AUC", f"{cv_auc:.4f}")
    with col3:
        test_auc = selected_artifact.get("test_results", {}).get("metrics", {}).get("roc_auc", 0)
        metric_card("Test ROC-AUC", f"{test_auc:.4f}")

    # Cross-validation results
    st.subheader("Cross-Validation Results")
    all_cv = selected_artifact.get("metadata", {}).get("all_cv_results", [])

    if all_cv:
        cv_df = pd.DataFrame([{
            "Model": r["model_name"],
            "ROC-AUC": f"{r['mean_scores'].get('roc_auc', 0):.4f} (±{r['std_scores'].get('roc_auc', 0):.4f})",
            "Balanced Acc": f"{r['mean_scores'].get('balanced_accuracy', 0):.4f}",
            "Precision": f"{r['mean_scores'].get('precision', 0):.4f}",
            "Recall": f"{r['mean_scores'].get('recall', 0):.4f}",
            "F1": f"{r['mean_scores'].get('f1', 0):.4f}",
        } for r in all_cv])
        st.dataframe(cv_df, use_container_width=True, hide_index=True)

    # Test set evaluation
    st.subheader("Held-Out Test Evaluation")
    test_results = selected_artifact.get("test_results", {})
    test_metrics = test_results.get("metrics", {})

    if test_metrics:
        cols = st.columns(7)
        for i, (metric, value) in enumerate(test_metrics.items()):
            with cols[i % 7]:
                metric_card(metric.replace("_", " ").title(), f"{value:.4f}")

    # Confusion matrix
    cm = test_results.get("confusion_matrix", [[0, 0], [0, 0]])
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred Benign", "Pred Malignant"],
        y=["Actual Benign", "Actual Malignant"],
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    fig.update_layout(title="Confusion Matrix", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Train new model
    st.markdown("---")
    st.subheader("Train New Model")

    if st.button("Train All Baselines", type="primary"):
        with st.spinner("Training all models..."):
            try:
                cv_results, test_result, model_path = run_full_training_pipeline(splits)
                st.success(f"Training complete! Best: {test_result.model_name} (Test ROC-AUC: {test_result.metrics['roc_auc']:.4f})")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")


def page_prediction_sandbox():
    """Research prediction sandbox page."""
    st.markdown('<div class="main-header">🔮 Research Prediction Sandbox</div>', unsafe_allow_html=True)
    display_research_notice()

    st.markdown("""
    <div class="warning-banner">
        <strong>⚠️ Research Sandbox Only:</strong> Predictions from this tool are <strong>not clinical diagnoses</strong>.
        They are generated by research models trained on a benchmark dataset and have not been validated for medical use.
        Do not use for any clinical decision-making.
    </div>
    """, unsafe_allow_html=True)

    model_path, artifact = load_latest_model()

    if artifact is None:
        st.warning("No trained model available. Train a model first.")
        return

    model = artifact["model"]
    scaler = artifact.get("scaler")  # Note: scaler is in data splits, not model artifact directly

    # Load splits to get scaler
    splits, _ = load_data()
    if splits is None:
        st.error("Cannot load data splits for scaling.")
        return

    st.subheader("Input Features")

    # Create input form with feature groups
    feature_groups = {}
    for feat in FEATURE_COLUMNS:
        base = "_".join(feat.split("_")[:-1])
        if base not in feature_groups:
            feature_groups[base] = []
        feature_groups[base].append(feat)

    definitions = get_feature_definitions()

    # Input method selector
    input_method = st.radio("Input Method", ["Manual Entry", "Random Sample", "Preset Cases"], horizontal=True)

    if input_method == "Manual Entry":
        # Manual input with sliders
        input_values = {}
        cols = st.columns(3)

        for i, (group, feats) in enumerate(feature_groups.items()):
            with cols[i % 3]:
                st.write(f"**{group}**")
                for feat in feats:
                    # Get reasonable range from training data
                    train_min = float(splits.X_train[feat].min())
                    train_max = float(splits.X_train[feat].max())
                    train_mean = float(splits.X_train[feat].mean())

                    val = st.number_input(
                        f"{feat}",
                        min_value=train_min,
                        max_value=train_max,
                        value=train_mean,
                        step=(train_max - train_min) / 100,
                        help=definitions.get(feat, "")
                    )
                    input_values[feat] = val

    elif input_method == "Random Sample":
        # Random sample from test set
        if st.button("Generate Random Sample"):
            idx = np.random.randint(0, len(splits.X_test))
            sample = splits.X_test.iloc[idx]
            st.session_state.random_sample = sample.to_dict()
            st.session_state.true_label = splits.y_test.iloc[idx]

        if "random_sample" in st.session_state:
            input_values = st.session_state.random_sample
            st.write("**Random Sample from Test Set**")
            st.write(f"True Label: {TARGET_NAMES[st.session_state.true_label]}")
        else:
            input_values = {feat: float(splits.X_train[feat].mean()) for feat in FEATURE_COLUMNS}

    else:  # Preset Cases
        preset = st.selectbox("Preset Case", [
            "Typical Benign (low values)",
            "Typical Malignant (high values)",
            "Borderline Case"
        ])

        if preset == "Typical Benign (low values)":
            input_values = {feat: float(splits.X_train[splits.y_train == 0][feat].mean()) for feat in FEATURE_COLUMNS}
        elif preset == "Typical Malignant (high values)":
            input_values = {feat: float(splits.X_train[splits.y_train == 1][feat].mean()) for feat in FEATURE_COLUMNS}
        else:
            input_values = {feat: float(splits.X_train[feat].median()) for feat in FEATURE_COLUMNS}

    # Make prediction
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        # Prepare input
        input_df = pd.DataFrame([input_values])[FEATURE_COLUMNS]
        input_scaled = pd.DataFrame(
            splits.scaler.transform(input_df),
            columns=FEATURE_COLUMNS
        )

        # Predict
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0, 1]

        # Display result
        st.markdown("---")
        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            label = TARGET_NAMES[pred]
            color = "#e74c3c" if pred == 1 else "#2ecc71"
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: {color}15; border-radius: 12px; border: 2px solid {color};">
                <div style="font-size: 1rem; color: {color}; font-weight: 600;">PREDICTED CLASS</div>
                <div style="font-size: 3rem; font-weight: 700; color: {color};">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 12px; border: 2px solid #3498db;">
                <div style="font-size: 1rem; color: #3498db; font-weight: 600;">MALIGNANT PROBABILITY</div>
                <div style="font-size: 3rem; font-weight: 700; color: #2c3e50;">{prob:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            # Risk interpretation
            if prob < 0.3:
                risk = "Low"
                risk_color = "#2ecc71"
            elif prob < 0.7:
                risk = "Intermediate"
                risk_color = "#f39c12"
            else:
                risk = "High"
                risk_color = "#e74c3c"

            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: {risk_color}15; border-radius: 12px; border: 2px solid {risk_color};">
                <div style="font-size: 1rem; color: {risk_color}; font-weight: 600;">RISK LEVEL</div>
                <div style="font-size: 3rem; font-weight: 700; color: {risk_color};">{risk}</div>
            </div>
            """, unsafe_allow_html=True)

        # Show feature contributions if SHAP available
        st.markdown("---")
        st.subheader("Feature Values (Scaled)")

        feat_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Value": input_scaled.iloc[0].values,
            "Description": [definitions.get(f, "") for f in FEATURE_COLUMNS]
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)


def page_explainability():
    """Explainability laboratory page."""
    st.markdown('<div class="main-header">🔍 Explainability Laboratory</div>', unsafe_allow_html=True)
    display_research_notice()

    model_path, artifact = load_latest_model()
    splits, _ = load_data()

    if artifact is None or splits is None:
        st.warning("No model or data available. Run pipeline first.")
        return

    model = artifact["model"]

    # Explanation method selector
    method = st.radio("Explanation Method", [
        "Permutation Importance",
        "SHAP Summary",
        "SHAP Dependence",
        "Feature Effects"
    ], horizontal=True)

    if method == "Permutation Importance":
        st.subheader("Permutation Feature Importance (Test Set)")

        with st.spinner("Computing permutation importance..."):
            imp = compute_permutation_importance(model, splits.X_test, splits.y_test, n_repeats=10)

        # Plot top 20
        top_n = st.slider("Top N Features", 5, 30, 20)
        fig = go.Figure(go.Bar(
            x=imp.importances[:top_n][::-1],
            y=imp.feature_names[:top_n][::-1],
            orientation="h",
            marker_color="#3498db",
            error_x=dict(type="data", array=imp.std_importances[:top_n][::-1], visible=True) if imp.std_importances is not None else None
        ))
        fig.update_layout(
            title=f"Top {top_n} Features by Permutation Importance (ROC-AUC)",
            height=600,
            xaxis_title="Importance (ROC-AUC decrease)",
            yaxis_title="Feature"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        imp_df = pd.DataFrame({
            "Rank": range(1, len(imp.feature_names) + 1),
            "Feature": imp.feature_names,
            "Importance": imp.importances,
            "Std": imp.std_importances if imp.std_importances is not None else [0]*len(imp.feature_names)
        })
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

    elif method == "SHAP Summary":
        st.subheader("SHAP Summary Plot")

        with st.spinner("Computing SHAP values..."):
            shap_result = compute_shap_values(model, splits.X_test, sample_size=100)

        if shap_result is None:
            st.warning("SHAP not available for this model type. Try Permutation Importance.")
        else:
            # SHAP beeswarm/summary
            shap_df = pd.DataFrame(shap_result.values, columns=shap_result.feature_names)

            # Mean absolute SHAP per feature
            mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

            top_n = st.slider("Top N Features", 5, 30, 20)
            top_features = mean_abs_shap.index[:top_n].tolist()

            fig = go.Figure()
            for feat in top_features[::-1]:
                fig.add_trace(go.Box(
                    y=[feat] * len(shap_df),
                    x=shap_df[feat].values,
                    name=feat,
                    orientation="h",
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                    marker=dict(size=3, opacity=0.5)
                ))

            fig.update_layout(
                title="SHAP Value Distribution (Beeswarm)",
                height=600,
                showlegend=False,
                xaxis_title="SHAP Value (impact on model output)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Mean |SHAP| bar chart
            fig2 = go.Figure(go.Bar(
                x=mean_abs_shap[:top_n][::-1].values,
                y=mean_abs_shap[:top_n][::-1].index,
                orientation="h",
                marker_color="#9b59b6"
            ))
            fig2.update_layout(
                title="Mean |SHAP| by Feature",
                height=500,
                xaxis_title="Mean |SHAP Value|"
            )
            st.plotly_chart(fig2, use_container_width=True)

    elif method == "SHAP Dependence":
        st.subheader("SHAP Dependence Plots")

        with st.spinner("Computing SHAP values..."):
            shap_result = compute_shap_values(model, splits.X_test, sample_size=100)

        if shap_result is None:
            st.warning("SHAP not available for this model type.")
        else:
            shap_df = pd.DataFrame(shap_result.values, columns=shap_result.feature_names)
            X_sample = pd.DataFrame(shap_result.data, columns=shap_result.feature_names)

            # Feature selector
            feat_x = st.selectbox("Feature (X-axis)", shap_result.feature_names, index=0)
            feat_color = st.selectbox("Color by Feature", ["None"] + shap_result.feature_names, index=0)

            fig = go.Figure()
            if feat_color != "None":
                # Color by another feature
                color_vals = X_sample[feat_color].values
                fig.add_trace(go.Scatter(
                    x=X_sample[feat_x],
                    y=shap_df[feat_x],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=color_vals,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title=feat_color)
                    ),
                    hovertemplate=f"{feat_x}: %{{x:.3f}}<br>SHAP: %{{y:.3f}}<br>{feat_color}: %{{marker.color:.3f}}"
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=X_sample[feat_x],
                    y=shap_df[feat_x],
                    mode="markers",
                    marker=dict(size=6, color="#3498db", opacity=0.6),
                    hovertemplate=f"{feat_x}: %{{x:.3f}}<br>SHAP: %{{y:.3f}}"
                ))

            fig.update_layout(
                title=f"SHAP Dependence: {feat_x}",
                xaxis_title=f"{feat_x} Value",
                yaxis_title=f"SHAP Value for {feat_x}",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    elif method == "Feature Effects":
        st.subheader("Partial Dependence / Feature Effects")

        from sklearn.inspection import PartialDependenceDisplay
        import matplotlib.pyplot as plt

        features_to_plot = st.multiselect(
            "Select Features",
            FEATURE_COLUMNS,
            default=FEATURE_COLUMNS[:3]
        )

        if features_to_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            PartialDependenceDisplay.from_estimator(
                model, splits.X_test, features_to_plot,
                ax=ax, n_jobs=-1, kind="average"
            )
            plt.tight_layout()
            st.pyplot(fig)


def page_reports():
    """Experiment reports page."""
    st.markdown('<div class="main-header">📄 Experiment Reports</div>', unsafe_allow_html=True)
    display_research_notice()

    reports = get_available_reports()

    if not reports:
        st.info("No reports available. Run evaluation to generate reports.")
        return

    # Report selector
    report_options = {f"{r.name} ({r.stat().st_mtime})": r for r in reports}
    selected_name = st.selectbox("Select Report", list(report_options.keys()))
    selected_report = report_options[selected_name]

    with open(selected_report) as f:
        report = json.load(f)

    # Display model card
    st.subheader("Model Card")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Model:** {report.get('model_details', {}).get('name', 'N/A')}")
        st.write(f"**Version:** {report.get('model_details', {}).get('version', 'N/A')}")
        st.write(f"**Date:** {report.get('model_details', {}).get('date', 'N/A')}")
        st.write(f"**Task:** {report.get('model_details', {}).get('task', 'N/A')}")

    with col2:
        st.write(f"**Dataset:** {report.get('model_details', {}).get('dataset', 'N/A')}")
        st.write(f"**Type:** {report.get('model_details', {}).get('type', 'N/A')}")

    # Evaluation
    eval_data = report.get("evaluation", {})
    if eval_data:
        st.subheader("Evaluation Results")

        test_metrics = eval_data.get("held_out_test", {}).get("metrics", {})
        if test_metrics:
            cols = st.columns(len(test_metrics))
            for i, (metric, value) in enumerate(test_metrics.items()):
                with cols[i]:
                    metric_card(metric.replace("_", " ").title(), f"{value:.4f}")

        # Calibration
        cal = eval_data.get("calibration", {})
        if cal:
            st.write(f"**Brier Score:** {cal.get('brier_score', 0):.4f}")
            st.write(f"**Expected Calibration Error:** {cal.get('expected_calibration_error', 0):.4f}")

    # Limitations
    limitations = report.get("limitations", [])
    if limitations:
        st.subheader("Limitations")
        for lim in limitations:
            st.write(f"- {lim}")

    # Ethical considerations
    ethical = report.get("ethical_considerations", [])
    if ethical:
        st.subheader("Ethical Considerations")
        for eth in ethical:
            st.write(f"- {eth}")

    # Full JSON
    with st.expander("View Full JSON"):
        st.json(report)

    # Download button
    st.download_button(
        "Download Model Card (JSON)",
        data=json.dumps(report, indent=2),
        file_name=selected_report.name,
        mime="application/json"
    )


def page_assistant():
    """AI Assistant page."""
    st.markdown('<div class="main-header">🤖 ONCOGUARD AI Assistant</div>', unsafe_allow_html=True)

    # Initialize assistant in session state
    if "assistant" not in st.session_state:
        st.session_state.assistant = OncoguardAssistant()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    assistant = st.session_state.assistant

    # Mode indicator
    mode_color = "#2ecc71" if assistant.mode == AssistantMode.ONLINE else "#f39c12"
    st.markdown(f"""
    <div style="display: inline-block; padding: 0.5rem 1rem; background: {mode_color}15; border: 1px solid {mode_color}; border-radius: 20px; margin-bottom: 1rem;">
        Mode: <strong style="color: {mode_color};">{assistant.mode.value.upper()}</strong> ({assistant.provider.name})
    </div>
    """, unsafe_allow_html=True)

    # Suggested starters
    with st.expander("💡 Suggested Questions", expanded=len(st.session_state.chat_history) == 0):
        starters = assistant.get_suggested_starters()
        cols = st.columns(2)
        for i, starter in enumerate(starters):
            with cols[i % 2]:
                if st.button(starter, key=f"starter_{i}", use_container_width=True):
                    st.session_state.pending_question = starter
                    st.rerun()

    # Chat history
    st.subheader("Conversation")

    # Display messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.write(f"- {src['name']} ({src['type']})")
            if msg.get("followups"):
                st.caption("Follow-ups: " + ", ".join(msg["followups"]))

    # Handle pending question from starters
    if "pending_question" in st.session_state:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question
    else:
        user_input = st.chat_input("Ask about the project, ML concepts, health info, or code...")

    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = assistant.chat(user_input)

            st.markdown(response.content)

            if response.sources:
                with st.expander("Sources"):
                    for src in response.sources:
                        st.write(f"- {src['name']} ({src['type']})")

            if response.suggested_followups:
                st.caption("Follow-ups: " + ", ".join(response.suggested_followups))

        # Add to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response.content,
            "sources": response.sources,
            "followups": response.suggested_followups,
            "confidence": response.confidence,
            "mode": response.mode.value
        })

    # Chat controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            assistant.reset_conversation()
            st.rerun()
    with col2:
        if st.button("💾 Export Chat", use_container_width=True):
            history_json = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button(
                "Download",
                data=history_json,
                file_name=f"oncoguard_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    with col3:
        if st.button("🔄 New Session", use_container_width=True):
            assistant.reset_conversation()
            st.session_state.chat_history = []
            st.rerun()


def page_about():
    """About page with methodology, limitations, privacy."""
    st.markdown('<div class="main-header">ℹ️ About ONCOGUARD AI</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Methodology", "Limitations", "Privacy & Security", "Documentation"])

    with tab1:
        st.subheader("Methodology")
        st.markdown("""
        ### Data Pipeline
        1. **Source:** Wisconsin Diagnostic Breast Cancer dataset from UCI ML Repository
        2. **Validation:** Schema validation, data quality checks, checksum verification
        3. **Splitting:** Stratified 70/10/20 (train/validation/test) with fixed random seed (42)
        4. **Preprocessing:** StandardScaler fit ONLY on training data (leakage prevention)

        ### Model Training
        - **Baselines:** DummyClassifier, Logistic Regression, SVM, Random Forest, Gradient Boosting
        - **CV Strategy:** Stratified 5-fold cross-validation
        - **Hyperparameter Tuning:** Grid/Random search on training data only
        - **Primary Metric:** ROC-AUC
        - **Calibration:** Isotonic calibration for probabilistic models

        ### Evaluation
        - **Test Set:** Held-out 20%, never used during training or tuning
        - **Metrics:** Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision
        - **Calibration:** Brier Score, Expected Calibration Error (ECE)
        - **Threshold Analysis:** Optimal threshold selection (F1, Balanced Accuracy, Youden's J)

        ### Explainability
        - **Permutation Importance:** Model-agnostic feature importance
        - **SHAP:** TreeExplainer for tree models, KernelExplainer fallback
        """)

    with tab2:
        st.subheader("Limitations & Caveats")
        limitations = [
            "**Small Dataset:** 569 samples limits statistical confidence and model complexity",
            "**Single Institution:** Data from University of Wisconsin only - limited geographic/demographic diversity",
            "**No Clinical Metadata:** Age, menopausal status, family history, treatment outcomes unavailable",
            "**Feature Extraction:** Features computed from FNA images, not raw whole-slide histopathology",
            "**No External Validation:** Performance on independent cohorts completely unknown",
            "**Research Only:** Not validated for clinical decision-making, screening, or diagnosis",
            "**Class Imbalance:** ~37% malignant requires careful metric selection (balanced accuracy, ROC-AUC)",
            "**Static Features:** No temporal or longitudinal information",
            "**Benchmark Scope:** Optimized for UCI benchmark performance, not clinical utility"
        ]
        for lim in limitations:
            st.write(f"- {lim}")

        st.markdown("---")
        st.subheader("Ethical Considerations")
        ethics = [
            "**False Negatives:** Missing cancer (predicting benign when malignant) has severe consequences",
            "**Model Bias Unknown:** No demographic data to assess bias across populations",
            "**Human Oversight Required:** Any real-world application requires expert physician review",
            "**Regulatory Gap:** Research benchmarks ≠ FDA-cleared medical devices",
            "**Informed Consent:** Original dataset collected under 1990s consent frameworks"
        ]
        for eth in ethics:
            st.write(f"- {eth}")

    with tab3:
        st.subheader("Privacy & Security")
        st.markdown("""
        ### Data Processing
        - All data processing occurs **locally** by default
        - No patient-identifying information in the dataset (only extracted features)
        - No data uploaded to external services without explicit consent

        ### AI Assistant
        - **Online Mode:** Messages sent to configured LLM provider (OpenAI by default)
        - **Offline Mode:** Fully local, no external calls
        - **No PHI:** Never send protected health information to the assistant
        - **Conversation Storage:** Only in browser session (not persisted server-side)

        ### Secrets Management
        - API keys loaded from environment variables or Streamlit secrets
        - Never hardcoded, logged, or committed to version control
        - Example `.env.example` provided with placeholders

        ### Security Practices
        - Input validation on all user-provided data
        - Model artifacts serialized with joblib (trusted local files only)
        - No arbitrary code execution from user inputs
        - Read-only tools for assistant (no filesystem/shell access)
        """)

    with tab4:
        st.subheader("Documentation")
        st.markdown("""
        ### Key Files
        - `README.md` - Project overview and quick start
        - `docs/architecture.md` - System architecture
        - `docs/data_card.md` - Dataset documentation
        - `docs/model_card.md` - Model documentation
        - `docs/methodology.md` - Detailed methodology
        - `docs/assistant_behavior.md` - Assistant behavior specification
        - `docs/privacy.md` - Privacy and security details
        - `docs/troubleshooting.md` - Common issues and solutions

        ### Running the Project
        ```bash
        # Setup
        python -m venv .venv
        .venv\\Scripts\\activate
        pip install -r requirements.txt

        # Pipeline
        python -m src.oncoguard.data
        python -m src.oncoguard.train
        python -m src.oncoguard.evaluate

        # Dashboard
        streamlit run src/oncoguard/app.py
        ```

        ### Testing
        ```bash
        pytest tests/ -v
        ```

        ### Configuration
        - `.env` - Environment variables (API keys)
        - `.streamlit/secrets.toml` - Streamlit secrets
        - `pyproject.toml` - Project metadata and tool config
        """)


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main application entry point."""

    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🔬 ONCOGUARD AI")
        st.caption("Breast Cancer Research Platform")

        page = st.radio(
            "Navigation",
            [
                "🏠 Home",
                "📊 Dataset Explorer",
                "🏋️ Model Laboratory",
                "🔮 Prediction Sandbox",
                "🔍 Explainability Lab",
                "📄 Reports",
                "🤖 AI Assistant",
                "ℹ️ About"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Status indicators
        st.caption("**System Status**")
        splits, info = load_data()
        model_path, artifact = load_latest_model()

        if splits:
            st.success("✅ Data Loaded")
        else:
            st.error("❌ Data Not Loaded")

        if artifact:
            st.success(f"✅ Model: {artifact.get('model_name', 'Loaded')}")
        else:
            st.error("❌ No Model")

        # Assistant status
        if "assistant" in st.session_state:
            mode = st.session_state.assistant.mode.value
            st.info(f"🤖 Assistant: {mode.title()}")

        st.markdown("---")
        st.caption("v0.1.0 | Research Use Only")

    # Route to pages
    page_map = {
        "🏠 Home": page_home,
        "📊 Dataset Explorer": page_dataset_explorer,
        "🏋️ Model Laboratory": page_model_lab,
        "🔮 Prediction Sandbox": page_prediction_sandbox,
        "🔍 Explainability Lab": page_explainability,
        "📄 Reports": page_reports,
        "🤖 AI Assistant": page_assistant,
        "ℹ️ About": page_about,
    }

    page_func = page_map.get(page)
    if page_func:
        page_func()


if __name__ == "__main__":
    main()