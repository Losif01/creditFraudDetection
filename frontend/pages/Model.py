import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.express as px
# ---  Set up paths ---
current_dir = Path(__file__).parent          # frontend/pages/
frontend_dir = current_dir.parent           # frontend/
root_dir = frontend_dir.parent              # project root

# Paths
model_dir = root_dir / "saved_models" / "saved"
x_test_path = root_dir / "src" / "data" / "processed" / "X_test.csv"
y_test_path = root_dir / "src" / "data" / "processed" / "y_test.csv"
results_path = root_dir / "notebooks" / "evaluation_results.json"

# ---  Helper: Load image safely ---
def load_image(image_name):
    img_path = current_dir / image_name
    return str(img_path) if img_path.exists() else None

# ---  Helper: Load model with caching ---
@st.cache_resource
def load_model(model_filename):
    model_path = model_dir / model_filename
    try:
        if not model_path.exists():
            st.error(f"❌ Model file not found: `{model_path.name}`")
            st.info(f"Looking in: `{model_path.parent.resolve()}`")
            available = [f.name for f in model_path.parent.glob("*.pkl")]
            if available:
                st.code("\n".join(available))
            else:
                st.code("No .pkl files found!")
            return None
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# ---  Helper: Load evaluation results ---
@st.cache_data
def load_evaluation_results():
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        st.error(f"❌ Could not load evaluation results: {e}")
        return {}

# ---  Load test data (features and labels) ---
@st.cache_data
def load_test_data():
    try:
        X_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path).values.ravel()  # Flatten to 1D
        return X_test, y_test
    except Exception as e:
        st.error(f"❌ Failed to load processed test data: {e}")
        st.stop()

# ---  Helper: Format metric names ---
def format_metric_name(name):
    return {
        "accuracy": "Accuracy",
        "f1_weighted": "F1-Score (Weighted)",
        "precision": "Precision",
        "recall": "Recall",
        "roc_auc": "ROC-AUC"
    }.get(name, name.title())

# --- Streamlit App ---
st.title(" Credit Card Fraud Detection Model")

st.markdown("""
This demo samples transactions from the **held-out test set** to evaluate how well the model detects real fraud cases.
""")

# ---- Load evaluation results
evaluation_results = load_evaluation_results()

# ---- Select Model
with st.expander("🔧 Choose a Model"):
    model_options = {
        "XGBoost": "xgboost.pkl",
        "Random Forest": "random_forest.pkl",
        "Logistic Regression": "logistic_regression.pkl",
        "SVM": "svm.pkl",
        "MLP": "mlp.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
        "Decision Tree": "decision_tree.pkl"
    }
    selected_label = st.selectbox("Select Model", options=list(model_options.keys()))
    selected_filename = model_options[selected_label]
    model_key = selected_label.replace(" ", "_").lower()  # e.g., "xgboost"

# Load model
model = load_model(selected_filename)
if model is None:
    st.stop()

# Load test data
X_test, y_test = load_test_data()
st.write(f"Loaded test data: `{X_test.shape[0]}` transactions")

# Add index back for selection
X_test = X_test.reset_index(drop=True)
y_test = pd.Series(y_test).reset_index(drop=True)

# ---- Show Model Performance Metrics
if evaluation_results and model_key in evaluation_results:
    st.markdown("### 📊 Model Performance Metrics")
    metrics = evaluation_results[model_key]

    col1, col2, col3 = st.columns(3)
    for i, (key, value) in enumerate(metrics.items()):
        col = [col1, col2, col3][i % 3]
        col.metric(format_metric_name(key), f"{value:.4f}")

    st.markdown("---")
else:
    st.info(f"No evaluation data found for `{selected_label}`.")

# ---- Button to Sample and Predict
st.markdown("### 🎯 Test Model with a Random Test Transaction")
if st.button("🎲 Sample Random Transaction & Predict"):

    # Sample one random row from test set
    idx = np.random.randint(0, len(X_test))
    sample_features = X_test.iloc[idx:idx+1]  # Keep DataFrame shape (1, n_features)
    y_true = y_test[idx]
    transaction_id = idx

    # Model prediction
    try:
        y_pred = model.predict(sample_features)[0]
        prob_fraud = model.predict_proba(sample_features)[0][1]

        # Display transaction info
        st.markdown("### 🔍 Sampled Transaction Details")
        st.write(f"**Transaction ID (in test set)**: `{transaction_id}`")
        st.write(f"**Amount**: ${sample_features['Amount'].values[0]:.2f}")
        st.write(f"**Time**: {sample_features['Time'].values[0]:.0f} seconds")
        st.write(f"**Features (V1–V5)**:")
        st.json({
            "V1": round(sample_features['V1'].values[0], 3),
            "V2": round(sample_features['V2'].values[0], 3),
            "V3": round(sample_features['V3'].values[0], 3),
            "V4": round(sample_features['V4'].values[0], 3),
            "V5": round(sample_features['V5'].values[0], 3)
        })

        # Show prediction vs actual
        st.markdown("### 🤖 Prediction vs Actual")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("True Label", "Fraud" if y_true == 1 else "Not Fraud")

        with col2:
            st.metric("Model Prediction", "Fraud" if y_pred == 1 else "Not Fraud")

        # Result verdict
        if y_pred == y_true:
            st.success("✅ Model prediction is **CORRECT**!")
        else:
            st.error("❌ Model prediction is **WRONG**!")

        # Confidence
        confidence = prob_fraud if y_pred == 1 else 1 - prob_fraud
        st.markdown(f"**Confidence**: `{confidence:.2%}`")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ---- Optional: Full Metrics Comparison
with st.expander("📋 Compare All Models"):
    if not evaluation_results:
        st.warning("No evaluation results available.")
    else:
        st.markdown("### Full Model Comparison")

        df_results = pd.DataFrame(evaluation_results).T
        df_results.index = df_results.index.map(lambda x: x.replace("_", " ").title())
        df_results = df_results.rename_axis("Model").round(4)

        st.dataframe(df_results.style.format("{:.4f}"))

        try:
            fig_roc = px.bar(
                df_results.reset_index(),
                x='Model',
                y='roc_auc',
                title='ROC-AUC Score Comparison',
                color='roc_auc',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        except Exception:
            st.warning("Could not generate ROC-AUC chart.")