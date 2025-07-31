import streamlit as st
import numpy as np
import joblib



st.title("🧠 Credit Card Fraud Detection Model")

st.markdown("""
This page allows you to test our fraud detection model using sample or custom transaction inputs.
""")

# ---- Section 1: Model Description
with st.expander("📌 Model Description"):
    st.markdown("""
    - **Model Used**: XGBoost Classifier  
    - **Goal**: Detect fraudulent credit card transactions based on transaction behavior.  
    - **Training**: Trained on imbalanced data with appropriate resampling techniques.
    """)

# ---- Section 2: Load model
model = joblib.load("model/xgb_model.pkl")  # adjust path if needed

# ---- Section 3: User Input
st.subheader("🧾 Enter Transaction Details")
col1, col2, col3 = st.columns(3)

V1 = col1.number_input("V1", value=0.0)
V2 = col2.number_input("V2", value=0.0)
V3 = col3.number_input("V3", value=0.0)
V4 = col1.number_input("V4", value=0.0)
V5 = col2.number_input("V5", value=0.0)
V6 = col3.number_input("V6", value=0.0)
Amount = st.number_input("Transaction Amount", value=100.0)
Time = st.number_input("Transaction Time", value=50000.0)

input_data = np.array([[Time, V1, V2, V3, V4, V5, V6, Amount]])

# ---- Section 4: Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Prediction: Fraudulent Transaction ({prob*100:.2f}% confidence)")
    else:
        st.success(f"✅ Prediction: Not Fraudulent Transaction ({(1-prob)*100:.2f}% confidence)")

# ---- Section 5: Model Metrics (optional)
with st.expander("📊 View Model Performance Metrics"):
    st.markdown("""
    - **Accuracy**: 0.985  
    - **Recall (Fraud Class)**: 0.91  
    - **Precision (Fraud Class)**: 0.92  
    - **AUC Score**: 0.97  
    - These metrics reflect the model's ability to correctly identify fraudulent transactions.
    """)
    st.image("frontend/pages/roc_curve.png", caption="ROC Curve", use_column_width=True)
