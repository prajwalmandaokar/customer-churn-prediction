"""
streamlit_app.py
-----------------
Interactive demo: enter a customer's details and get a live churn-risk
prediction from the trained model. Great for a resume/portfolio demo video
or a live link (Streamlit Community Cloud is free to deploy on).

Run:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predict import ChurnPredictor  # noqa: E402

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a customer's profile below to estimate their probability of churning, "
    "using a model trained on 5,000+ telecom customer records."
)


@st.cache_resource
def load_predictor():
    return ChurnPredictor()


try:
    predictor = load_predictor()
except FileNotFoundError:
    st.error(
        "No trained model found. Run `python src/train_model.py` from the project "
        "root first, then relaunch this app."
    )
    st.stop()

st.caption(f"Serving model: **{predictor.best_model_name.replace('_', ' ').title()}**")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"]) == "Yes"
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 145.0, 70.0)
    total_charges = st.number_input(
        "Total Charges to Date ($)", min_value=0.0, value=float(monthly_charges * tenure)
    )

    submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

if submitted:
    customer = {
        "customerID": "APP-PREVIEW",
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    result = predictor.predict_single(customer)
    prob = result["ChurnProbability"]
    pred = result["ChurnPrediction"]

    st.divider()
    if pred == "Yes":
        st.error(f"⚠️ **High churn risk** — estimated probability: **{prob:.1%}**")
    else:
        st.success(f"✅ **Low churn risk** — estimated probability: **{prob:.1%}**")

    st.progress(min(prob, 1.0))
    st.caption(
        "This is a probability estimate from a model trained on synthetic data for "
        "demonstration purposes — not a guarantee of customer behavior."
    )

st.divider()
with st.expander("📊 Model performance (on held-out test data)"):
    import json

    metrics_path = Path(__file__).resolve().parents[1] / "reports" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        best = metrics.get("best_model", "")
        rows = [
            {"model": k.replace("_", " ").title(), **v}
            for k, v in metrics.items()
            if isinstance(v, dict)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Selected model: **{best.replace('_', ' ').title()}** (highest ROC-AUC)")
    else:
        st.write("Run `python src/train_model.py` to generate metrics.")
