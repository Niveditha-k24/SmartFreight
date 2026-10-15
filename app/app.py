
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# 1. Configure the page
st.set_page_config(
    page_title="Freight Cost Predictor",
    page_icon="🚚",
    layout="centered"
)

# 2. Find the trained model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "linear_regression.pkl"

# 3. Load the trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# 4. Page heading and description
st.title("🚚 Freight Cost Prediction System")
st.write(
    "Estimate freight costs using historical vendor invoice data."
)

st.info(
    "This is an experimental estimate based on historical data, "
    "not a guaranteed shipping quotation."
)

# 5. Collect user inputs
st.subheader("Enter Invoice Details")

with st.form("freight_prediction_form"):
    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=100,
        step=1
    )

    dollars = st.number_input(
        "Invoice Value (Dollars)",
        min_value=0.01,
        value=1000.00,
        step=100.00
    )

    submitted = st.form_submit_button("Predict Freight Cost")

# 6. Predict when the user submits the form
if submitted:
    input_data = pd.DataFrame(
        [[quantity, dollars]],
        columns=["Quantity", "Dollars"]
    )

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.metric(
        label="Predicted Freight Cost",
        value=f"{prediction:,.2f}"
    )

    if prediction < 0:
        st.warning(
            "The model returned a negative estimate. "
            "This is not a valid freight quote; further model "
            "validation is needed."
        )
    else:
        st.success("Prediction generated successfully.")

    st.caption(
        "The estimate uses only Quantity and Dollars. "
        "Actual freight charges may differ."
    )
