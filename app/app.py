import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)


# --------------------------------------------------
# Load model files
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "logistic_regression_model.pkl")
)

preprocessor = joblib.load(
    os.path.join(BASE_DIR, "models", "preprocessor.pkl")
)

input_columns = joblib.load(
    os.path.join(BASE_DIR, "models", "input_columns.pkl")
)

baseline_applicant = joblib.load(
    os.path.join(BASE_DIR, "models", "baseline_applicant.pkl")
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("💳 Credit Risk Prediction System")

st.write(
    "Predict the repayment risk of a loan applicant "
    "using a machine learning model."
)

st.divider()


# --------------------------------------------------
# Applicant information
# --------------------------------------------------

st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=180000.0,
        step=10000.0
    )

    credit = st.number_input(
        "Credit Amount",
        min_value=0.0,
        value=500000.0,
        step=10000.0
    )

    annuity = st.number_input(
        "Loan Annuity",
        min_value=0.0,
        value=25000.0,
        step=1000.0
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    employment_years = st.number_input(
        "Employment Years",
        min_value=0.0,
        max_value=60.0,
        value=5.0,
        step=1.0
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["M", "F"]
    )

    education = st.selectbox(
        "Education",
        [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree"
        ]
    )

    family_status = st.selectbox(
        "Family Status",
        [
            "Married",
            "Single / not married",
            "Civil marriage",
            "Separated",
            "Widow"
        ]
    )

    income_type = st.selectbox(
        "Income Type",
        [
            "Working",
            "Commercial associate",
            "Pensioner",
            "State servant",
            "Student",
            "Unemployed"
        ]
    )

    housing_type = st.selectbox(
        "Housing Type",
        [
            "House / apartment",
            "With parents",
            "Municipal apartment",
            "Rented apartment",
            "Office apartment",
            "Co-op apartment"
        ]
    )


# --------------------------------------------------
# Additional important financial features
# --------------------------------------------------

st.subheader("Additional Financial Information")

col1, col2 = st.columns(2)

with col1:
    goods_price = st.number_input(
        "Goods Price",
        min_value=0.0,
        value=450000.0,
        step=10000.0
    )

    ext_source_2 = st.slider(
        "External Source 2",
        0.0,
        1.0,
        0.5,
        0.01
    )

with col2:
    ext_source_3 = st.slider(
        "External Source 3",
        0.0,
        1.0,
        0.5,
        0.01
    )

    flag_emp_phone = st.selectbox(
        "Work Phone Available",
        [0, 1]
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Credit Risk", use_container_width=True):

    # Start with baseline applicant
    applicant = baseline_applicant.copy()

    # Update user-provided values
    applicant.loc[0, "AMT_INCOME_TOTAL"] = income
    applicant.loc[0, "AMT_CREDIT"] = credit
    applicant.loc[0, "AMT_ANNUITY"] = annuity

    applicant.loc[0, "DAYS_BIRTH"] = -(age * 365)
    applicant.loc[0, "DAYS_EMPLOYED"] = -(employment_years * 365)

    applicant.loc[0, "CODE_GENDER"] = gender
    applicant.loc[0, "NAME_EDUCATION_TYPE"] = education
    applicant.loc[0, "NAME_FAMILY_STATUS"] = family_status
    applicant.loc[0, "NAME_INCOME_TYPE"] = income_type
    applicant.loc[0, "NAME_HOUSING_TYPE"] = housing_type

    applicant.loc[0, "AMT_GOODS_PRICE"] = goods_price
    applicant.loc[0, "EXT_SOURCE_2"] = ext_source_2
    applicant.loc[0, "EXT_SOURCE_3"] = ext_source_3
    applicant.loc[0, "FLAG_EMP_PHONE"] = flag_emp_phone

    # Update engineered features
    if "AGE_YEARS" in applicant.columns:
        applicant.loc[0, "AGE_YEARS"] = age

    if "EMPLOYMENT_YEARS" in applicant.columns:
        applicant.loc[0, "EMPLOYMENT_YEARS"] = employment_years

    if "CREDIT_INCOME_RATIO" in applicant.columns:
        applicant.loc[0, "CREDIT_INCOME_RATIO"] = (
            credit / income if income > 0 else 0
        )

    if "ANNUITY_INCOME_RATIO" in applicant.columns:
        applicant.loc[0, "ANNUITY_INCOME_RATIO"] = (
            annuity / income if income > 0 else 0
        )

    # Ensure exact column order
    applicant = applicant[input_columns]

    # Preprocess
    applicant_processed = preprocessor.transform(applicant)

    # Predict probability
    risk_probability = model.predict_proba(
        applicant_processed
    )[0, 1]

    prediction = int(risk_probability >= 0.5)

    st.divider()

    # Results
    st.subheader("Prediction Result")

    st.metric(
        "Predicted Risk Probability",
        f"{risk_probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error("⚠️ HIGH RISK")
        st.write(
            "The model predicts a higher likelihood of repayment difficulties."
        )
    else:
        st.success("✅ LOW RISK")
        st.write(
            "The model predicts a lower likelihood of repayment difficulties."
        )

    st.caption(
        "This prediction is generated by a machine learning model "
        "and should not be treated as a standalone lending decision."
    )