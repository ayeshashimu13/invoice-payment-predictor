import streamlit as st
import pandas as pd
import numpy as np
import joblib

reg_model = joblib.load("reg_model.pkl")
clf_model = joblib.load("clf_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("feature_columns.pkl")

vendors    = [c.replace("vendor_name_", "") for c in columns if c.startswith("vendor_name_")]
categories = [c.replace("category_", "")    for c in columns if c.startswith("category_")]
regions    = [c.replace("region_", "")      for c in columns if c.startswith("region_")]

st.title("📄 Invoice Payment Predictor")
st.write("Fill in an invoice and I'll predict when it'll be paid.")

vendor   = st.selectbox("Vendor", sorted(vendors))
category = st.selectbox("Category", sorted(categories))
region   = st.selectbox("Region", sorted(regions))
amount   = st.number_input("Amount", min_value=0.0, value=1000.0)
terms    = st.selectbox("Payment terms (days)", [15, 30, 45, 60, 90])

if st.button("Predict"):
    row = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    row["amount"] = amount
    row["payment_terms"] = terms
    if f"vendor_name_{vendor}" in row: row[f"vendor_name_{vendor}"] = 1
    if f"category_{category}" in row: row[f"category_{category}"] = 1
    if f"region_{region}" in row: row[f"region_{region}"] = 1

    days = reg_model.predict(row)[0]
    late = clf_model.predict(scaler.transform(row))[0]

    st.subheader("Prediction")
    st.write(f"⏱️ Estimated days after due date: **{days:.1f}**")
    if late == 1:
        st.error("🚨 Likely to be paid LATE")
    else:
        st.success("✅ Likely to be paid ON TIME")