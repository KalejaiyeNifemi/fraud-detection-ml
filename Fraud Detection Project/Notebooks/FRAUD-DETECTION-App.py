import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('rf_model.pkl')

st.title("Fraud Detection System")
st.markdown("Upload a CSV file to predict fraudulent transactions.")

# File upload
uploaded_file = st.file_uploader("Upload CSV:", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", df.head())

    # Run predictions
    predictions = model.predict(df)
    df['Prediction'] = predictions
    st.write("Prediction Results")
    st.dataframe(df)

    fraud_count = df['Prediction'].sum()
    st.success(f"Total Fraudulent Transactions Detected: {fraud_count}")
