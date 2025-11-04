import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ------------------------------
# Load Trained Model
# ------------------------------
with open("insurance_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Insurance Charges Prediction", page_icon="💰", layout="centered")

# ------------------------------
# App Title
# ------------------------------
st.title("💰 Insurance Charges Prediction App")
st.markdown("This app predicts **medical insurance charges** based on user inputs using a Linear Regression model.")

# ------------------------------
# User Inputs
# ------------------------------
st.header("Enter the Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# ------------------------------
# Preprocess Inputs
# ------------------------------
def preprocess(age, sex, bmi, children, smoker, region):
    # Create DataFrame with one sample
    data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Charges"):
    input_data = preprocess(age, sex, bmi, children, smoker, region)
    prediction = model.predict(input_data)
    st.success(f"💵 Estimated Insurance Charge: **${prediction[0]:,.2f}**")

st.caption("Developed by **Dikesh Chavhan** | Streamlit App powered by Linear Regression")
