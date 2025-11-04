import streamlit as st
import pandas as pd
import joblib

# App configuration
st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")
st.title("💰 Insurance Charges / Cost Predictor")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("insurance_model.pkl")

model = load_model()

# Sidebar inputs
st.sidebar.header("Enter Customer Details")
age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=30)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])

# Convert inputs to dataframe
data = {
    "age": [age],
    "bmi": [bmi],
    "smoker": [smoker]
}
df = pd.DataFrame(data)

# ✅ Final Preprocessing Function (matches your model)
def preprocess(df):
    df = df.copy()
    df["smoker_yes"] = df["smoker"].map({"yes": 1, "no": 0})
    df.drop("smoker", axis=1, inplace=True)
    return df[["age", "bmi", "smoker_yes"]]

# Prediction
if st.button("🔮 Predict Charges"):
    X = preprocess(df)
    try:
        prediction = model.predict(X)[0]
        st.success(f"💸 Estimated Insurance Charges: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"❌ Error while predicting: {e}")
