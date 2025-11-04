import streamlit as st
import pandas as pd
import joblib

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
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
region = st.sidebar.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# Convert inputs to dataframe
data = {
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
}
df = pd.DataFrame(data)

# Preprocess
def preprocess(df):
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
    df = pd.get_dummies(df, columns=["region"], drop_first=True)
    return df

# Prediction
if st.button("🔮 Predict Charges"):
    X = preprocess(df)
    try:
        prediction = model.predict(X)[0]
        st.success(f"Estimated Insurance Charges: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"Error while predicting: {e}")
