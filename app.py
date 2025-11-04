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

# 🔍 Debug block: Show model's expected feature names
try:
    st.write("### 🧠 Model expected features:")
    st.write(list(model.feature_names_in_))
except Exception as e:
    st.write("Model feature names not available:", e)


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

# ✅ Corrected Preprocessing Function
def preprocess(df):
    df = df.copy()

    # Encode categorical variables to match training model format
    df["sex_male"] = df["sex"].map({"male": 1, "female": 0})
    df["smoker_yes"] = df["smoker"].map({"yes": 1, "no": 0})

    # One-hot encode 'region'
    region_dummies = pd.get_dummies(df["region"], prefix="region")
    df = pd.concat([df, region_dummies], axis=1)

    # Drop original categorical columns
    df.drop(["sex", "smoker", "region"], axis=1, inplace=True)

    # Ensure all expected columns exist
    expected_cols = [
        "age", "bmi", "children",
        "sex_male", "smoker_yes",
        "region_northeast", "region_northwest",
        "region_southeast", "region_southwest"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    return df[expected_cols]

# Prediction
if st.button("🔮 Predict Charges"):
    X = preprocess(df)
    try:
        prediction = model.predict(X)[0]
        st.success(f"💸 Estimated Insurance Charges: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"❌ Error while predicting: {e}")
