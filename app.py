import streamlit as st
import pandas as pd
import joblib

# --- Page Configuration ---
st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

# --- Custom Styling ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom right, #e3f2fd, #fce4ec);
        font-family: 'Poppins', sans-serif;
    }
    .main-container {
        background-color: #ffffff;
        padding: 2.5rem;
        border-radius: 1.5rem;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.1);
        max-width: 650px;
        margin: 2rem auto;
    }
    .title {
        text-align: center;
        font-size: 2.4rem;
        color: #2c3e50;
        font-weight: 700;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #1976d2;
        font-weight: 600;
        margin-top: 1.2rem;
    }
    .predict-btn button {
        background: linear-gradient(to right, #1976d2, #42a5f5);
        color: white !important;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        height: 3rem;
    }
    .predict-btn button:hover {
        background: linear-gradient(to right, #1565c0, #64b5f6);
        transform: scale(1.02);
        transition: 0.2s;
    }
    .result-box {
        background: #e8f5e9;
        border-left: 6px solid #43a047;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1.5rem;
        text-align: center;
    }
    .summary-box {
        background: #f1f8e9;
        border-left: 6px solid #9ccc65;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1.5rem;
    }
    .footer {
        text-align: center;
        color: #777;
        margin-top: 3rem;
        font-size: 0.95rem;
    }
    .footer span {
        color: #e91e63;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.markdown("<div class='title'>💰 Insurance Charges Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Estimate your medical insurance cost using Machine Learning</div>", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load("insurance_model.pkl")

model = load_model()

# --- Main Container ---
with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    st.markdown("<h4 class='section-header'>Enter Your Details</h4>", unsafe_allow_html=True)

    # Input Fields
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0)
    smoker = st.selectbox("Do you smoke?", ["no", "yes"])

    # Convert inputs to dataframe
    data = {
        "age": [age],
        "bmi": [bmi],
        "smoker": [smoker]
    }
    df = pd.DataFrame(data)

    # --- Preprocessing ---
    def preprocess(df):
        df = df.copy()
        df["smoker_yes"] = df["smoker"].map({"yes": 1, "no": 0})
        df.drop("smoker", axis=1, inplace=True)
        return df[["age", "bmi", "smoker_yes"]]

    # --- Prediction Button ---
    st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)
    if st.button("🔮 Predict Insurance Charges"):
        X = preprocess(df)
        try:
            prediction = model.predict(X)[0]

            # Summary of user inputs
            st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
            st.subheader("🧾 Summary of Entered Details")
            st.write(f"**Age:** {age}")
            st.write(f"**BMI:** {bmi}")
            st.write(f"**Smoker:** {'Yes' if smoker == 'yes' else 'No'}")
            st.markdown("</div>", unsafe_allow_html=True)

            # Display result
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.subheader("💸 Estimated Insurance Charges")
            st.markdown(f"<h3>₹ {prediction:,.2f}</h3>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error while predicting: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Footer Section ---
st.markdown("""
    <div class='footer'>
        Made with <span>❤️</span> by <b>Dikesh Chavhan</b>
    </div>
""", unsafe_allow_html=True)
