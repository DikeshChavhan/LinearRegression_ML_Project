import streamlit as st
import pandas as pd
import joblib

# --- Page Configuration ---
st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Animated Gradient Background */
    body {
        background: linear-gradient(-45deg, #f3e5f5, #e3f2fd, #fce4ec, #e8f5e9);
        background-size: 400% 400%;
        animation: gradientBG 12s ease infinite;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Title Styling */
    .title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(to right, #1565c0, #8e24aa, #e91e63);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1.5px;
        animation: glow 3s ease-in-out infinite alternate;
        margin-top: 1.5rem;
    }

    @keyframes glow {
        from { text-shadow: 0 0 15px rgba(25,118,210,0.4), 0 0 25px rgba(233,30,99,0.3); }
        to { text-shadow: 0 0 30px rgba(233,30,99,0.6), 0 0 40px rgba(156,39,176,0.5); }
    }

    .subtitle {
        text-align: center;
        font-size: 1.15rem;
        color: #333;
        margin-bottom: 2rem;
    }

    /* Input Container */
    .input-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 2rem;
        border-radius: 1.2rem;
        backdrop-filter: blur(12px);
        background: rgba(255, 255, 255, 0.15);
        box-shadow: 0 6px 25px rgba(0,0,0,0.1);
    }

    .section-header {
        color: #1976d2;
        font-weight: 600;
        text-align: center;
        font-size: 1.4rem;
        margin-bottom: 1rem;
    }

    /* Predict Button */
    .predict-btn button {
        background: linear-gradient(to right, #1976d2, #42a5f5);
        color: white !important;
        font-weight: 600;
        border-radius: 10px;
        width: 100%;
        height: 3rem;
        transition: all 0.3s ease;
    }

    .predict-btn button:hover {
        background: linear-gradient(to right, #1565c0, #64b5f6);
        transform: scale(1.04);
        box-shadow: 0 0 10px rgba(21,101,192,0.4);
    }

    /* Result Box */
    .result-box {
        background: #e8f5e9;
        border-left: 6px solid #43a047;
        padding: 1.2rem;
        border-radius: 1rem;
        margin-top: 2rem;
        text-align: center;
        animation: fadeIn 0.8s ease-in-out;
    }

    /* Summary Box */
    .summary-box {
        background: #fffde7;
        border-left: 6px solid #fbc02d;
        padding: 1rem;
        border-radius: 1rem;
        margin-top: 1.5rem;
        text-align: center;
        animation: fadeIn 0.8s ease-in-out;
    }

    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #444;
        margin-top: 3rem;
        font-size: 1rem;
        font-weight: 500;
    }

    .footer span {
        color: #e91e63;
        font-weight: 700;
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

# --- Input Section ---
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
st.markdown("<h4 class='section-header'>Enter Your Details</h4>", unsafe_allow_html=True)

age = st.number_input("Age", min_value=1, max_value=100, value=30)
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0)
smoker = st.selectbox("Do you smoke?", ["no", "yes"])

data = {"age": [age], "bmi": [bmi], "smoker": [smoker]}
df = pd.DataFrame(data)

# --- Preprocessing ---
def preprocess(df):
    df = df.copy()
    df["smoker_yes"] = df["smoker"].map({"yes": 1, "no": 0})
    df.drop("smoker", axis=1, inplace=True)
    return df[["age", "bmi", "smoker_yes"]]

# --- Predict Button ---
st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)
if st.button("🔮 Predict Insurance Charges"):
    X = preprocess(df)
    try:
        prediction = model.predict(X)[0]

        st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
        st.subheader("🧾 Summary of Your Details")
        st.write(f"**Age:** {age}")
        st.write(f"**BMI:** {bmi}")
        st.write(f"**Smoker:** {'Yes' if smoker == 'yes' else 'No'}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.subheader("💸 Estimated Insurance Charges")
        st.markdown(f"<h3>₹ {prediction:,.2f}</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error while predicting: {e}")
st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <div class='footer'>
        Made with <span>❤️</span> by <b>Dikesh Chavhan</b>
    </div>
""", unsafe_allow_html=True)
