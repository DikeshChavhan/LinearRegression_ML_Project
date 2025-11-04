import streamlit as st
import pandas as pd
from joblib import load
from utils import preprocess_input
from pathlib import Path

st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")
st.title("Insurance Charges / Cost Predictor")

MODEL_PATH = Path(__file__).parent / "insurance_model.pkl"

@st.cache_resource
def load_model(path):
    try:
        return load(path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model(MODEL_PATH)

st.sidebar.header("Manual input")
inputs = {}
feature_list = ["age", "bmi", "charges"]
for col in feature_list:
    if col in ['sex','smoker','region']:
        if col == 'sex':
            inputs[col] = st.sidebar.selectbox("Sex", options=['male','female'])
        elif col == 'smoker':
            inputs[col] = st.sidebar.selectbox("Smoker", options=['no','yes'])
        else:
            inputs[col] = st.sidebar.selectbox("Region", options=['southeast','southwest','northeast','northwest'])
    else:
        default = 30 if col=='age' else 0 if col=='children' else 25.0
        inputs[col] = st.sidebar.number_input(col.capitalize(), value=default)

use_manual = st.sidebar.button("Predict for manual input")
uploaded = st.file_uploader("Upload CSV file", type=['csv'])

def predict_df(df):
    if model is None:
        st.error("Model not loaded.")
        return
    X = preprocess_input(df)
    try:
        preds = model.predict(X)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return
    df_out = df.copy()
    df_out['predicted_charges'] = preds
    return df_out

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Input preview:")
    st.dataframe(df.head())
    out = predict_df(df)
    if out is not None:
        st.write("Predictions:")
        st.dataframe(out)
        st.download_button("Download predictions as CSV", out.to_csv(index=False).encode('utf-8'), "predictions.csv", "text/csv")

if use_manual:
    import pandas as pd
    dfm = pd.DataFrame([inputs])
    st.write("Manual input:")
    st.dataframe(dfm)
    out = predict_df(dfm)
    if out is not None:
        st.metric("Predicted charges", f"₹{out['predicted_charges'].iloc[0]:.2f}")
