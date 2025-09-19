import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load(r"D:\training\model.pkl")
scaler = joblib.load(r"D:\training\scaler.pkl")


st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("🩺 Heart Disease Prediction Model")
st.write("This tool predicts the likelihood of heart disease based on your health metrics.")

# --- Sidebar inputs ---
st.sidebar.header("Enter Your Health Details")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 100, 30)
    gender = st.sidebar.selectbox("Gender", ("Female", "Male"))
    height = st.sidebar.slider("Height (cm)", 120, 220, 170)
    weight = st.sidebar.slider("Weight (kg)", 40, 150, 70)
    ap_hi = st.sidebar.slider("Systolic Blood Pressure (ap_hi)", 80, 200, 120)
    ap_lo = st.sidebar.slider("Diastolic Blood Pressure (ap_lo)", 50, 150, 80)
    cholesterol = st.sidebar.selectbox("Cholesterol Level", ["Normal (1)", "Above Normal (2)", "Well Above Normal (3)"])
    gluc = st.sidebar.selectbox("Glucose Level", ["Normal (1)", "Above Normal (2)", "Well Above Normal (3)"])
    smoke = st.sidebar.radio("Do you smoke?", ("No", "Yes"))
    alco = st.sidebar.radio("Do you consume alcohol?", ("No", "Yes"))
    active = st.sidebar.radio("Are you physically active?", ("No", "Yes"))

    data = {
        "age": age,
        "gender": 1 if gender == "Female" else 2,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": int(cholesterol.strip()[-2]),
        "gluc": int(gluc.strip()[-2]),
        "smoke": 1 if smoke == "Yes" else 0,
        "alco": 1 if alco == "Yes" else 0,
        "active": 1 if active == "Yes" else 0
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# Align with training column order
columns_order = ['active', 'age', 'alco', 'ap_hi', 'ap_lo', 'cholesterol',
                 'gender', 'gluc', 'height', 'smoke', 'weight']
input_df = input_df[columns_order]

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)[0][1]

# --- Output ---
st.subheader("🧠 Model Prediction")

if prediction == 1:
    st.error("⚠️ The model predicts that you MAY HAVE heart disease.")
else:
    st.success("✅ The model predicts that you are UNLIKELY to have heart disease.")

st.metric(label="Prediction Confidence", value=f"{prediction_proba*100:.2f}%")

# Expandable input section
with st.expander("📋 See the input details you provided"):
    st.write(input_df)
