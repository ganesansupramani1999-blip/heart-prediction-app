import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("heart_model.pkl")

st.title("❤️ Heart Disease Prediction App")

# --- Inputs (same names as model.feature_names_in_) ---
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
chest_pain = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar >120 mg/dl (1 = True, 0 = False)", [1, 0])
rest_ecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
st_slope = st.selectbox("ST Slope (0 = Upsloping, 1 = Flat, 2 = Downsloping)", [0, 1, 2])

# --- Build input dict with exact names ---
input_dict = {
    "Age": age,
    "Sex": sex,
    "ChestPainType": chest_pain,
    "RestingBP": resting_bp,
    "Cholesterol": chol,
    "FastingBS": fasting_bs,
    "RestingECG": rest_ecg,
    "MaxHR": max_hr,
    "ExerciseAngina": exang,
    "Oldpeak": oldpeak,
    "ST_Slope": st_slope
}

# --- Arrange features in correct order ---
input_list = [input_dict[feat] for feat in model.feature_names_in_]

# --- Predict ---
if st.button("🔍 Predict"):
    features = np.array([input_list], dtype=float)
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ The patient is at risk of Heart Disease.")
    else:
        st.success("✅ The patient is not at risk of Heart Disease.")
