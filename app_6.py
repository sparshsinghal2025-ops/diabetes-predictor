import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("🏥 Diabetes Risk Prediction System")
st.write("Enter patient clinical data below to evaluate diabetes risk.")

# Load machine learning artifacts safely
@st.cache_resource
def load_pipeline():
    with open("DiabetesPrediction.pkl", "rb") as f:
        return pickle.load(f)

try:
    artifacts = load_pipeline()
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    scaler = artifacts["scaler"]
except FileNotFoundError:
    st.error("Error: 'DiabetesPrediction.pkl' not found. Please run your training script first.")
    st.stop()

# Layout layout fields with 2 columns
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.4, step=0.1)
    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=33)

# Handle prediction actions
if st.button("Analyze Risk Profile", type="primary"):
    # Formulate inputs to mirror training shapes
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age
    }])
    
    # Process zeros just like the training loop if vital features are missing
    zero_invalid_features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_invalid_features:
        if input_data.loc[0, col] == 0:
            input_data.loc[0, col] = np.nan
            
    # Scale and prepare data
    processed_data = preprocessor.transform(input_data)
    scaled_data = scaler.transform(processed_data)
    
    # Generate predictions
    prediction = model.predict(scaled_data)[0]  # Extract single integer element
    probabilities = model.predict_proba(scaled_data)[0]  # Extract single probability array
    
    st.markdown("---")
    st.subheader("Results Analysis")
    
    # Correct array indexing: probabilities[1] is Diabetic risk, probabilities[0] is Non-Diabetic risk
    if prediction == 1:
        risk_percentage = probabilities[1] * 100
        st.error(f"⚠️ **High Risk Detected.** The model predicts a positive outcome with a **{risk_percentage:.1f}%** confidence probability.")
    else:
        healthy_percentage = probabilities[0] * 100
        st.success(f"✅ **Low Risk Detected.** The model predicts a negative outcome with a **{healthy_percentage:.1f}%** confidence probability.")
