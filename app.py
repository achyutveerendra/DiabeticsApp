import streamlit as st
import numpy as np
import pickle

# Load saved model and scaler
with open("model_1.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_1.pkl", "rb") as f:
    scaler = pickle.load(f)

# App title
st.title("Diabetes Prediction Using Logistic Regression")
st.write("Enter patient details to predict diabetes")

# User inputs
Glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)

BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)

Age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prediction button
if st.button("Predict Diabetes"):
    # Arrange input in same order as training
    input_data = np.array([[ Glucose, BMI, Age]])

    # Standardize input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Display result
    if prediction == 1:
        st.error(f"⚠️ Prediction: Diabetic\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Prediction: Not Diabetic\nProbability: {probability:.2f}")
