# Import required libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Title of the Web App
st.title("Diabetes Prediction App")

# Load the trained model
model_path = 'trained_model.sav'
try:
    with open(model_path, 'rb') as file:
        classifier = pickle.load(file)
except FileNotFoundError:
    st.error("⚠️ Model file not found! Please train and save the model first.")
    st.stop()

# User Input Form
st.header("Enter the Patient's Details")
Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
Glucose = st.number_input("Glucose Level", min_value=0, max_value=300)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
Insulin = st.number_input("Insulin Level", min_value=0, max_value=900)
BMI = st.number_input("BMI", min_value=0.0, max_value=60.0, step=0.1)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
Age = st.number_input("Age", min_value=0, max_value=120)

# Prediction Button
if st.button("Predict Diabetes"):
    # Convert input to numpy array
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, 
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    # Define column names
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    # Convert to DataFrame with column names
    input_data_df = pd.DataFrame(input_data, columns=column_names)

    # Get Prediction
    prediction = classifier.predict(input_data_df)

    # Show Result
    if prediction[0] == 0:
        st.success("✅ The person is **NOT diabetic**")
    else:
        st.error("⚠️ The person is **diabetic**")

