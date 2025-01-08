# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:46:59 2025

@author: Dell
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier

# Load the trained model (Pipeline)
model = joblib.load('chrunp.pkl')

# Set the title of the application
st.title("Customer Churn Prediction")

# Define the prediction function
def predict(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'CreditScore': [CreditScore],
        'Geography': [Geography],
        'Gender': [Gender],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'EstimatedSalary': [EstimatedSalary]
    })

    # Preprocess and predict with the model (assuming the pipeline handles preprocessing)
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.markdown('### Enter the customer details to predict churn')

    # Create input fields for user input
    CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    Geography = st.selectbox("Geography", options=["France", "Germany", "Spain"])
    Gender = st.selectbox("Gender", options=["Male", "Female"])
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    Tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3)
    Balance = st.number_input("Balance", min_value=0.0, value=10000.0)
    NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    HasCrCard = st.selectbox("Has Credit Card", options=[0, 1], index=1)
    IsActiveMember = st.selectbox("Is Active Member", options=[0, 1], index=1)
    EstimatedSalary = st.number_input("Estimated Salary", min_value=10000.0, value=50000.0)

    # Predict on button click
    if st.button("Predict Churn"):
        prediction = predict(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
        
        if prediction == 0:
            st.success("This customer is likely to stay!")
        else:
            st.warning("This customer is likely to churn!")

# Run the app
if __name__ == "__main__":
    main()
