import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# Load the trained model (Pipeline)
model = joblib.load('cpred.pkl')  # Make sure the filename corresponds to your saved model

# Function to preprocess user input and convert it into a DataFrame
def preprocess_input(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    """
    Converts input data into a DataFrame format that the model can accept.
    """
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
    return input_data

# Streamlit App
def main():
    st.title("Customer Churn Prediction")

    # Input fields
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
        # Preprocess the input and make a prediction
        input_data = preprocess_input(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
        prediction = model.predict(input_data)

        # Display the result
        if prediction == 0:
            st.success("This customer is likely to stay!")
        else:
            st.warning("This customer is likely to churn!")

# Run the app
if __name__ == "__main__":
    main()
