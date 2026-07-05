import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
model = pickle.load(open('breast_cancer_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Breast Cancer Prediction System")
st.write("Enter the patient details below.")
#Load dataset
data=pd.read_csv("data.csv")
# Drop unnecessary columns
data = data.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1)
# Get feature names
feature_names=data.columns
#create input boxes

# Create an empty list to store user inputs
user_input = []

st.header("Enter Tumor Details")

# Create one input box for each feature
for feature in feature_names:
    value = st.selectbox(
    label=feature.replace("_", " ").title(),
    options=[i / 10 for i in range(0, 1001)]  # 0.0 to 100.0 in steps of 0.1

    )
    user_input.append(value)

input_data = np.array(user_input).reshape(1, -1)
input_data = scaler.transform(input_data)
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🔴 Malignant (Cancer Detected)")
    else:
        st.success("🟢 Benign (No Cancer Detected)")