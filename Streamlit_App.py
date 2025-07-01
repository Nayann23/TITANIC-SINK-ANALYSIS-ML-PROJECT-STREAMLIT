import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and columns
model = joblib.load("saved_models/svc_model.pkl")
scaler = joblib.load("saved_models/scaler.pkl")
columns = joblib.load("saved_models/columns.pkl")

# Streamlit UI
st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details below to predict if the passenger would have survived.")

# Input Fields
Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Age = st.slider("Age", 0, 100, 30)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
Fare = st.number_input("Fare Paid", 0.0, 500.0, 50.0)
Sex = st.selectbox("Sex", ["male", "female"])
Embarked = st.selectbox("Embarked", ["C", "Q", "S"])
Title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])

# One-hot encode categorical features manually
input_data = pd.DataFrame(columns=columns)
input_data.loc[0] = 0  # initialize all features to zero

# Assign numerical inputs
input_data.at[0, "Pclass"] = Pclass
input_data.at[0, "Age"] = Age
input_data.at[0, "SibSp"] = SibSp
input_data.at[0, "Fare"] = float(Fare)  # Ensure it's float for compatibility

# Encode Sex
if f"Sex_{Sex}" in input_data.columns:
    input_data.at[0, f"Sex_{Sex}"] = 1

# Encode Embarked
if f"Embarked_{Embarked}" in input_data.columns:
    input_data.at[0, f"Embarked_{Embarked}"] = 1

# Encode Title
if f"Title_{Title}" in input_data.columns:
    input_data.at[0, f"Title_{Title}"] = 1

# Scale numerical columns
numerical_cols = ["Pclass", "Age", "SibSp", "Fare"]
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success("Prediction: Survived")  # Green box
    else:
        st.warning("Prediction: Did Not Survive")  # Yellow/Orange box
