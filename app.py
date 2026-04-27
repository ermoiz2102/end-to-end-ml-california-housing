import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load("rf_pipeline_model.pkl")

st.title("🏠 House Price Prediction App")

st.write("Enter housing details to predict price")

# Input fields
MedInc = st.number_input("Median Income", value=5.0)
HouseAge = st.number_input("House Age", value=20.0)
AveRooms = st.number_input("Average Rooms", value=5.0)
AveBedrms = st.number_input("Average Bedrooms", value=1.0)
Population = st.number_input("Population", value=1000.0)
AveOccup = st.number_input("Average Occupancy", value=3.0)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms,
                                Population, AveOccup, Latitude, Longitude]],
                              columns=["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                                       "Population", "AveOccup", "Latitude", "Longitude"])

    prediction = model.predict(input_data)

    st.success(f"🏡 Predicted House Price: ${prediction[0]*100000:.2f}")