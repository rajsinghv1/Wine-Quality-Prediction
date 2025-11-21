import streamlit as st
import joblib 
import numpy as np

# Load the trained model once at startup
@st.cache_resource

    return joblib.load("wine_quality_pred.pkl")


model = load_model()

st.header("Wine Quality Prediction")
st.subheader("Adjust the sliders and predict the wine quality")

volatile_acidity = st.slider("Volatile Acidity", 0.0, 2.0)
citric_acid = st.slider("Citric Acid", 0.0, 0.2)
sulphates = st.slider("Sulphates", 0.0, 2.0)
alcohol = st.slider("Alcohol", 0.0, 15.0)

input_data = np.array([volatile_acidity, citric_acid, sulphates, alcohol]).reshape(1, -1)

if st.button("Predict"):
    result = model.predict(input_data)
    if result[0] == 1:
        st.success("Wine quality is good")
    else:
        st.error("Wine quality is bad")
