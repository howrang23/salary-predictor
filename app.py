import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("💼 Salary Predictor")

experience = st.selectbox("Your Experience", [0,1,2,3],
    format_func=lambda x: ["Entry Level","Mid Level","Senior","Executive"][x])

company_size = st.selectbox("Company Size", [0,1,2],
    format_func=lambda x: ["Small","Medium","Large"][x])

remote = st.slider("Remote Work %", 0, 100, 50)

job_title = st.number_input("Job Title Code", min_value=0, max_value=50, value=10)

if st.button("Predict My Salary 💰"):
    result = model.predict([[experience, job_title, company_size, remote]])
    st.success(f"Estimated Salary: ${result[0]:,.0f} per year")