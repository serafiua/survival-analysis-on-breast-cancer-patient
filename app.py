import streamlit as st
import pandas as pd
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

st.set_page_config(page_title="Breast Cancer Survival Analysis", layout="wide")
st.title("📊 Cox Proportional Hazards Model - Breast Cancer")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("NKI_cleaned.csv")
    selected_cols = [
        "eventdeath", "timerecurrence", "age", "chemo", "hormonal", "amputation",
        "histtype", "diam", "posnodes", "grade", "angioinv", "lymphinfil"
    ]
    return df[selected_cols]

data = load_data()

# Train model
cph = CoxPHFitter()
cph.fit(data, duration_col='timerecurrence', event_col='eventdeath')

# Show summary
st.subheader("🔍 Model Summary (Hazard Ratios)")
st.dataframe(cph.summary.style.format("{:.3f}"))

# Survival plot for a few example patients
st.subheader("📈 Example Survival Curves")
st.markdown("Berikut ini survival function untuk 5 pasien pertama di dataset.")
fig, ax = plt.subplots(figsize=(10, 6))
cph.predict_survival_function(data.iloc[:5]).plot(ax=ax)
plt.title("Survival Function - 5 Sample Patients")
plt.xlabel("Time (months)")
plt.ylabel("Survival Probability")
st.pyplot(fig)

# Custom patient prediction
st.subheader("🧪 Predict Survival for a New Patient")
with st.form("input_form"):
    age = st.number_input("Age", 20, 100, 50)
    chemo = st.selectbox("Chemotherapy", [0, 1])
    hormonal = st.selectbox("Hormonal Therapy", [0, 1])
    amputation = st.selectbox("Amputation", [0, 1])
    histtype = st.selectbox("Histological Type (1=ductal, 2=lobular, ...) ", [1, 2, 3])
    diam = st.slider("Tumor Diameter (mm)", 0, 100, 25)
    posnodes = st.slider("Positive Lymph Nodes", 0, 20, 3)
    grade = st.selectbox("Tumor Grade", [1, 2, 3])
    angioinv = st.selectbox("Angio Invasion", [0, 1])
    lymphinfil = st.selectbox("Lymphocyte Infiltration", [0, 1])
    submit = st.form_submit_button("Predict")

    if submit:
        new_patient = pd.DataFrame([{
            "age": age,
            "chemo": chemo,
            "hormonal": hormonal,
            "amputation": amputation,
            "histtype": histtype,
            "diam": diam,
            "posnodes": posnodes,
            "grade": grade,
            "angioinv": angioinv,
            "lymphinfil": lymphinfil
        }])

        surv_func = cph.predict_survival_function(new_patient)
        st.markdown("### 🔮 Predicted Survival Function")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        surv_func.plot(ax=ax2)
        plt.title("Survival Prediction for Inputted Patient")
        plt.xlabel("Time (months)")
        plt.ylabel("Survival Probability")
        st.pyplot(fig2)

st.markdown("---")
st.caption("Model: Cox Proportional Hazards | Data: NKI Breast Cancer | Created by @serafiua on IG")
