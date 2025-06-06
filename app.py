import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
import seaborn as sns

@st.cache_data
def load_data():
    df = pd.read_csv("NKI_cleaned.csv")
    selected_cols = [
        "eventdeath", "timerecurrence", "age", "chemo", "hormonal", "amputation",
        "histtype", "diam", "posnodes", "angioinv", "lymphinfil"
    ]
    return df[selected_cols]

data = load_data()

st.title("🩺 Survival Analysis on Breast Cancer Patient")
st.markdown("""
This project uses Kaplan-Meier and Cox Proportional Hazards to analyze the survival time of breast cancer patients based on medical data.
""")
st.warning("""
**⚠️ Disclaimer:**  
The results from this analysis are based on a specific dataset (NKI Breast Cancer) and may not generalize to other patient populations.
Use these predictions for educational or exploratory purposes only — not for clinical decision-making.
""")

# CoxPH model fit
cph = CoxPHFitter()
cph.fit(data, duration_col='timerecurrence', event_col='eventdeath')

# --- Input section (main page) ---
st.subheader("🧪 Input Patient Data")

yes_no_map = {"No": 0, "Yes": 1}
angioinv_map = {"Absent": 0, "Present": 1}
lymphinfil_map = {"Low": 1, "Intermediate": 2, "High": 3}

with st.form("prediction_form"):
    age = st.slider("Age", 20, 100, 50)
    chemo = yes_no_map[st.selectbox("Chemotherapy", list(yes_no_map.keys()))]
    hormonal = yes_no_map[st.selectbox("Hormonal therapy", list(yes_no_map.keys()))]
    amputation = yes_no_map[st.selectbox("Amputation", list(yes_no_map.keys()))]
    diam = st.slider("Tumor diameter (mm)", 0, 100, 10)
    posnodes = st.slider("Number of positive lymph nodes", 0, 30, 3)
    angioinv = angioinv_map[st.selectbox("Angiolymphatic invasion", list(angioinv_map.keys()))]
    lymphinfil = lymphinfil_map[st.selectbox("Tumor-Infiltrating Lymphocytes", list(lymphinfil_map.keys()))]
    histtype = st.selectbox("Histological types", sorted(data["histtype"].dropna().unique()))

    submitted = st.form_submit_button("Predict Hazard Ratio")

if submitted:
    new_patient = pd.DataFrame([{
        "age": age,
        "chemo": chemo,
        "hormonal": hormonal,
        "amputation": amputation,
        "diam": diam,
        "posnodes": posnodes,
        "angioinv": angioinv,
        "lymphinfil": lymphinfil,
        "histtype": histtype,
    }])

    st.subheader("🧮 Predicted Hazard Ratio for Inputted Patient")
    try:
        pred_hr = np.exp(cph.predict_log_partial_hazard(new_patient))[0]

        if pred_hr > 1:
            st.success(f"The predicted hazard ratio is **{pred_hr:.2f}**")
            st.markdown(":warning: This indicates an **increased risk of death** compared to the baseline.")
        elif pred_hr < 1:
            st.success(f"The predicted hazard ratio is **{pred_hr:.2f}**")
            st.markdown(":green_heart: This indicates a **decreased risk of death** compared to the baseline.")
        else:
            st.success(f"The predicted hazard ratio is **{pred_hr:.2f}**")
            st.markdown("The risk is **same as the baseline**.")

        st.markdown("### 🔮 Predicted Survival Function")

        st.markdown("""
        **ℹ️ Explanation:**
        - The graph above represents the **patient's probability of survival** over time (in years).
        - **X-axis** = time since diagnosis (years).
        - **Y-axis** = survival probability (ranges from 0 to 1).
        - A **steep drop** in the curve indicates a higher risk of death early on.
        - A **flat or slowly declining curve** suggests a better prognosis and longer survival.
        """)


        surv_func = cph.predict_survival_function(new_patient)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        surv_func.plot(ax=ax2)
        plt.title("Survival Prediction for Inputted Patient")
        plt.xlabel("Time (year)")
        plt.ylabel("Survival Probability")
        st.pyplot(fig2)

    except Exception as e:
        st.warning(f"⚠️ Cannot calculate prediction: {e}")

st.markdown("---")
st.caption("Model: Cox Proportional Hazards | Data: NKI Breast Cancer | Created by @serafiua")
