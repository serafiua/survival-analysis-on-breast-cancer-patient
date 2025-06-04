import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
import seaborn as sns

# Load data
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

cph = CoxPHFitter()
cph.fit(data, duration_col='timerecurrence', event_col='eventdeath')

st.subheader("📊 Kaplan-Meier Survival Curve")
feature_to_plot = st.selectbox("Select variables for survival curve:", ["chemo", "hormonal", "amputation"])

kmf = KaplanMeierFitter()
fig, ax = plt.subplots()
colors = ["#4a6378", "#a42a69"]
for i, group in enumerate(data[feature_to_plot].unique()):
    ix = data[feature_to_plot] == group
    kmf.fit(data.loc[ix, 'timerecurrence'], data.loc[ix, 'eventdeath'], label=f'{feature_to_plot}={group}')
    kmf.plot_survival_function(ax=ax, color=colors[i % len(colors)])

plt.title(f"Survival Prediction for {feature_to_plot} Patient")
plt.xlabel("Time (year)")
plt.ylabel("Survival Probability")
plt.grid(True)
st.pyplot(fig)

st.subheader("📈 Cox Proportional Hazards Model")
cph = CoxPHFitter()
cph.fit(data, duration_col='timerecurrence', event_col='eventdeath')

st.write(cph.summary)


st.sidebar.header("🔍 Predict Survival for a New Patient")

age = st.sidebar.slider("Age", 20, 100, 50)
chemo = st.sidebar.selectbox("Chemotherapy (0=no, 1=yes)", [0, 1])
hormonal = st.sidebar.selectbox("Hormonal therapy (0=no, 1=yes)", [0, 1])
amputation = st.sidebar.selectbox("Amputation (0=no, 1=yes)", [0, 1])
diam = st.sidebar.slider("Tumor diameter (mm)", 0, 100, 10)
posnodes = st.sidebar.slider("Number of positive lymph nodes", 0, 30, 3)
angioinv = st.sidebar.selectbox("Angiolymphatic invasion (0=absent, 1=present)", [0, 1])
lymphinfil = st.sidebar.selectbox("Tumor-Infiltrating Lymphocytes (1=low, 2=intermediate, 3=high)", [1, 2, 3])
histtype = st.sidebar.selectbox("Histological types", sorted(data["histtype"].dropna().unique()))

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
        st.success(f"The predicted hazard ratio for this patient is **{pred_hr:.2f}**.")
        st.markdown(
            ":warning: This indicates an **increased risk of death** compared to the baseline."
        )
    elif pred_hr < 1:
        st.success(f"The predicted hazard ratio for this patient is **{pred_hr:.2f}**.")
        st.markdown(
            ":green_heart: This indicates a **decreased risk of death** compared to the baseline."
        )
    else:
        st.success(f"The predicted hazard ratio for this patient is **{pred_hr:.2f}**.")
        st.markdown("The risk is the **same as the baseline**.")

except:
    st.warning("The model cannot calculate a prediction for this input. Try changing some values.")

surv_func = cph.predict_survival_function(new_patient)
st.markdown("### 🔮 Predicted Survival Function")
fig2, ax2 = plt.subplots(figsize=(10, 6))
surv_func.plot(ax=ax2)
plt.title("Survival Prediction for Inputted Patient")
plt.xlabel("Time (year)")
plt.ylabel("Survival Probability")
st.pyplot(fig2)

st.markdown("---")
st.caption("Model: Cox Proportional Hazards | Data: NKI Breast Cancer | Created by @serafiua on IG")
