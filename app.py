import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
import seaborn as sns

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("NKI_cleaned.csv")

data = load_data()

st.title("🩺 Survival Analysis untuk Pasien Breast Cancer")
st.markdown("""
Aplikasi ini menggunakan Kaplan-Meier dan Cox Proportional Hazards untuk menganalisis waktu survival pasien kanker payudara berdasarkan data medis.
""")

# Sidebar untuk input user
st.sidebar.header("🔍 Masukkan Karakteristik Pasien")

age = st.sidebar.slider("Umur Pasien", 20, 100, 50)
chemo = st.sidebar.selectbox("Terapi Kemoterapi", [0, 1])
hormonal = st.sidebar.selectbox("Terapi Hormonal", [0, 1])
amputation = st.sidebar.selectbox("Amputasi", [0, 1])
diam = st.sidebar.slider("Diameter Tumor (mm)", 0, 100, 25)
posnodes = st.sidebar.slider("Jumlah Node Positif", 0, 30, 3)
grade = st.sidebar.selectbox("Grade Tumor", sorted(data["grade"].dropna().unique()))
angioinv = st.sidebar.selectbox("Invasivitas Angio", [0, 1])
lymphinfil = st.sidebar.selectbox("Infiltrasi Limfosit", [0, 1])
histtype = st.sidebar.selectbox("Histologi", sorted(data["histtype"].dropna().unique()))

# Tampilkan survival curve berdasarkan fitur tertentu
st.subheader("📊 Kaplan-Meier Survival Curve")
feature_to_plot = st.selectbox("Pilih variabel untuk survival curve:", ["chemo", "hormonal", "amputation"])

kmf = KaplanMeierFitter()
fig, ax = plt.subplots()
colors = ["#4a6378", "#a42a69"]
for i, group in enumerate(data[feature_to_plot].unique()):
    ix = data[feature_to_plot] == group
    kmf.fit(data.loc[ix, 'timerecurrence'], data.loc[ix, 'eventdeath'], label=f'{feature_to_plot}={group}')
    kmf.plot_survival_function(ax=ax, color=colors[i % len(colors)])

plt.title(f"Kaplan-Meier Curve berdasarkan {feature_to_plot}")
plt.xlabel("Waktu (bulan)")
plt.ylabel("Probabilitas Survival")
plt.grid(True)
st.pyplot(fig)

# CoxPHFitter
st.subheader("📈 Cox Proportional Hazards Model")
cph = CoxPHFitter()
cph.fit(data, duration_col='timerecurrence', event_col='eventdeath')

st.write("Ringkasan model:")
st.write(cph.summary)

# Tambahan: prediksi hazard ratio untuk input user
input_df = pd.DataFrame([{
    "age": age,
    "chemo": chemo,
    "hormonal": hormonal,
    "amputation": amputation,
    "diam": diam,
    "posnodes": posnodes,
    "grade": grade,
    "angioinv": angioinv,
    "lymphinfil": lymphinfil,
    "histtype": histtype,
}])

st.subheader("🧮 Prediksi Hazard Ratio untuk Pasien Ini")
try:
    pred_hr = np.exp(cph.predict_log_partial_hazard(input_df))[0]
    st.success(f"Hazard Ratio pasien ini diprediksi sebesar **{pred_hr:.2f}**.")
except:
    st.warning("Model tidak bisa menghitung prediksi untuk input ini. Coba ubah beberapa nilai.")

st.caption("Data berdasarkan dataset NKI_cleaned.csv")

st.markdown("---")
st.caption("Model: Cox Proportional Hazards | Data: NKI Breast Cancer | Created by @serafiua on IG")
