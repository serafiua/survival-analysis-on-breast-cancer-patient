# 🩺 Survival Analysis on Breast Cancer Patients

A Streamlit-based application for survival analysis of breast cancer patients using **Kaplan-Meier** and **Cox Proportional Hazards (CoxPH)** models.  
This project provides an interactive way to explore the NKI Breast Cancer dataset and predict patient-specific hazard ratios and survival functions.  

⚠️ **Disclaimer**:  
This project is intended for **educational and exploratory purposes only**.  
It is not designed for clinical decision-making.

---

## 📂 Dataset
- **Source**: NKI Breast Cancer dataset  
- **Columns used**:
  - `eventdeath` (event occurrence: death indicator)  
  - `timerecurrence` (time until event or censoring)  
  - `age`, `chemo`, `hormonal`, `amputation`, `histtype`, `diam`, `posnodes`, `angioinv`, `lymphinfil`  

---

## 🚀 Features
- 📊 Load and explore survival dataset (NKI Breast Cancer).
- 📈 Kaplan-Meier and CoxPH survival analysis.
- 🧮 Predict **hazard ratio** based on patient input data.
- 🔮 Visualize **predicted survival functions**.
- 🎛️ Interactive input with sliders and dropdowns.

---

## 📦 Tech Stack
- [Streamlit](https://streamlit.io/) – Interactive web app framework  
- [Pandas](https://pandas.pydata.org/) – Data handling  
- [NumPy](https://numpy.org/) – Numerical computations  
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) – Visualization  
- [Lifelines](https://lifelines.readthedocs.io/) – Survival analysis  

---
