import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Screening", layout="wide")

@st.cache_resource
def load_pipeline():
    # Prefer pipeline that includes preprocessing + model
    try:
        with open("diabetes_pipeline.pkl", "rb") as f:
            return pickle.load(f), "pipeline"
    except FileNotFoundError:
        # Fallback to old model (no preprocessing, no probabilities)
        with open("diabetes_model.sav", "rb") as f:
            return pickle.load(f), "legacy"

pipe, mode = load_pipeline()

st.title("Diabetes Risk Screening (Riset/Skrining — Bukan Diagnosis)")

st.info(
    "Aplikasi ini memberikan *risk score* (probabilitas) berbasis model data. "
    "Untuk kepastian diagnosis, tetap perlu pemeriksaan laboratorium (FPG, HbA1c, OGTT)."
)

def risk_level(p: float) -> str:
    if p < 0.30:
        return "Rendah"
    if p < 0.60:
        return "Sedang"
    return "Tinggi"

# Input UI (lebih aman: number_input)
c1, c2 = st.columns(2)
with c1:
    pregnancies = st.number_input("Pregnancies (jumlah kehamilan)", min_value=0, max_value=20, value=0, step=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=400.0, value=120.0, step=1.0)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=99.0, value=20.0, step=1.0)
    bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=80.0, value=27.0, step=0.1)
with c2:
    blood_pressure = st.number_input("Blood Pressure (diastolic, mmHg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
    insulin = st.number_input("Insulin (µU/mL)", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.40, step=0.01, format="%.2f")
    age = st.number_input("Age (tahun)", min_value=1, max_value=120, value=35, step=1)

# Prepare row
row = {
    "Pregnancies": int(pregnancies),
    "Glucose": float(glucose),
    "BloodPressure": float(blood_pressure),
    "SkinThickness": float(skin_thickness),
    "Insulin": float(insulin),
    "BMI": float(bmi),
    "DiabetesPedigreeFunction": float(dpf),
    "Age": int(age),
}
X = pd.DataFrame([row])

# Warnings for likely-missing values
missing_like = []
for k in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
    if X.loc[0, k] == 0:
        missing_like.append(k)

if missing_like:
    st.warning(
        "Nilai 0 pada kolom berikut umumnya dianggap data hilang pada dataset Pima: "
        + ", ".join(missing_like)
        + ". Jika ini bukan data hilang, pastikan memang benar nilainya 0."
    )

if st.button("Hitung Risk Score", type="primary"):
    if mode == "pipeline":
        proba = float(pipe.predict_proba(X)[:, 1][0])
        pred = int(pipe.predict(X)[0])
        lvl = risk_level(proba)

        st.subheader("Hasil Model")
        st.metric("Risk score (Probabilitas Outcome=1)", f"{proba*100:.1f}%")
        st.write(f"Kategori risiko: **{lvl}**")
        st.write("Prediksi kelas (threshold default 0.5):", "**Diabetes**" if pred == 1 else "**Tidak diabetes**")

    else:
        # Legacy: no preprocessing, no probability
        try:
            pred = int(pipe.predict(X.values)[0])
            st.subheader("Hasil Model (Legacy)")
            st.write("Model lama tidak mendukung probabilitas. Output hanya kelas prediksi.")
            st.write("Prediksi:", "**Diabetes**" if pred == 1 else "**Tidak diabetes**")
        except Exception as e:
            st.error("Model legacy gagal memproses input. Gunakan diabetes_pipeline.pkl (pipeline lengkap).")
            st.exception(e)

    st.subheader("Anjuran Konfirmasi Laboratorium (Standar)")
    st.markdown(
        """- **Gula darah puasa (FPG)**: Diabetes bila **≥ 126 mg/dL** (puasa ≥ 8 jam)  
- **OGTT 2 jam (75 g)**: Diabetes bila **≥ 200 mg/dL**  
- **HbA1c**: Diabetes bila **≥ 6.5%**  
- **Gula darah sewaktu**: **≥ 200 mg/dL** disertai gejala klasik → kuat mengarah diabetes  
"""
    )
    st.caption("Jika hasil model sedang/tinggi, sebaiknya lakukan tes lab di atas dan konsultasi tenaga kesehatan.")
