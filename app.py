import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Screening", layout="wide")

MODEL_PATH = "diabetes_pipeline.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def risk_level(p: float) -> str:
    if p < 0.30:
        return "Rendah"
    if p < 0.60:
        return "Sedang"
    return "Tinggi"

st.title("Diabetes Risk Screening (Skrining — Bukan Diagnosis)")

st.info(
    "Aplikasi ini memberikan *risk score* (probabilitas) berdasarkan model data. "
    "Untuk diagnosis diabetes, tetap diperlukan pemeriksaan laboratorium (WHO/ADA)."
)

try:
    pipe = load_model()
except FileNotFoundError:
    st.error(f"File model '{MODEL_PATH}' tidak ditemukan. Jalankan training dulu (train_diabetes.py).")
    st.stop()
except Exception as e:
    st.error("Gagal load model pipeline. Pastikan file .pkl berasal dari training terbaru.")
    st.exception(e)
    st.stop()

# =========================
# INPUTS (SAFE)
# =========================
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

# Peringatan untuk nilai 0 yang biasanya missing di dataset Pima
zero_like_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
zero_entered = [c for c in zero_like_cols if X.loc[0, c] == 0]
if zero_entered:
    st.warning(
        "Catatan: nilai 0 pada kolom ini sering dianggap data hilang pada dataset Pima dan akan diimputasi median oleh pipeline: "
        + ", ".join(zero_entered)
    )

# =========================
# PREDICT
# =========================
if st.button("Hitung Risk Score", type="primary"):
    try:
        proba = float(pipe.predict_proba(X)[:, 1][0])
        pred = int(pipe.predict(X)[0])
        lvl = risk_level(proba)

        st.subheader("Hasil Model")
        st.metric("Risk score (Probabilitas Outcome=1)", f"{proba*100:.1f}%")
        st.write(f"Kategori risiko: **{lvl}**")
        st.write("Prediksi kelas (threshold default 0.5):", "**Diabetes**" if pred == 1 else "**Tidak diabetes**")

        st.subheader("Anjuran Konfirmasi Laboratorium (WHO/ADA)")
        st.markdown(
            "- **Gula darah puasa (FPG)**: Diabetes bila **≥ 126 mg/dL** (puasa ≥ 8 jam)\n"
            "- **OGTT 2 jam (75 g)**: Diabetes bila **≥ 200 mg/dL**\n"
            "- **HbA1c**: Diabetes bila **≥ 6.5%**\n"
            "- **Gula darah sewaktu**: **≥ 200 mg/dL** + gejala klasik → kuat mengarah diabetes\n"
        )
        st.caption("Jika risk score sedang/tinggi, lakukan tes lab di atas dan konsultasi tenaga kesehatan.")

    except Exception as e:
        st.error("Gagal melakukan prediksi. Pastikan versi library konsisten dengan requirements.")
        st.exception(e)