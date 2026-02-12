import pickle
import streamlit as st

diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

st.title('Diabetes Prediction Model')

coll1, col2 = st.columns(2)
with coll1:
    pregnancies = st.text_input('Number of Pregnancies')
with col2:
    glucose = st.text_input('Glucose Level')
with coll1:
    blood_pressure = st.text_input('Blood Pressure value')
with col2:
    skin_thickness = st.text_input('Skin Thickness value')
with coll1:
    insulin = st.text_input('Insulin Level')
with col2:    
    bmi = st.text_input('BMI value')
with coll1:
    diabetes_pedigree_function = st.text_input('Diabetes Pedigree Function value')
with col2:
    age = st.text_input('Age of the Person')
diagnosis = ''
if st.button('Diabetes Test Result'):
    diagnosis = diabetes_model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    if (diagnosis[0] == 1):
        st.success('The person is diabetic')
    else:
        st.success('The person is not diabetic')

    st.success(diagnosis)