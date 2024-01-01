import pickle
import streamlit as st


model = pickle.load(open('heart_disease.sav', 'rb'))

st.title('Prediksi Penyakit Jantung')

col1, col2, col3 = st.columns(3)

with col1:
    age = st.text_input('Input usia')
with col1:
    sex = st.text_input('Input jenis kelamin')
with col1:
    cp = st.text_input('Input jenis nyeri dada')
with col1:
    trestbps = st.text_input('Input tekanan darah')
with col1:
    chol = st.text_input('Input kadar kolesterol')
with col2:
    fbs = st.text_input('Input gula darah')
with col2:
    restecg	 = st.text_input('Input hasil elektrokardiogram')
with col2:
    thalach = st.text_input('Input Denyut jantung maksimum yang dicapai')
with col2:
    exang = st.text_input('Input Angina akibat olahraga')
with col2:
    oldpeak = st.text_input('Input oldpeak')
with col3:
    slope = st.text_input('Input Kemiringan')
with col3:
    ca = st.text_input('Input Jumlah pembuluh darah utama')
with col3:
    thal = st.text_input('Input tipe thalassemia')


predict = ''

if st.button('Prediksi Mempunyai penyakit jantung'):
    predict = model.predict(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    )
    st.write('Prediksi : ', predict)