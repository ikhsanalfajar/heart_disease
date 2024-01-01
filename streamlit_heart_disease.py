import pickle
import streamlit as st


model = pickle.load(open('heart_disease.sav', 'rb'))

st.title('Prediksi Penyakit Jantung')

age = st.text_input('Input usia')
sex = st.text_input('Input jenis kelamin')
cp = st.text_input('Input jenis nyeri dada')
trestbps = st.text_input('Input tekanan darah')
chol = st.text_input('Input kadar kolesterol')
fbs = st.text_input('Input gula darah')
restecg	 = st.text_input('Input hasil elektrokardiogram')
thalach = st.text_input('Input Denyut jantung maksimum yang dicapai')
exang = st.text_input('Input Angina akibat olahraga')
oldpeak = st.text_input('Input oldpeak')
slope = st.text_input('Input Kemiringan')
ca = st.text_input('Input Jumlah pembuluh darah utama')
thal = st.text_input('Input tipe thalassemia')


predict = ''

if st.button('Prediksi Mempunyai penyakit jantung'):
    predict = model.predict(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    )
    st.write('Prediksi : ', predict)