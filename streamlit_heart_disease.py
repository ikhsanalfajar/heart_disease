import pickle
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier


model = pickle.load(open('heart_disease.sav', 'rb'))

st.title('Prediksi Penyakit Jantung')

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Input usia')
with col1:
    sex = st.number_input('Input jenis kelamin')
with col1:
    cp = st.number_input('Input jenis nyeri dada')
with col1:
    trestbps = st.number_input('Input tekanan darah')
with col1:
    chol = st.number_input('Input kadar kolesterol')
with col2:
    fbs = st.number_input('Input gula darah')
with col2:
    restecg	 = st.number_input('Input hasil elektrokardiogram')
with col2:
    thalach = st.number_input('Input Denyut jantung maksimum yang dicapai')
with col2:
    exang = st.number_input('Input Angina akibat olahraga')
with col2:
    oldpeak = st.number_input('Input oldpeak')
with col3:
    slope = st.number_input('Input Kemiringan')
with col3:
    ca = st.number_input('Input Jumlah pembuluh darah utama')
with col3:
    thal = st.number_input('Input tipe thalassemia')


predict = ''

if st.button('Prediksi Mempunyai penyakit jantung'):
    predict = model.predict(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    )

    if (predict == 1):
            st.warning("Orang tersebut aman dari penyakit jantung")
    else:
            st.success("Orang tersebut rentan terkena penyakit jantung")
    st.write('Prediksi : ', predict)