import pickle
import streamlit as st

model = pickle.load(open('estimasi_Mobile_Price_rediction.sav', 'rb'))

st.title('Mobile Price Prediksi')

age = st.number_input('Input usia')
sex = st.number_input('Input jenis kelamin')
cp = st.number_input('Input jenis nyeri dada')
trestbps	 = st.number_input('Input tekanan darah')
chol = st.number_input('Input kadar kolesterol')
fbs = st.number_input('Input gula darah')
restecg	 = st.number_input('Input hasil elektrokardiogram')
thalach = st.number_input('Input Denyut jantung maksimum yang dicapai')
exang = st.number_input('Input Angina akibat olahraga')
oldpeak = st.number_input('Input oldpeak')
slope = st.number_input('Input Kemiringan')
ca = st.number_input('Input Jumlah pembuluh darah utama')
thal = st.number_input('Input tipe thalassemia')


predict = ''

if st.button('Estimasi Mobile'):
    predict = model.predict(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    )
    st.write('Estimasi : ', predict)
