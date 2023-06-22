import streamlit as st
import warnings
import requests
warnings.filterwarnings('ignore')
url = "http://127.0.0.1:8000/predict"

st.header('Duplicate Questions')

q1 = st.text_input('Enter Question 1')
q2 = st.text_input('Enter Question 2')

if st.button('Find'):
    response=requests.post(url,json={"q1":q1,"q2":q2})
    if response.json()['data']["predictions"][0]['predicted_tag'] == "duplicate":
        st.header("Duplicate")
    else:
        st.header("Not duplicate")