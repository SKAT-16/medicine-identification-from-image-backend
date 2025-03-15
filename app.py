import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/identify/"

st.title("Medicine Identification App")

uploaded_files = st.file_uploader("Upload medicine images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if st.button("Identify Medicine"):
    if uploaded_files:
        files = [("files", (file.name, file, file.type)) for file in uploaded_files]
        response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            st.json(response.json())
        else:
            st.error("Failed to identify medicine.")
