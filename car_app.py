import streamlit as st
import predict

st.title("Car Parts Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = uploaded_file
else:
    image = "example/6857.png"

col1, col2 = st.columns(2)
col1.image(image)

pred = predict.predict_image(image)
for key, val in pred.items():
    if val == 'open':
        color = 'green'
    else:
        color = 'red'
    col2.badge(f'{key}: {val}', color=color)