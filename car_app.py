import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import predict

st.title("Car Parts Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = uploaded_file # Image.open(uploaded_file)
else:
    image = "car_dataset_real/images/1534.png" # Image.open("car_dataset_real/images/1534.png")

col1, col2 = st.columns(2)
col1.image(image)

pred = predict.predict_image(image)
for key, val in pred.items():
    if val == 'open':
        color = 'green'
    else:
        color = 'red'
    col2.badge(f'{key}: {val}', color=color)