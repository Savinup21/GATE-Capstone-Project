import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

st.set_page_config(page_title="X Ray Fracture Detection", layout="centered")

IMG_SIZE = (224, 224)
CLASS_LABELS = ['No Fracture Detected', 'Fracture was Detected']

MODEL_PATH = "model_weights.h5"
MODEL_URL = "https://github.com/Savinup21/GATE-Capstone-Project/releases/download/Model/model_weights.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

@st.cache_resource
def load_model():
    download_model()
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu', name="target_conv_layer"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.load_weights(MODEL_PATH)
    return model

model = load_model()

st.title("X-Ray Bone Fracture Detection with AI")

uploaded_file = st.file_uploader("Upload an X-ray Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    pil_image = Image.open(uploaded_file).convert('RGB')
    image_resized = pil_image.resize(IMG_SIZE)
    image_array = np.array(image_resized) / 255.0
    image_array_expanded = np.expand_dims(image_array, axis=0)
    output = model.predict(image_array_expanded)
    predicted_class_index = int(np.argmax(output))

    st.success(f"**Prediction:** {CLASS_LABELS[predicted_class_index]}")
    st.image(pil_image, caption="Uploaded X-ray Image", use_container_width=True)

else:
    st.info("Please upload an X-ray image to begin.")

st.markdown(
    """
    <hr style="margin-top: 2em;">
    <p style="font-size: 0.9em; color: gray; text-align: center;">
        ⚠️ This tool is for educational purposes only. Predictions may be inaccurate and should not be used for medical diagnosis.<br><br>
        Made by Savinu Perera
    </p>
    """,
    unsafe_allow_html=True
)
