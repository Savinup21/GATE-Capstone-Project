import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="X-Ray Fracture Detection", layout="centered")

IMG_SIZE = (224, 224)
CLASS_LABELS = ['No Fracture Detected', 'Fracture was Detected']
MODEL_PATH = "my_model1.tflite"

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image):
    image_resized = image.resize(IMG_SIZE)
    image_array = np.array(image_resized).astype(np.float32)

    image_array = (image_array / 127.5) - 1.0 
    return np.expand_dims(image_array, axis=0) 

def predict_with_tflite(interpreter, image_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    expected_dtype = input_details[0]['dtype']

    image_array = image_array.astype(expected_dtype)

    interpreter.set_tensor(input_index, image_array)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output

interpreter = load_tflite_model()

st.title("X-Ray Bone Fracture Detection with AI")

uploaded_file = st.file_uploader("Upload an X-ray Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    pil_image = Image.open(uploaded_file).convert('RGB')
    image_input = preprocess_image(pil_image)

    # Make prediction
    output = predict_with_tflite(interpreter, image_input)
    predicted_class_index = int(np.argmax(output))
    confidence = output[0][predicted_class_index] * 100

    st.image(pil_image, caption="📷 Uploaded X-ray Image", use_container_width=True)
    st.success(f"Prediction: **{CLASS_LABELS[predicted_class_index]}** ({confidence:.2f}%)")
else:
    st.info("⬆️ Please upload an X-ray image to begin.")

st.markdown(
    """
    <hr style="margin-top: 2em;">
    <p style="font-size: 0.9em; color: gray; text-align: center;">
        ⚠This tool is for educational purposes only. Predictions may be inaccurate and should not be used for medical diagnosis.<br><br>
        Made by Savinu Perera
    </p>
    """,
    unsafe_allow_html=True
)
