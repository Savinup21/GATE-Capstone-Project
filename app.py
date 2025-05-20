import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="MRI Fracture Detection", layout="centered")

# Constants
IMG_SIZE = (224, 224)
CLASS_LABELS = ['Fracture Detected', 'No Fracture was Detected']
MODEL_PATH = "my_model1.tflite"  # Local TFLite model path

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite(image_array):
    image_array = image_array.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# UI
st.title("MRI Bone Fracture Detection with AI")

uploaded_file = st.file_uploader("Upload MRI Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert('RGB')
    image_resized = pil_image.resize(IMG_SIZE)
    image_array = np.array(image_resized) / 255.0
    image_array_expanded = np.expand_dims(image_array, axis=0)

    output = predict_tflite(image_array_expanded)
    predicted_class_index = int(np.argmax(output))
    confidence = float(np.max(output)) * 100

    st.success(f"**Prediction:** {CLASS_LABELS[predicted_class_index]} ({confidence:.2f}% confidence)")
    st.image(pil_image, caption="Uploaded MRI Image", use_container_width=True)
else:
    st.info("Please upload an MRI image to begin.")
