import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="X Ray Fracture Detection", layout="centered")

# Constants
IMG_SIZE = (224, 224)
CLASS_LABELS = ['No Fracture Detected', 'Fracture was Detected']
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
st.title("X Ray Bone Fracture Detection with AI")

uploaded_file = st.file_uploader("Upload X ray Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert('RGB')
    image_resized = pil_image.resize(IMG_SIZE)
    image_array = np.array(image_resized) / 255.0
    image_array_expanded = np.expand_dims(image_array, axis=0)

    output = predict_tflite(image_array_expanded)
    predicted_class_index = int(np.argmax(output))

    st.success(f"**Prediction:** {CLASS_LABELS[predicted_class_index]}")
    st.image(pil_image, caption="Uploaded X ray Image", use_container_width=True)
else:
    st.info("Please upload an X ray image to begin.")


st.markdown(
    """
    <hr style="margin-top: 2em;">
    <p style="font-size: 0.9em; color: gray; text-align: center;">
        ⚠️ This tool is for educational purposes only. Predictions may be inaccurate and should not be used for medical diagnosis.























Made by Savinu Perera
    </p>
    """,
    unsafe_allow_html=True
)

