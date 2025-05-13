import tensorflow as tf
import numpy as np
import streamlit as st
import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)  
CLASS_LABELS = ['Fracture Detected', 'No Fracture was Detected']
import os
import zipfile
import gdown

# Download and unzip model if not already present
model_file = "my_model.keras"
zip_file = "my_model.zip"

if not os.path.exists(model_file):
    # Extract file ID from the link: https://drive.google.com/file/d/12aIPsfp8GlSE7jDNd_RUbuUlKWA8a9ds/view?usp=drive_link
    file_id = "12aIPsfp8GlSE7jDNd_RUbuUlKWA8a9ds"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, zip_file, quiet=False)

    # Unzip the file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()

def get_grad_cam(model, image, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + tf.keras.backend.epsilon())
    return heatmap.numpy()

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if 'conv' in layer.name and len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("No convolutional layer found.")

st.set_page_config(page_title="MRI Fracture Detector with AI", layout="centered")
st.title("MRI Bone Fracture Detection with AI")

model = tf.keras.models.load_model(MODEL_PATH)
TARGET_LAYER_NAME = find_last_conv_layer(model)

uploaded_file = st.file_uploader("Upload MRI Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert('RGB')
    image_resized = pil_image.resize(IMG_SIZE)
    image_array = img_to_array(image_resized) / 255.0
    image_array_expanded = np.expand_dims(image_array, axis=0)

    output = model.predict(image_array_expanded)
    predicted_class_index = np.argmax(output)
    confidence = float(np.max(output)) * 100

    st.success(f"**Prediction:** {CLASS_LABELS[predicted_class_index]}")

    heatmap = get_grad_cam(model, image_array_expanded, TARGET_LAYER_NAME)
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    original_cv = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(original_cv, 0.7, heatmap_colored, 0.3, 0)

    st.image(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB),
             caption='Grad-CAM Highlight', use_container_width=True)
else:
    st.info("Please upload an MRI image to begin.")
