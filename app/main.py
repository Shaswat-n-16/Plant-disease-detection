import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Set up the working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction.h5")

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class indices from the JSON file
with open(os.path.join(working_dir, "class_indices.json")) as f:
    class_indices = json.load(f)

# Function to load and preprocess the image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)  # Resize the image
        img_array = np.array(img)  # Convert the image to a numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array.astype('float32') / 255.  # Scale the image values to [0, 1]
        return img_array
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Function to predict the class of an image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    if preprocessed_img is not None:
        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown Class")
        return predicted_class_name
    return "Error in prediction"

# Streamlit App
st.set_page_config(page_title='Plant Disease Classifier', layout='wide')

st.title('🌿 Plant Disease Classifier')
st.write("""
    Upload a plant leaf image, and this application will classify it based on the disease detected.
    This tool is designed to assist farmers and plant enthusiasts in identifying plant health issues.
""")

# Sidebar for user information
st.sidebar.header('Instructions')
st.sidebar.write("""
    1. Click on the "Upload an image..." button to upload a leaf image.
    2. Click on the "Classify" button to get predictions.
    3. Ensure the image is clear and well-lit for better results.
""")

# File uploader
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Uploaded Image')
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.subheader('Prediction')
            st.success(f'Predicted Disease: {str(prediction)}')


