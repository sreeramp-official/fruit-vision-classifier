import streamlit as st
import numpy as np
import joblib
from PIL import Image
import os

# Load the trained model
model_path = '../notebooks/fruit_classifier_model.pkl'  # Path to the model
if os.path.exists(model_path):  # Check if the model exists
    model = joblib.load(model_path)  # Load the saved model
else:
    st.error("The model file 'fruit_classifier_model.pkl' is not found. Please ensure the model is trained and saved.")
    # Show an error message if the model file is not found

# Function to preprocess the uploaded image
def preprocess_image(image):  # Prepare the uploaded image for the model
    img = Image.open(image)  # Open the image file
    img = img.resize((128, 128))  # Resize to 128x128 pixels
    img_array = np.array(img)  # Convert image to a format the model can read
    
    # Calculate the mean RGB values (same as during training)
    mean_rgb = np.mean(img_array, axis=(0, 1)).tolist()  # Get average color (R, G, B)
    return np.array(mean_rgb)  # Return the color information for prediction

# Streamlit app title and styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f7ff;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    h1 {
        font-family: 'Arial', sans-serif;
        font-size: 40px;
        color: #9381ff;
    }
    .stFileUploader {
        background-color: #b8b8ff;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton {
        background-color: #9381ff;
        color: white;
        border-radius: 5px;
        font-size: 16px;
        width: 100%;
    }
    .stImage {
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .result {
        background-color: #b8b8ff;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        font-size: 20px;
        font-weight: bold;
        color: #f8f7ff;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1>Fruit Classifier</h1>", unsafe_allow_html=True)

# Option to upload an image for prediction with user-friendly message
st.markdown("<p style='font-size: 18px;'>Upload a picture of a fruit, and I will tell you what it is!</p>", unsafe_allow_html=True)
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# If an image is uploaded, process and classify it
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)  # Display the uploaded image
    
    # Preprocess the uploaded image
    image_features = preprocess_image(uploaded_image)
    
    # Reshape the features to match the input shape expected by the model
    image_features = image_features.reshape(1, -1)
    
    # Predict the class using the trained model
    predicted_class = model.predict(image_features)[0]  # Get the predicted fruit class
    
    # Display the predicted fruit name
    st.markdown(f"<div class='result'>The fruit is: {predicted_class.capitalize()}</div>", unsafe_allow_html=True)
else:
    st.markdown("<p style='font-size: 16px; color: #9381ff;'>Please upload a fruit image to get started.</p>", unsafe_allow_html=True)

