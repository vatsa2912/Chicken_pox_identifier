import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model("best_model.h5")

# Function to predict the image
def predict_image(model, image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize the image
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    result = "Chickenpox" if prediction[0][0] > 0.5 else "Non-Chickenpox"

    if result == "Chickenpox":
        return result, "myproject_images/yes.png"
    else:
        return result, "myproject_images/no.png"

# Streamlit app starts here
st.set_page_config(page_title="Image Classifier", layout="wide", initial_sidebar_state="collapsed")
st.title("Chickenpox Image Classifier")
st.markdown("Upload an image to classify it as **Chickenpox** or **Non-Chickenpox**.")

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded_file is not None:
    # Display the uploaded image with a fixed width
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)  # Set width to 400 pixels

    # Save the uploaded file temporarily
    image_path = "temp_image.jpg"
    image.save(image_path)

    # Predict the uploaded image
    with st.spinner("Classifying..."):
        prediction, result_image_path = predict_image(model, image_path)

    # Show the prediction result
    st.subheader(f"Prediction: {prediction}")
    
    # Display result image with fixed width
    result_image = Image.open(result_image_path)
    st.image(result_image, caption="Result", width=400)  # Set width to 400 pixels

# Add an About Section
st.sidebar.title("About")
st.sidebar.info(
    """
    This app classifies images as Chickenpox or Non-Chickenpox using a Convolutional Neural Network.
    """
)
