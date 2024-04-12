import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib

# Define a function to load the saved model
def load_model(file_path):
    # Load the model based on the file extension
    if file_path.endswith('.h5'):
        model = tf.keras.models.load_model(file_path)
    # elif file_path.endswith('.pb'):
    #     model = tf.saved_model.load(file_path)
    # elif file_path.endswith('.pkl'):
    #     model = joblib.load(file_path)
    # else:
    #     raise ValueError("Unsupported model format")
    return model

# Define a function to process the uploaded image
def process_image(image, model):
    img = image.resize((32, 32))  # Resize image to match model input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape((1, 32, 32, 3))  # Reshape for model input
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    return predicted_label

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def main():
    st.title("Image Classifier")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Predict'):
            model = load_model('my_model.h5')  # Load the model
            predicted_label = process_image(image, model)
            predicted_class = class_names[predicted_label]  # Get the class name
            st.write(f"Predicted class: {predicted_class}")

if __name__ == '__main__':
    main()
