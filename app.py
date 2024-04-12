from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf 
import joblib

app = Flask(__name__)

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
def process_image(image_path, model):
    img = Image.open(image_path)
    img = img.resize((32, 32))  # Resize image to match model input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape((1, 32, 32, 3))  # Reshape for model input
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    return predicted_label

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = secure_filename(file.filename)
        file_path = f"uploads/{filename}"
        file.save(file_path)
        model = load_model('my_model.h5')  # Load the model
        predicted_label = process_image(file_path, model)
        predicted_class = class_names[predicted_label]  # Get the class name
        return render_template('upload.html', prediction=predicted_class)


# if __name__ == '__main__':
#     app.run(debug=True)
