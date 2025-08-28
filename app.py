from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import requests
import gdown
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Constants
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.h5')
GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?id=1TleZ9aRZdi7kABt9LMSxeT_O0dsX-Pmj'

def download_model():
    """Download the model from Google Drive if it doesn't exist locally"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    if not os.path.exists(MODEL_PATH):
        print("Downloading model file...")
        gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)
        print("Model downloaded successfully!")

# Global variable for model
model = None

def load_keras_model():
    """Load the Keras model into memory"""
    global model
    download_model()  # Ensure model is downloaded
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if an image file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        try:
            # Load and preprocess the image
            img = Image.open(file.stream).convert('L')  # Convert to grayscale
            img = img.resize((64, 64))  # Resize to match model input
            
            # Convert to array and normalize
            img_array = image.img_to_array(img)
            img_array = img_array.reshape(1, 64, 64, 1)  # Reshape for model input
            
            # Make prediction
            global model
            if model is None:
                load_keras_model()
                
            result = model.predict(img_array)
            result = np.around(result)
            
            # Interpret result
            if result[0][0] == 0:
                prediction = 'benign'
            else:
                prediction = 'malignant'
                
            return jsonify({
                'prediction': prediction,
                'confidence': float(abs(result[0][0] - 0.5) * 2)  # Convert to confidence score
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})

# Load model when app starts
@app.before_first_request
def before_first_request():
    load_keras_model()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
