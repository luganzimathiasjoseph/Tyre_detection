from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/": {"origins": "*"}})  # Allow all origins

# Define the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a smaller model, MobileNetV2 (for smaller model size)
model = models.mobilenet_v2(pretrained=True)

# Modify the final layer to fit the number of classes (2: good and defective)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)

# Move model to device
model = model.to(device)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-like input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the trained model weights (after training is completed separately)
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'best_model.pth')

# Function to load the trained model
def load_model():
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided'}), 400

        print(f"📸 Received file: {file.filename}")  # Debugging log

        # Convert file to numpy array & OpenCV format
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Convert BGR (OpenCV format) to RGB (PIL format)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        # Apply image transformations
        img = transform(pil_img)
        img = img.unsqueeze(0).to(device)  # Add batch dimension & move to device

        # Run prediction
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)  # Get predicted class

            # Map prediction to class labels
            result = "Good" if predicted.item() == 1 else "Defective"

        return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
