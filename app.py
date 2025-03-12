from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})

# Device setup (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the MobileNetV2 model
def load_model():
    global model
    model = models.mobilenet_v2(pretrained=True)   # Load MobileNetV2
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)  # Binary classification (Good vs. Defective)
    model = model.to(device)
    model.eval()

    # Load model weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'best_model.pth')

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Call load_model() when the app starts
load_model()

# Image preprocessing function (must match training settings)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Ensure this matches training
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    print(f"Received file: {file.filename}")

    # Convert file to OpenCV image
    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    # Apply transformations
    img = transform(pil_img).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        result = "Good" if predicted.item() == 1 else "Defective"

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
