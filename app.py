import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Device setup (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# Load the MobileNetV2 model
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)  # Binary classification
    model = model.to(device)
    model.eval()
    
    # Load model weights
    model_path = "best_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
    
    return model

model = load_model()

# Image preprocessing function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("Tyre Defect Detection")
st.write("Upload an image of a tyre")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Convert image to tensor
    img = transform(image).unsqueeze(0).to(device)
    
    if st.button("Predict"):
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            result = "Good" if predicted.item() == 1 else "Defective"
            st.success(f"Prediction: {result}")
