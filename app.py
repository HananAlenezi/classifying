import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import requests
import io
import os

# === Model Setup (No Hardcoded Paths!) ===
MODEL_FILES = {
    "EfficientNet-B0": "efficientnet_b0_retrained.pth",
    "MobileNetV2": "mobilenet_v2_retrained.pth",
    "ResNet50": "resnet50_retrained.pth",
    "AlexNet": "alexnet_retrained.pth"
}

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy","background",
]

# === Image Transformation ===
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Fixed Model Loading ===
def load_model(model_name):
    """Load model from current directory"""
    try:
        # Initialize model architecture
        if model_name == "EfficientNet-B0":
            model = torch.hub.load('pytorch/vision', 'efficientnet_b0', pretrained=False)
            model.classifier[1] = torch.nn.Linear(1280, len(CLASS_NAMES))
        elif model_name == "MobileNetV2":
            model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=False)
            model.classifier[1] = torch.nn.Linear(1280, len(CLASS_NAMES))
        elif model_name == "ResNet50":
            model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=False)
            model.fc = torch.nn.Linear(2048, len(CLASS_NAMES))
        elif model_name == "AlexNet":
            model = torch.hub.load('pytorch/vision', 'alexnet', pretrained=False)
            model.classifier[6] = torch.nn.Linear(4096, len(CLASS_NAMES))
        
        # Load weights
        if os.path.exists(MODEL_FILES[model_name]):
            model.load_state_dict(torch.load(MODEL_FILES[model_name], map_location='cpu'))
        else:
            st.error(f"Model file missing! Please upload {MODEL_FILES[model_name]}")
            return None
            
        return model.eval()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# === Streamlit UI ===
st.title("🌿 Plant Doctor Pro")
st.write("Upload, paste a URL, or use camera to diagnose plant diseases")

# === Input Options ===
tab1, tab2, tab3 = st.tabs(["📷 Camera", "📂 Upload", "🌐 URL"])

with tab1:
    picture = st.camera_input("Take a leaf photo")
    if picture:
        img = Image.open(picture)
        if st.button("🔄 Flip Camera Image"):
            img = ImageOps.mirror(img)
        st.image(img, caption="Your Leaf", width=300)

with tab2:
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=300)

with tab3:
    url = st.text_input("Paste image URL")
    if url:
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(io.BytesIO(response.content))
            st.image(img, caption="URL Image", width=300)
        except:
            st.error("Invalid URL or couldn't load image")

# === Classification ===
if 'img' in locals():
    model_name = st.selectbox("Select Model", list(MODEL_FILES.keys()))
    
    if st.button("🔍 Diagnose"):
        with st.spinner("Analyzing..."):
            model = load_model(model_name)
            if model:
                img_tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0)
                with torch.no_grad():
                    output = model(img_tensor)
                    prediction = CLASS_NAMES[torch.argmax(output).item()]
                
                st.success(f"**Diagnosis:** {prediction}")
                
