import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Farm Insect Classifier",
    page_icon="🐛",
    layout="wide"
)

# Title and description
st.title("🐛 Farm Insect Classifier")
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("Upload an image of an insect to identify which species it is.")

# Cache the model loading to avoid reloading
@st.cache_resource
def load_model():
    # Load class names
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    
    # Load trained weights
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, class_names, device

# Load model once
try:
    model, class_names, device = load_model()
    st.success(f"✅ Model loaded successfully! Ready to classify {len(class_names)} insect types.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Make sure 'best_model.pth' and 'class_names.json' exist in the current directory.")
    st.stop()

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)

# Prediction function
def predict(image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()[0]

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Upload Insect Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a clear image of an insect for best results"
    )
    
    # Option to use camera
    camera_image = st.camera_input("Or take a photo with your camera")
    
    # Use camera image if available, otherwise use uploaded file
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", use_column_width=True)
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.info("👈 Upload an image or take a photo to get started!")
        image = None

with col2:
    st.subheader("🔍 Prediction Results")
    
    if image is not None:
        # Show loading spinner while predicting
        with st.spinner("Analyzing image..."):
            # Preprocess and predict
            image_tensor = preprocess_image(image)
            probabilities = predict(image_tensor)
        
        # Get top 5 predictions
        top5_idx = np.argsort(probabilities)[::-1][:5]
        
        # Display main prediction
        top_prediction = class_names[top5_idx[0]]
        top_confidence = probabilities[top5_idx[0]]
        
        st.markdown(f"""
        ### 🎯 Top Prediction:
        ### **{top_prediction}**
        ### Confidence: **{top_confidence:.2%}**
        """)
        
        # Color code confidence
        if top_confidence > 0.8:
            st.success("High confidence prediction ✅")
        elif top_confidence > 0.6:
            st.warning("Medium confidence prediction ⚠️")
        else:
            st.error("Low confidence prediction - try another image ❌")
        
        # Display all predictions as a bar chart
        st.markdown("---")
        st.subheader("📊 All Predictions")
        
        # Prepare data for chart
        top5_names = [class_names[i] for i in top5_idx]
        top5_probs = [probabilities[i] for i in top5_idx]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(top5_names)), top5_probs, color='skyblue')
        ax.set_yticks(range(len(top5_names)))
        ax.set_yticklabels(top5_names)
        ax.set_xlabel('Confidence')
        ax.set_title('Top 5 Predictions')
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, top5_probs)):
            ax.text(prob, bar.get_y() + bar.get_height()/2, 
                   f'{prob:.1%}', ha='left', va='center')
        
        st.pyplot(fig)
        
        # Display as a table
        st.markdown("### Detailed Results")
        results_data = {
            "Insect Type": top5_names,
            "Confidence": [f"{p:.2%}" for p in top5_probs]
        }
        st.table(results_data)
        
    else:
        st.info("Waiting for image...")

# Add sidebar with information
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("""
    This app uses a **ResNet18** deep learning model trained to classify farm insects.
    
    ### Model Performance
    - **Best Validation Accuracy:** 65.68%
    - **Training Epochs:** 10
    - **Model Architecture:** ResNet18
    
    ### Supported Insects
    """)
    
    # Display list of insects
    for i, insect in enumerate(class_names, 1):
        st.markdown(f"{i}. {insect}")
    
    st.markdown("---")
    st.markdown("### 📊 Tips for Best Results")
    st.markdown("""
    - Use clear, well-lit images
    - Center the insect in the frame
    - Avoid blurry images
    - Show the insect from above if possible
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit & PyTorch")

# Add footer
st.markdown("---")
st.markdown("*Upload an image to classify farm insects. For best results, use clear, well-lit images.*")