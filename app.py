import gradio as gr
import torch
from torchvision import transforms, models
import json
from PIL import Image
import torch.nn as nn

# Load model and classes
with open("class_names.json", "r") as f:
    class_names = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify_image(img):
    img = Image.fromarray(img).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    
    return {class_names[i]: float(probs[0][i]) for i in range(len(class_names))}

iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="Farm Insect Classifier",
    description="Upload an insect image to classify it"
)

iface.launch(share=True)  # Creates a public URL