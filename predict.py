import torch
import json
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
import sys

def predict_image(image_path, model_path="best_model.pth", class_names_path="class_names.json"):
    # Load class names
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load and transform image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    print(f"Predicted: {class_names[predicted.item()]}")
    print(f"Confidence: {confidence.item():.2%}")
    
    # Show top 3 predictions
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    print("\nTop 3 predictions:")
    for i in range(3):
        print(f"  {class_names[top3_idx[0][i]]}: {top3_prob[0][i].item():.2%}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_image(sys.argv[1])
    else:
        print("Usage: py predict.py path/to/insect_image.jpg")