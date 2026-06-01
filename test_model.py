import torch
import json
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
from torchvision import models
import torch.nn as nn

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Load test data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Use same DATA_ROOT as training
DATA_ROOT = "./dataset"
dataset = datasets.ImageFolder(root=DATA_ROOT, transform=transform)
test_size = int(0.15 * len(dataset))
_, test_ds = torch.utils.data.random_split(dataset, [len(dataset)-test_size, test_size])

test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Evaluate
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Print results
print("\n=== Test Results ===")
print(classification_report(all_labels, all_preds, target_names=class_names))