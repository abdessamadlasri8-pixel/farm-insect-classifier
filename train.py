
import torch
import torch.nn.functional as F
import numpy as np
import json, cv2, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets, models
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.patches as mpatches

# ── Config ─────────────────────────────────────────────────────────────────
DATA_ROOT   = "./data/farm_insects"
BATCH_SIZE  = 32
NUM_EPOCHS  = 10
LR          = 0.001
SEED        = 42
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transforms ─────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Dataset ────────────────────────────────────────────────────────────────
dataset     = datasets.ImageFolder(root=DATA_ROOT, transform=val_test_transform)
class_names = dataset.classes
num_classes = len(class_names)

total = len(dataset)
train_size, val_size = int(0.70*total), int(0.15*total)
test_size = total - train_size - val_size

generator = torch.Generator().manual_seed(SEED)
train_ds, val_ds, test_ds = random_split(
    dataset, [train_size, val_size, test_size], generator=generator)

train_ds.dataset.transform = train_transform
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ── Model ──────────────────────────────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ── Training ───────────────────────────────────────────────────────────────
best_val_acc = 0.0
for epoch in range(NUM_EPOCHS):
    model.train()
    correct, total_n, train_loss = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total_n += labels.size(0)

    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    val_acc = val_correct / val_total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

    scheduler.step()
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Acc: {correct/total_n:.4f} | Val Acc: {val_acc:.4f}")

with open("class_names.json", "w") as f:
    json.dump(class_names, f)

print(f"")
Best Val Acc: {best_val_acc:.4f}")
print("Saved best_model.pth and class_names.json")
