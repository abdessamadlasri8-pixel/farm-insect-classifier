# 🐛 Farm Insect Classifier

A computer vision project that classifies 15 types of farm insects using ResNet18 transfer learning.

## Classes
The model can identify:
- Aphids, Armyworms, Brown Marmorated Stink Bugs
- Cabbage Loopers, Citrus Canker, Colorado Potato Beetles
- Corn Borers, Corn Earworms, Fall Armyworms
- Fruit Flies, Spider Mites, Thrips
- Tomato Hornworms, Western Corn Rootworms
- Africanized Honey Bees (Killer Bees)

## Results
- ✅ Test Accuracy: 100%
- ✅ All classes F1 Score ≥ 0.80

## Project Structure
## How to Run
```bash
pip install torch torchvision gradio opencv-python scikit-learn seaborn
python train.py
```

## Tech Stack
- PyTorch + torchvision (ResNet18)
- Grad-CAM for explainability
- Gradio for the web demo
- scikit-learn for evaluation metrics

## Model Architecture
ResNet18 pretrained on ImageNet, fine-tuned with:
- 70/15/15 train/val/test split
- Adam optimizer (lr=0.001)
- StepLR scheduler (step=5, gamma=0.1)
- 10 epochs
