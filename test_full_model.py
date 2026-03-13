import torch
from torch.utils.data import DataLoader
from utils.dataset_loader import FaceDataset
from models.deepfake_model import DeepfakeDetector

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = FaceDataset("dataset/faces_test")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = DeepfakeDetector().to(device)
model.eval()

for images in loader:
    images = images.to(device)
    outputs = model(images)
    print("Output shape:", outputs.shape)
    print("Predictions:", outputs)
    break
    