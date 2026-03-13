import torch
from torch.utils.data import DataLoader
from utils.dataset_loader import FaceDataset
from models.cnn_model import CNNFeatureExtractor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset
dataset = FaceDataset("dataset/faces_test")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load model
model = CNNFeatureExtractor().to(device)
model.eval()

# Test one batch
for images in loader:
    images = images.to(device)
    features = model(images)
    print("Feature shape:", features.shape)
    break