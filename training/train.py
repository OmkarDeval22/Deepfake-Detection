import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.deepfake_model import DeepfakeDetector
from utils.labeled_dataset import DeepfakeDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Load dataset
dataset = DeepfakeDataset("F:/deepfake/dataset")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = DeepfakeDetector().to(device)

# Loss & Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

# Training
epochs = 10

for epoch in range(epochs):
    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save model in ROOT (important for UI)
torch.save(model.state_dict(), "deepfake_model.pth")
print("Model saved successfully.")