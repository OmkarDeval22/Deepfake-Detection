import torch
import torch.nn as nn
from models.cnn_model import CNNFeatureExtractor


class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()

        self.feature_extractor = CNNFeatureExtractor()

        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output