import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        # ✅ Improved transforms (augmentation + normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        for label, category in enumerate(["real", "fake"]):
            folder = os.path.join(root_dir, category)

            if not os.path.exists(folder):
                raise ValueError(f"Folder not found: {folder}")

            for file in os.listdir(folder):
                path = os.path.join(folder, file)

                # Only load images
                if path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return image, label