import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CheXpertDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        """
        Args:
            csv_file (str): Path to train.csv / val.csv / test.csv
            images_dir (str): Directory containing images
            transform (callable, optional): Image transformations
        """
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["image"]
        label = int(self.data.iloc[idx]["label"])

        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ------------------------------------------
# Transforms (ImageNet-compatible)
# ------------------------------------------

def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
