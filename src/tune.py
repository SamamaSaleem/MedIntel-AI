import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import CheXpertDataset, get_transforms
from src.model import DenseNet121


def train_and_evaluate(config, device):
    BASE_DIR = "/content/drive/MyDrive/MedIntel/data/chexpert/processed"
    IMG_DIR = os.path.join(BASE_DIR, "images")

    train_dataset = CheXpertDataset(
        csv_file=os.path.join(BASE_DIR, "train.csv"),
        images_dir=IMG_DIR,
        transform=get_transforms("train")
    )

    val_dataset = CheXpertDataset(
        csv_file=os.path.join(BASE_DIR, "val.csv"),
        images_dir=IMG_DIR,
        transform=get_transforms("val")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2
    )

    model = DenseNet121(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    best_val_acc = 0.0

    for epoch in range(config["epochs"]):
        model.train()
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        best_val_acc = max(best_val_acc, val_acc)

    return best_val_acc, model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    configs = [
        {"lr": 1e-3, "epochs": 5, "batch_size": 16},
        {"lr": 1e-4, "epochs": 10, "batch_size": 16},
    ]

    best_acc = 0.0
    best_model = None
    best_config = None

    for i, cfg in enumerate(configs):
        print(f"\nRunning config {i+1}: {cfg}")
        val_acc, model = train_and_evaluate(cfg, device)
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            best_config = cfg

    os.makedirs("/content/drive/MyDrive/MedIntel/models", exist_ok=True)
    torch.save(
        best_model.state_dict(),
        "/content/drive/MyDrive/MedIntel/models/best_tuned_densenet121.pth"
    )

    print("\nBest Configuration:", best_config)
    print("Best Validation Accuracy:", best_acc)


if __name__ == "__main__":
    main()
