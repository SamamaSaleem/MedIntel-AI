import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import CheXpertDataset, get_transforms
from src.model import DenseNet121


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    # ----------------------------------
    # Paths
    # ----------------------------------
    BASE_DIR = "/content/drive/MyDrive/MedIntel/data/chexpert/processed"
    IMG_DIR = os.path.join(BASE_DIR, "images")
    MODEL_DIR = "/content/drive/MyDrive/MedIntel/models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ----------------------------------
    # FINAL ANTI-OVERFITTING HYPERPARAMS
    # ----------------------------------
    BATCH_SIZE = 16
    NUM_EPOCHS = 8
    LR = 5e-5
    WEIGHT_DECAY = 1e-4   # L2 regularization

    # ----------------------------------
    # Device
    # ----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----------------------------------
    # Datasets & Loaders
    # ----------------------------------
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
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ----------------------------------
    # Model, Loss, Optimizer
    # ----------------------------------
    model = DenseNet121(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # ----------------------------------
    # Training Loop
    # ----------------------------------
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best model (validation-based)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(
                model.state_dict(),
                os.path.join(MODEL_DIR, "best_densenet121_final.pth")
            )

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
