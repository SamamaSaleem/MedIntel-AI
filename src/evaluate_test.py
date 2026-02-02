import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import CheXpertDataset, get_transforms
from src.model import DenseNet121


def main():
    # ----------------------------------
    # Paths
    # ----------------------------------
    BASE_DIR = "/content/drive/MyDrive/MedIntel/data/chexpert/processed"
    IMG_DIR = os.path.join(BASE_DIR, "images")
    MODEL_PATH = "/content/drive/MyDrive/MedIntel/models/best_densenet121_final.pth"

    # ----------------------------------
    # Hyperparameters
    # ----------------------------------
    BATCH_SIZE = 16

    # ----------------------------------
    # Device
    # ----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----------------------------------
    # Test Dataset & Loader
    # ----------------------------------
    test_dataset = CheXpertDataset(
        csv_file=os.path.join(BASE_DIR, "test.csv"),
        images_dir=IMG_DIR,
        transform=get_transforms("val")  # same as validation
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ----------------------------------
    # Load Model
    # ----------------------------------
    model = DenseNet121(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # ----------------------------------
    # Evaluation
    # ----------------------------------
    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(test_dataset)

    # ----------------------------------
    # Metrics
    # ----------------------------------
    print("\n========== TEST RESULTS ==========")
    print(f"Test Loss: {test_loss:.4f}\n")

    print("Classification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=["Normal", "Abnormal"],
            digits=4
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("==================================\n")


if __name__ == "__main__":
    main()
