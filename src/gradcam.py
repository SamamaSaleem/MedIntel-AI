import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

from src.model import DenseNet121
from src.dataset import get_transforms


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # Use DenseNet last conv layer BEFORE inplace ReLU
        target_layer = self.model.model.features.denseblock4.denselayer16.conv2
        self._register_hooks(target_layer)

    def _register_hooks(self, layer):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam


def generate_gradcam_comparison(image_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # Load trained model
    # -------------------------------
    model = DenseNet121(num_classes=2).to(device)
    model.load_state_dict(
        torch.load(
            "/content/drive/MyDrive/MedIntel/models/best_densenet121_final.pth",
            map_location=device
        )
    )
    model.eval()

    gradcam = GradCAM(model)

    # -------------------------------
    # Load and preprocess image
    # -------------------------------
    transform = get_transforms("val")
    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # -------------------------------
    # Prediction
    # -------------------------------
    outputs = model(input_tensor)
    probs = F.softmax(outputs, dim=1)
    class_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][class_idx].item()

    # -------------------------------
    # Generate Grad-CAM
    # -------------------------------
    cam = gradcam.generate(input_tensor, class_idx)

    # -------------------------------
    # Visualization
    # -------------------------------
    original_img = cv2.imread(image_path)
    original_img = cv2.resize(original_img, (224, 224))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    comparison = np.hstack((original_img, overlay))

    cv2.imwrite(output_path, comparison)

    print("\nGrad-CAM comparison saved to:", output_path)
    print("Predicted class:", "Abnormal" if class_idx == 1 else "Normal")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    SAMPLE_IMAGE = (
        "/content/drive/MyDrive/MedIntel/data/chexpert/processed/images/"
        "train_patient00069_study14_view1_frontal.jpg"
    )

    OUTPUT_PATH = (
        "/content/drive/MyDrive/MedIntel/results/gradcam_comparison.jpg"
    )

    os.makedirs("/content/drive/MyDrive/MedIntel/results", exist_ok=True)

    generate_gradcam_comparison(SAMPLE_IMAGE, OUTPUT_PATH)
