import os
import torch
import torch.nn.functional as F
from PIL import Image

from src.model import DenseNet121
from src.dataset import get_transforms


class MedIntelAgent:
    def __init__(
        self,
        model_path,
        device=None,
        high_conf=0.85,
        medium_conf=0.65
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = DenseNet121(num_classes=2).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

        self.transform = get_transforms("val")

        self.high_conf = high_conf
        self.medium_conf = medium_conf

    def analyze_image(self, image_path):
        """
        Runs agentic analysis on a single X-ray image.
        """
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probs = F.softmax(outputs, dim=1)

        abnormal_prob = probs[0][1].item()
        normal_prob = probs[0][0].item()

        decision = self._agent_decision(abnormal_prob)

        return {
            "image": os.path.basename(image_path),
            "normal_probability": round(normal_prob, 4),
            "abnormal_probability": round(abnormal_prob, 4),
            "agent_decision": decision
        }

    def _agent_decision(self, abnormal_prob):
        """
        Rule-based agent decision logic.
        """
        if abnormal_prob >= self.high_conf:
            return "Likely abnormal (high confidence) – immediate review recommended"
        elif abnormal_prob >= self.medium_conf:
            return "Possibly abnormal – clinical review advised"
        else:
            return "Uncertain / likely normal – human verification recommended"


if __name__ == "__main__":
    # -------------------------------
    # Example usage
    # -------------------------------
    MODEL_PATH = "/content/drive/MyDrive/MedIntel/models/best_densenet121_final.pth"
    SAMPLE_IMAGE = "/content/drive/MyDrive/MedIntel/data/chexpert/processed/images/sample.jpg"

    agent = MedIntelAgent(model_path=MODEL_PATH)

    result = agent.analyze_image(SAMPLE_IMAGE)
    print("\nAGENT OUTPUT")
    for k, v in result.items():
        print(f"{k}: {v}")
