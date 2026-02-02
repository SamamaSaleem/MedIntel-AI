# ==============================
# MedIntel – Gradio Interface
# ==============================

import os
import sys

# -------------------------------------------------
# Ensure project root is in PYTHONPATH
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# -------------------------------------------------
# Imports
# -------------------------------------------------
import gradio as gr
import torch
from PIL import Image

from src.agent import MedIntelAgent
from src.gradcam import generate_gradcam_comparison

# -------------------------------------------------
# Device (CPU safest for Colab demo)
# -------------------------------------------------
device = torch.device("cpu")

# -------------------------------------------------
# Paths
# -------------------------------------------------
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_densenet121_final.pth"
)

# -------------------------------------------------
# Initialize Agent (Agent owns the model)
# -------------------------------------------------
agent = MedIntelAgent(
    model_path=MODEL_PATH,
    device=device
)

# -------------------------------------------------
# Inference + Agent + Explainability
# -------------------------------------------------
def analyze_xray(image: Image.Image):
    try:
        if image is None:
            return {"error": "No image uploaded"}, None

        # ----------------------------------
        # Save uploaded image temporarily
        # ----------------------------------
        temp_image_path = "/tmp/input_xray.png"
        image = image.convert("RGB")
        image.save(temp_image_path)

        # ----------------------------------
        # Agentic analysis (CORRECT DESIGN)
        # ----------------------------------
        agent_result = agent.analyze_image(temp_image_path)

        result = {
            "Normal probability": agent_result["normal_probability"],
            "Abnormal probability": agent_result["abnormal_probability"],
            "Agent decision": agent_result["agent_decision"]
        }

        # ----------------------------------
        # Grad-CAM (NON-FATAL)
        # ----------------------------------
        cam_image = None
        try:
            cam_output_path = "/tmp/gradcam_result.png"

            generate_gradcam_comparison(
                image_path=temp_image_path,
                output_path=cam_output_path
            )

            cam_image = Image.open(cam_output_path)

        except Exception as cam_error:
            print("Grad-CAM failed:", cam_error)

        return result, cam_image

    except Exception as e:
        return {"error": str(e)}, None

# -------------------------------------------------
# Gradio Interface
# -------------------------------------------------
demo = gr.Interface(
    fn=analyze_xray,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray"),
    outputs=[
        gr.JSON(label="Prediction & Agent Decision"),
        gr.Image(label="Grad-CAM (Original vs Heatmap)")
    ],
    title="MedIntel – Agentic Chest X-ray Decision Support",
    description=(
        "This agentic AI system analyzes chest X-rays and provides:\n"
        "• Prediction probabilities\n"
        "• Confidence-aware agent decision\n"
        "• Grad-CAM visual explanation\n\n"
        "For academic demonstration only."
    ),
    flagging_mode="never"
)

# -------------------------------------------------
# Launch (PUBLIC LINK – GUARANTEED TO WORK)
# -------------------------------------------------
if __name__ == "__main__":
    demo.launch(share=True)
