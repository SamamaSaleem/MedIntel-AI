# MedIntel — Agentic Chest X-ray Decision Support System

## Overview
MedIntel is an AI-powered clinical decision support prototype that analyzes chest X-ray images and provides:

• Abnormality prediction probabilities  
• Confidence-aware agent decision logic  
• Grad-CAM visual explanations  
• Interactive Gradio web interface  

This system demonstrates how Computer Vision and Agent-based reasoning can assist clinicians in fast triage of medical images.

This project is developed strictly for academic demonstration purposes.

---

## Features
- PyTorch DenseNet121 CNN classifier
- CheXpert dataset preprocessing pipeline
- Training + evaluation scripts
- Rule-based agent reasoning layer
- Grad-CAM explainability
- Gradio web UI demo

---

## Tech Stack
- Python 3.10+
- PyTorch
- Torchvision
- OpenCV
- Scikit-learn
- Pandas
- Gradio

---

## Project Structure
```
MedIntel/
│
├── app/               # Gradio UI
├── src/               # Core logic (model, training, agent, gradcam)
├── data/              # Processed dataset
├── models/            # Saved weights
├── results/           # Outputs & heatmaps
├── notebooks/         # Demo notebook
├── README.md
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python -m src.train
```

---

## Evaluation

```bash
python -m src.evaluate_test
```

---

## Agent Inference

```bash
python -m src.agent
```

---

## Grad-CAM Explanation

```bash
python -m src.gradcam
```

---

## Run Web Demo

```bash
python app/app_gradio.py
```

Open browser → http://localhost:7860

---

## Demo Workflow
1. Upload chest X-ray
2. System predicts normal/abnormal probability
3. Agent provides clinical recommendation
4. Grad-CAM heatmap highlights suspicious regions

---

## Academic Note
This project is a research/educational prototype and NOT a medical diagnostic system.
