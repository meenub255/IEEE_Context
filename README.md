# 🛡️ Context-Aware Road Hazard Prediction System

[![IEEE Prototype](https://img.shields.io/badge/IEEE-Prototype-blue.svg)](https://ieeexplore.ieee.org/)
[![Neural Architecture](https://img.shields.io/badge/Architecture-Hybrid--DL-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

A high-fidelity, Tesla-inspired autonomous safety system designed for real-time road hazard prediction. This project leverages a multi-modal neural architecture to provide **Forward Collision Warnings (FCW)** and **Autonomous Emergency Braking (AEB)** by synthesizing spatial, temporal, and environmental context.

Developed for IEEE presentation by **The Mad Scientists**.

---

## 🚀 Key Features

*   **🧠 Hybrid Spatial-Temporal Engine**: Combines YOLOv8 (Object Detection), ResNet-18 (Spatial Encoding), and a Uni-GRU (Temporal Anticipation).
*   **🌍 Context-Aware Logic**: Dynamically adjusts hazard sensitivity based on weather (Clear, Rain, Fog, Snow), lighting (Day, Night), and road types (Highway, Urban).
*   **🌙 Tesla-Grade Night Vision**: A dedicated image processing pipeline featuring dynamic gamma correction, glare suppression, and bilateral de-noising for extreme low-light performance.
*   **⚠️ Speed-Adaptive FCW**: Intelligent Time-to-Collision (TTC) zones that escalate alerts based on relative closing speeds, not just static distance.
*   **🛑 Hardened AEB System**: Temporal consistency checks prevent "phantom braking" by requiring sustained hazard probability before engagement.
*   **🕳️ Static Hazard Scanning**: Parallel inference stream for detecting potholes and structural road damage.
*   **📊 Unified Telemetry Dash**: Real-time visualization via Streamlit with downloadable CSV logs for post-drive analysis.

---

## 🏗️ Neural Architecture

The system utilizes a complex multi-stage pipeline to achieve "anticipatory" safety:

1.  **Object Detection (YOLOv8n)**: Real-time identification of dynamic actors (cars, pedestrians, trucks) and static hazards (potholes).
2.  **Spatial Encoder (ResNet-18)**: Extracts 256-D feature vectors representing the global visual context of each frame.
3.  **Temporal Uni-GRU**: A Gated Recurrent Unit that processes 15-frame sequences of:
    *   **Visual Embeddings** (256-D)
    *   **Vehicle Telemetry** (Distance, Relative Speed, Friction Modifiers)
    *   **Environmental Embeddings** (Weather and Road categoricals)
4.  **Prediction Heads**: Outputs continuous **Hazard Probability (0-1)** and **Estimated TTC (seconds)**.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/IEEE_Context.git
cd IEEE_Context
```

### 2. Set Up Environment
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Model Weights
Ensure the following weights are in the root directory:
*   `spatial_encoder.pt` (ResNet-18 Features)
*   `temporal_gru.pt` (Uni-GRU Logic)
*   `yolov8n.pt` (Standard Detection)
*   `yolov8_night_highway.pt` (Optimized Night Model)
*   `yolov8_pothole.pt` (Static Hazard Model)

---

## 🖥️ Usage

### DASHBOARD (Recommended)
Run the interactive Streamlit dashboard to process videos with real-time HUD overlays:
```bash
streamlit run app.py
```

### Command Line Pipeline
To process a video file directly:
```bash
python video_pipeline.py input_dashcam.mp4 output_processed.mp4
```

---

## 🌦️ Environmental Modifiers

The system calculates a **Friction/Visibility Modifier ($f_{mod}$)** which scales the hazard intensity:

| Condition | Friction Coeff | Logic Bias |
| :--- | :--- | :--- |
| **Clear / Day** | 1.00x | Nominal Sensitivity |
| **Rain** | 0.75x | Early Braking Bias |
| **Fog** | 0.85x | Contrast Recovery Active |
| **Snow** | 0.50x | Hyper-Vigilance Mode |
| **Night** | 0.50x | Night Vision Pipeline Engaged |

---

## 📜 Acknowledgments

Developed by **The Mad Scientists** as a research prototype for context-aware autonomous systems.
*   **Neural Models**: Custom-trained on ACDC, ExDark, and Crash datasets.
*   **Inspiration**: Tesla Autopilot (V12 HUD aesthetics and FCW logic).
*   **Frameworks**: PyTorch, Ultralytics, Streamlit.

---
> [!IMPORTANT]
> This is a research prototype. Do not use for actual vehicle control. AEB and FCW outputs are simulated for demonstration purposes.
