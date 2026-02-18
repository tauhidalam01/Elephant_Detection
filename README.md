# 🐘 Elephant Detection using YOLOv8 + Streamlit

A deep learning-based computer vision system designed to **detect elephants in thermal images** using **YOLOv8**.

The model is trained on a **multi-class dataset**, where:

- **Class 0 → Elephant (Primary Target)**
- Other classes → Animal / Human / Device / Unknown

🎯 **Primary Goal:** Reliable Elephant Warning System

---

## 🚀 Live Demo

🔗 Streamlit App:  
https://tauhidalam01-elephant-detection-app-btyyfx.streamlit.app/

Upload a thermal image → Get detection results instantly.

---

## 🎯 Project Objective

Although trained as a **multi-class detector**, the real-world application focuses on:

✅ **Elephant vs Non-Elephant Detection**

Logic:

- If **Class 0 (Elephant)** detected → Trigger Alert 🚨
- Any other class → Treated as Non-Elephant

---

## 🧠 Why Multi-Class Training?

✔ Improves elephant discrimination  
✔ Helps differentiate elephants from humans/devices/animals  
✔ Reduces false positives  
✔ Enhances robustness  

---

## 🏗 Model Details

| Component | Value |
|----------|-------|
| **Architecture** | YOLOv8n (Nano) |
| **Framework** | Ultralytics YOLOv8 |
| **Task** | Object Detection |
| **Primary Class** | Elephant (Class 0) |

---

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| **mAP@0.5 (all classes)** | **0.822** |
| **Best F1 (all classes)** | **0.83 @ confidence 0.62** |
| **Elephant Confidence** | **> 0.95 @ 0.62** |

Model optimized primarily for **Elephant detection**.

---

## 🧠 Model Architecture – How It Works

### 🔹 1. Input
Thermal image uploaded via Streamlit UI.

### 🔹 2. Preprocessing
Handled internally by YOLO:

✔ Resize  
✔ Normalization  
✔ Tensor conversion  

### 🔹 3. Feature Extraction (Backbone)
Extracts spatial features.

### 🔹 4. Multi-Scale Fusion (Neck)
Combines features at multiple resolutions.

### 🔹 5. Detection Head
Predicts:

✔ Bounding Boxes  
✔ Class Labels  
✔ Confidence Scores  

### 🔹 6. Post-Processing
✔ Non-Max Suppression (NMS)  
✔ Confidence filtering  

### 🔹 7. Output
Annotated detection image shown in Streamlit.

---

## 🚨 Elephant Warning Logic

Example:

```python
if cls == 0 and conf > 0.62:
    trigger_alert()
```
---

## 🛠 Tech Stack

- Python
- Ultralytics YOLOv8
- PyTorch
- Streamlit
- OpenCV
- NumPy
- Pillow

---

## 📂 Project Structure
elephant_detection/
│── app.py
│── train.py
│── data.yaml
│── requirements.txt
│── packages.txt
│── runtime.txt
│── train_results/
│ └── elephant_yolov8n_multi_class/
│ └── weights/
│ └── best.pt

---

## ⚙️ Installation (Local Setup)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/tauhidalam01/elephant_detection.git
cd elephant_detection

---

## create virtual environment

python -m venv venv

---

## for windows

venv\Scripts\activate

---

## for linux

source venv/bin/activate

---

## requirements.txt

streamlit
ultralytics
opencv-python-headless
pillow
numpy
torch

---

## packages.txt(streamlit cloud)

libgl1
libglib2.0-0

---

## train the model

python train.py

---

## Fututre Improvement

- Real-time thermal camera feed

- Edge device deployment (Jetson Nano / Raspberry Pi)

- SMS / Siren-based alert system

- Elephant-only binary classifier

- Model quantization for faster inference

---

## Author

Tauhid Alam

---




