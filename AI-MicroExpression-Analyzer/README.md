# 🧠 AI Micro-Expression Analyzer

Real-time facial micro-expression analysis system that estimates **stress, hesitation, and emotional leakage** while a person speaks — powered by **MediaPipe Face Mesh (478 landmarks)** and **OpenCV**.

---

## ✨ Features

| Capability | How it works |
|---|---|
| **Eyebrow movement** | Tracks vertical distance between brow landmarks and upper eyelid anchor |
| **Lip tension** | Computes mouth width / height ratio; clenched lips → high tension |
| **Blink rate** | Eye Aspect Ratio (EAR) per frame; counts blink events per minute |
| **Head micro-nods** | Frame-to-frame nose-tip Y delta normalized by head length |
| **Facial symmetry** | Left-cheek vs right-cheek distance to nose tip |

All five signals are fused with a weighted heuristic model to produce a single **stress score** mapped to three output levels:

| Level | Indicator |
|---|---|
| 🟢 **Calm** | Score < 0.35 |
| 🟡 **Slight Stress** | 0.35 ≤ Score < 0.65 |
| 🔴 **High Stress / Possible Deception** | Score ≥ 0.65 |

---

## 📁 Project Structure

```
AI-MicroExpression-Analyzer/
├── __init__.py              # Package marker
├── face_mesh_module.py      # MediaPipe FaceLandmarker wrapper & camera stream
├── feature_engineering.py   # Extract 5 facial features from 478 landmarks
├── stress_model.py          # Weighted heuristic stress estimator
├── data_logger.py           # CSV session logger
├── dashboard.py             # Terminal text dashboard
├── main.py                  # Entry-point: OpenCV visual overlay + main loop
└── face_landmarker.task     # MediaPipe model (downloaded at setup)
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/AI-MicroExpression-Analyzer.git
cd AI-MicroExpression-Analyzer
```

### 2. Create a virtual environment & install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

pip install mediapipe opencv-python numpy
```

### 3. Download the FaceLandmarker model

```bash
### 3. Download the FaceLandmarker Model (Required)

This project depends on the MediaPipe **Face Landmarker Task model**.

Download the file:

👉 https://developers.google.com/mediapipe/solutions/vision/face_landmarker

OR directly:

https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

After downloading, place the file inside the project package directory:


```

### 4. Run the analyzer

```bash
# With live OpenCV window (default)
python -m ai_micro_expression.main


# With verbose terminal output
python -m ai_micro_expression.main --verbose
# Headless mode (terminal + CSV logging only)
python -m ai_micro_expression.main --no-display
```

Press **`q`** in the OpenCV window to quit.

---

## 🖥️ OpenCV Live Dashboard

When running with display enabled you will see:

- **Left panel** — your live camera feed with 478 face-mesh landmark dots and a colour-coded border (green / amber / red)
- **Right panel** — dark sidebar showing:
  - Stress level banner with score
  - Progress bars for each metric (Eyebrow Raise, Lip Tension, Head Nod, Symmetry, Blink Rate)
  - Colour legend

---
---
## 📸 Demo

<p align="center">
  <img src="assets/dashboard.png" width="800">
</p>



## ⚙️ CLI Options

| Flag | Default | Description |
|---|---|---|
| `--camera-index` | `0` | Camera device index |
| `--log-path` | `logs/session.csv` | Path for the CSV session log |
| `--no-display` | off | Disable the OpenCV window |
| `--verbose` | off | Print full metric breakdown to terminal |

---

## 📊 Data Logging

Every frame's metrics are appended to a CSV file (default `logs/session.csv`) with columns:

```
timestamp, eyebrow_raise, lip_tension, head_nod_intensity, symmetry_delta, blink_rate, stress_score
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **MediaPipe** (Tasks API — `FaceLandmarker`, 478 landmarks)
- **OpenCV** (camera capture + visual overlay)
- **NumPy** (vector math & signal smoothing)

---

## 📌 Notes

- This is a **heuristic-based** system. It does **not** use a trained ML classifier for stress detection — it maps hand-crafted facial features to stress levels via weighted thresholds.
- The `face_landmarker.task` model file (~3.7 MB) is **not** committed to the repo. Download it using the curl command above.
- macOS users may need to grant **camera permission** to Terminal / VS Code.

---

## 📄 License

MIT — feel free to use, modify, and distribute.
