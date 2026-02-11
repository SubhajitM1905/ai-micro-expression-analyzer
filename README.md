# AI-MICROEXPRESSION-ANALYZER

Real-time facial micro-expression analysis system that estimates **stress, hesitation, and emotional leakage** while a person speaks — powered by **MediaPipe Face Mesh (478 landmarks)** and **OpenCV**.



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


## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.
