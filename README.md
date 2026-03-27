# 🐦 BirdTracker-Pro

> **Real-time bird detection, tracking, and analytics powered by YOLOv8 + DeepSORT**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen?style=flat-square)](https://ultralytics.com)
[![DeepSORT](https://img.shields.io/badge/Tracking-DeepSORT-orange?style=flat-square)](https://github.com/nwojke/deep_sort)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-blueviolet?style=flat-square)](CONTRIBUTING.md)

BirdTracker-Pro is a production-ready computer vision pipeline that processes bird videos frame-by-frame — detecting every bird, assigning persistent unique IDs, and exporting a fully annotated output video. No duplicates. No setup headaches. Just drop in a video and run.

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [How It Works](#-how-it-works)
- [Performance Tips](#-performance-tips)
- [Example Output](#-example-output)
- [Tech Stack](#-tech-stack)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Accurate Detection** | YOLOv8 (COCO class 14) with configurable confidence threshold |
| 🧭 **Persistent Tracking** | DeepSORT assigns stable unique IDs across all frames |
| 🔢 **Duplicate-Free Counting** | Counts total unique birds, not per-frame detections |
| 📊 **Live Overlays** | Real-time FPS counter and running bird count on-screen |
| 🎨 **Clean Visualization** | Deterministic color-coded bounding boxes and ID labels |
| 🧠 **Frame Preprocessing** | Brightness, contrast, and noise reduction before detection |
| ⚡ **CPU & GPU Support** | Automatic CUDA detection with graceful CPU fallback |
| 🎥 **Full Video Pipeline** | Reads any MP4 input → writes annotated MP4 output |
| 🛡️ **Production-Ready** | Input validation, structured error handling, clean architecture |

---

## 🗂️ Project Structure

```
BirdTracker-Pro/
├── data/
│   ├── input/
│   │   └── input.mp4          ← Place your video here
│   └── output/
│       └── output.mp4         ← Auto-generated annotated result
├── models/
│   └── yolov8s.pt             ← Auto-downloaded on first run (~22 MB)
├── src/
│   ├── detection/
│   │   └── detector.py        ← YOLOv8 inference wrapper
│   ├── tracking/
│   │   └── tracker.py         ← DeepSORT integration
│   ├── visualization/
│   │   └── visualizer.py      ← Bounding boxes, labels, overlays
│   ├── utils/
│   │   ├── config.py          ← All tunable parameters
│   │   └── helpers.py         ← Shared utility functions
│   └── main.py                ← Entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- pip
- *(Optional but recommended)* NVIDIA GPU with CUDA for real-time speed

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/HussnainZahid/BirdTracker-Pro.git
cd BirdTracker-Pro
```

**2. Create and activate a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your video**

Copy or move your bird footage to the input folder:
```
data/input/input.mp4
```

> The YOLOv8 model weights (`yolov8s.pt`) are downloaded automatically the first time you run the system.

---

## ▶️ Usage

Run the full pipeline with a single command:

```bash
python -m src.main
```

The system will:

1. Validate the input video path and format
2. Process each frame through detection → tracking → visualization
3. Display a live preview window with real-time overlays
4. Save the fully annotated video to `data/output/output.mp4`
5. Print a summary to the console when complete

**To stop early:** press `ESC` in the preview window.

**Console output on completion:**
```
🎉 Processing complete!
   📁 Output saved:            data/output/output.mp4
   🐦 Total unique birds:      83
   📊 Total frames processed:  110
```

---

## 🔧 Configuration

All tunable parameters live in `src/utils/config.py`:

```python
class Config:
    # Paths
    VIDEO_PATH   = "data/input/input.mp4"
    OUTPUT_PATH  = "data/output/output.mp4"
    MODEL_PATH   = "models/yolov8s.pt"

    # Detection
    CONFIDENCE   = 0.4    # Minimum detection confidence (0.0 – 1.0)
    IOU          = 0.5    # Non-maximum suppression IoU threshold
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

    # Display
    SHOW_FPS         = True
    SHOW_TOTAL_COUNT = True

    # Tracking
    MAX_AGE  = 30   # Frames to keep a lost track alive
    N_INIT   = 3    # Frames required before a track is confirmed
```

| Parameter | Effect |
|---|---|
| `CONFIDENCE` | Lower → more detections (higher false positives); raise to filter noise |
| `MAX_AGE` | Higher → tracks survive longer occlusions |
| `N_INIT` | Higher → fewer ghost tracks; lower → faster track confirmation |
| `MODEL_PATH` | Swap to `yolov8n.pt` for faster speed or `yolov8m.pt` for more accuracy |

---

## 🧠 How It Works

```
Input Video
    │
    ▼
Frame Reading        — Load frames sequentially via OpenCV
    │
    ▼
Preprocessing        — Enhance contrast, reduce noise
    │
    ▼
Detection (YOLOv8)   — Identify birds with confidence filtering (COCO class 14)
    │
    ▼
Tracking (DeepSORT)  — Assign and maintain persistent unique IDs across frames
    │
    ▼
Visualization        — Draw colored boxes, ID labels, FPS, and total bird count
    │
    ▼
Output Video         — Write processed frames to annotated MP4
```

DeepSORT combines a **Kalman filter** (for motion prediction) with **appearance features** (for re-identification), ensuring each bird keeps its ID even when temporarily occluded or leaving and re-entering the frame.

---

## 📈 Performance Tips

- **Use a GPU.** CUDA acceleration can be 5–20× faster than CPU, enabling near real-time processing on 1080p video.
- **Choose the right model size.** `yolov8n.pt` is fastest; `yolov8s.pt` balances speed and accuracy; `yolov8m.pt` maximizes accuracy at the cost of speed.
- **Resolution matters.** 720p–1080p gives the best accuracy-to-speed ratio.
- **Tune `CONFIDENCE`.** For dense flocks, lower it slightly. For clean footage with few subjects, raise it to reduce noise.
- **Adjust `N_INIT`.** Setting it to `1` confirms tracks immediately — useful for fast-moving birds that appear briefly.

---

## 📸 Example Output

After processing, your annotated video will be saved to `data/output/output.mp4`. Each bird is rendered with:

- A **unique, color-coded bounding box**
- A **persistent ID label** (e.g., `Bird #7`)
- A **live FPS counter** and **total unique bird count** in the top-left corner

---

## 🛠️ Tech Stack

| Library | Role |
|---|---|
| [Ultralytics YOLOv8](https://ultralytics.com) | Object detection |
| [DeepSORT](https://github.com/nwojke/deep_sort) | Multi-object tracking |
| [OpenCV](https://opencv.org) | Video I/O and frame rendering |
| [PyTorch](https://pytorch.org) | Deep learning backend |
| [NumPy](https://numpy.org) | Numerical operations |

---

## 🗺️ Roadmap

- [ ] Entry/exit zone counting with configurable regions
- [ ] Bird trajectory paths and heatmap overlays
- [ ] Species classification using a fine-tuned model
- [ ] Streamlit web dashboard for live monitoring
- [ ] Real-time webcam and RTSP stream support
- [ ] Alert system for activity anomalies

---

## 🤝 Contributing

Contributions are welcome and appreciated! To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to your branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please make sure your code follows the existing style and includes appropriate comments.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute it with attribution.

---

<div align="center">

**Built with ❤️ by [Hussnain Zahid](https://github.com/HussnainZahid)**

*Drop your video in `data/input/` and run `python -m src.main` — that's all it takes.* 🐦

⭐ **Star this repo** if you find it useful — it helps a lot!

</div># BirdTracker-Pro
