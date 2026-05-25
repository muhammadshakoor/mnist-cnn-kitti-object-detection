# MNIST CNN & KITTI Object Detection 🧠

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![KITTI](https://img.shields.io/badge/Dataset-KITTI-blue?style=flat-square)
![MNIST](https://img.shields.io/badge/Dataset-MNIST-green?style=flat-square)

> **CET3013 Deep Learning — Assessment 2.** Two deep learning tasks: comparing 5 CNN architectures on handwritten digit recognition (MNIST), and fine-tuning a pre-trained Faster R-CNN model to detect Cars, Pedestrians, and Cyclists in real driving footage (KITTI).

---

## ✨ Features

- 🔢 **Task 1** — Build and compare 5 CNN architectures (Baseline, More Filters, Deeper, BatchNorm, ELU+BN) on the MNIST digit dataset
- 🚗 **Task 2** — Fine-tune Faster R-CNN (MobileNetV3 backbone) on KITTI dashcam data across 3 experiments
- 📊 Automatic generation of training curves, confusion matrices, mAP comparisons, and prediction visualisations
- 🧪 Three detection experiments: frozen backbone, full fine-tuning, and temporal frame stacking
- 💾 Trained model weights saved as `.pth` files for reuse without re-training
- 📓 Both `.ipynb` notebooks and plain `.py` scripts provided

---

## 📁 Project Structure

```
DL Assignment Dataset/
├── task1_mnist_cnn.ipynb        # Task 1 — MNIST CNN notebook
├── task1_mnist_cnn.py           # Task 1 — plain Python script
├── task2_kitti_detection.ipynb  # Task 2 — KITTI detection notebook
├── task2_kitti_detection.py     # Task 2 — plain Python script
├── README.md                    # This file
├── SETUP_GUIDE.md               # Step-by-step setup instructions
├── PROJECT_OVERVIEW.md          # Full code walkthrough with outputs explained
├── .gitignore                   # Excludes dataset, models, and generated images
│
├── VideoThree/ … VideoEight/    # KITTI sequences — NOT in repo (download separately)
└── mnist_data/                  # MNIST cache — NOT in repo (auto-downloads)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Deep Learning | Python · PyTorch · TorchVision |
| Task 1 Dataset | MNIST (auto-downloaded via TorchVision) |
| Task 2 Dataset | KITTI Raw Data (download separately — see below) |
| Detection Model | Faster R-CNN · MobileNetV3 · FPN |
| Evaluation | IoU · Average Precision · mAP |
| Visualisation | Matplotlib · Seaborn · scikit-learn |
| Environment | Anaconda · Jupyter Notebook |

---

## 📋 Task Summary

### Task 1 — MNIST CNN Comparison

| Model | Architecture | Batch Norm | Activation | Best Accuracy |
|-------|-------------|------------|------------|---------------|
| A | 2 conv (32, 64) | No | ReLU | ~99.2% |
| B | 2 conv (64, 128) | No | ReLU | ~99.3% |
| C | 3 conv (32, 64, 128) | No | ReLU | ~99.4% |
| D | 3 conv (32, 64, 128) | Yes | ReLU | ~99.5% |
| E | 3 conv (32, 64, 128) | Yes | ELU | ~99.5% |

### Task 2 — KITTI Object Detection

| Experiment | Setup | mAP |
|------------|-------|-----|
| Exp 1 | Frozen backbone, head only | ~0.52 |
| Exp 2 | Full fine-tuning | ~0.57 |
| Exp 3 (Bonus) | Temporal frame stacking (6-channel) | ~0.59 |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/muhammadshakoor/mnist-cnn-kitti-object-detection.git
cd mnist-cnn-kitti-object-detection

# 2. Install dependencies (in Anaconda Prompt)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib Pillow scikit-learn seaborn jupyter

# 3. Verify installation
python -c "import torch; import torchvision; print('ALL OK')"

# 4. Launch Jupyter
jupyter notebook
```

- Open `task1_mnist_cnn.ipynb` — MNIST downloads automatically on first run
- Open `task2_kitti_detection.ipynb` — update `BASE_DIR` to your local path, then run

> See [SETUP_GUIDE.md](SETUP_GUIDE.md) for the complete step-by-step guide including troubleshooting.

---

## 📦 KITTI Dataset Setup

The KITTI data (~3.5 GB) is **not included** in this repository. Download it from the official source:

1. Register at **https://www.cvlibs.net/datasets/kitti/raw_data.php**
2. Download the drives listed below (`_sync.zip` + `_tracklets.zip` + `2011_09_26_calib.zip`):

| Folder | KITTI Drive | Role |
|--------|-------------|------|
| `VideoThree` | `2011_09_26_drive_0005` | Training |
| `VideoFive` | `2011_09_26_drive_0011` | Training |
| `VideoSeven` | `2011_09_26_drive_0014` | Training |
| `Video11` | `2011_09_26_drive_0051` | Training |
| `Video12` | `2011_09_26_drive_0056` | Training |
| `Video16` | `2011_09_26_drive_0084` | Validation |
| `Video15` | `2011_09_26_drive_0060` | Test |

> Minimum download (training + val + test only): ~1.5 GB. Full dataset with all 18 sequences: ~3.5 GB.

---

## 📄 License

MIT © [Muhammad Shakoor](https://github.com/muhammadshakoor)
