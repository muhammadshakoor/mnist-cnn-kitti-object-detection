# CET3013 Deep Learning — Project Overview
### University of Sunderland | School of Computer Science & Engineering
### Assessment 2 of 2 | Individual Practical Deep Learning Project | 70% of Final Mark

---

## Assignment Details

| Field | Details |
|-------|---------|
| Module Code | CET3013 |
| Module Title | Deep Learning |
| Module Leader | Ibrahim Alqatawneh |
| Assessment | 2 of 2 |
| Assessment Title | Individual Practical Deep Learning Project |
| Assessment Value | 70% of final module mark |
| Submission | Electronic submission to Canvas |
| Referencing Style | Harvard Referencing |

---

## What This Assignment Requires

The assignment tests two key learning outcomes:

- **LO1** — Design, implement, and critically appraise deep learning solutions using industry-standard tools, frameworks, and methodologies
- **LO2** — Analyse and justify the selection, adaptation, and design of deep learning approaches to meet specific project requirements

The overall aim is to assess understanding of key computer vision techniques through two complementary tasks — one on image classification from scratch, and one on real-world object detection using transfer learning.

---

---

# TASK 1 — What the Brief Requires vs What Was Achieved

## What the Brief Asked For

The brief required building a CNN from scratch for multi-class image classification on the MNIST dataset (70,000 grayscale images of handwritten digits 0–9).

Specifically, the brief required:

1. Load and preprocess the MNIST dataset using PyTorch
2. Partition data into training and test sets
3. Design a CNN architecture from scratch
4. Experiment with core architectural components:
   - Number of layers (depth)
   - Number of filters
   - Activation functions (e.g., ReLU, ELU)
   - Batch Normalisation
   - Pooling layers
5. Compare multiple architectural variations systematically
6. Use appropriate evaluation metrics (accuracy, loss, F1 score, confusion matrix)
7. Produce accuracy and loss learning curves
8. Use academic literature to justify design choices
9. Select and justify the optimal architecture

---

## What Was Achieved in Task 1

### Dataset Handling
- MNIST dataset loaded using `torchvision.datasets.MNIST` with automatic download
- Applied standard preprocessing: `ToTensor()` and `Normalize(mean=0.1307, std=0.3081)`
- Split into 60,000 training samples and 10,000 test samples
- Used `DataLoader` with batch size 64 for efficient mini-batch training

### Five CNN Architectures Designed and Compared

| Model | Architecture | Filters | Activation | Batch Norm | Parameters |
|-------|-------------|---------|------------|------------|------------|
| **A — Baseline** | 2 conv layers | 32, 64 | ReLU | No | 421,642 |
| **B — More Filters** | 2 conv layers | 64, 128 | ReLU | No | 1,682,954 |
| **C — Deeper** | 3 conv layers | 32, 64, 128 | ReLU | No | 1,701,130 |
| **D — BatchNorm** | 3 conv layers | 32, 64, 128 | ReLU | **Yes** | 1,702,090 |
| **E — ELU + BN** | 3 conv layers | 32, 64, 128 | **ELU** | Yes | 1,702,090 |

**Design progression followed:**
- Started simple (Model A — baseline)
- Increased filter capacity (Model B)
- Added depth (Model C)
- Added Batch Normalisation (Model D)
- Replaced activation with ELU (Model E)

This systematic approach directly satisfies the brief's requirement for an evidence-based investigation.

### Training Setup
- Optimiser: Adam (lr = 0.001)
- Loss function: CrossEntropyLoss
- Epochs: 10 per model
- Random seed fixed (42) for reproducibility
- All models trained on the same data for fair comparison

### Evaluation Metrics Used
- Training and test accuracy per epoch
- Training and test loss per epoch
- Confusion matrix (10×10 heatmap)
- Per-class accuracy bar chart
- Classification report with Precision, Recall, F1-score per digit class
- Macro F1 Score (overall model quality metric)

### Visual Outputs Produced

| Output Image | What It Shows | Brief Requirement Met |
|-------------|---------------|----------------------|
| `task1_mnist_samples.png` | 10 sample digit images with labels | Data loading verification |
| `task1_class_distribution.png` | Count of training examples per digit | Dataset analysis |
| `task1_model_comparison.png` | Loss & accuracy curves for all 5 models | Architecture comparison |
| `task1_learning_curves.png` | Train vs Test curves per model | Overfitting analysis |
| `task1_confusion_matrix.png` | Which digits are confused | Detailed performance analysis |
| `task1_per_class_acc.png` | Per-digit accuracy | Per-class evaluation |
| `task1_predictions.png` | Correct and incorrect prediction examples | Qualitative evaluation |

### Brief Requirements Checklist — Task 1

| Requirement | Status |
|-------------|--------|
| Load and preprocess MNIST | ✓ Done |
| Train/test split | ✓ Done (60k/10k) |
| CNN designed from scratch | ✓ Done |
| Multiple architectural variations | ✓ 5 models compared |
| Vary number of layers | ✓ 2-layer vs 3-layer |
| Vary number of filters | ✓ 32/64 vs 64/128 vs 32/64/128 |
| Vary activation functions | ✓ ReLU vs ELU |
| Batch Normalisation explored | ✓ With and without |
| Accuracy and loss curves | ✓ Plotted for all models |
| Confusion matrix | ✓ Produced |
| F1 score | ✓ Computed (macro F1) |
| Best model identified | ✓ Automatically selected |
| Overfitting analysis | ✓ Train vs test curves |

---

---

# TASK 2 — What the Brief Requires vs What Was Achieved

## What the Brief Asked For

The brief required fine-tuning a pre-trained CNN for object detection on the KITTI dataset — a real-world driving dataset. Unlike Task 1 (building from scratch), this task focused on transfer learning and scientific experimentation.

Specifically, the brief required:

1. Load and preprocess the KITTI dataset (custom PyTorch Dataset class)
2. Parse the label/annotation files and visualise ground-truth bounding boxes on images
3. Select and justify a pre-trained backbone
4. Replace/adapt the model head for the target detection classes
5. Document which layers are frozen and which are fine-tuned
6. Train the model and log loss curves across epochs
7. Evaluate using:
   - IoU (Intersection over Union)
   - mAP (mean Average Precision) per class — Car, Pedestrian, Cyclist
8. Produce visual outputs comparing predictions vs ground truth on unseen images
9. Use at least 5 sequences with at least 3 object classes (Car, Pedestrian, Cyclist)
10. Split data at sequence level (not frame level) to avoid data leakage
11. Explore temporal aspects of the sequential KITTI frames (encouraged, Hint 4)
12. Justify all design choices with academic literature

---

## What Was Achieved in Task 2

### Dataset Loading and Preprocessing

**Custom data pipeline built entirely from scratch:**

1. **Calibration loading** — Parsed `calib_velo_to_cam.txt` and `calib_cam_to_cam.txt` to extract transformation matrices (R, T, R_rect, P_rect)
2. **3D-to-2D projection** — Implemented the full Velodyne LiDAR → Camera 0 → Rectification → Image plane projection pipeline
3. **Tracklet XML parsing** — Parsed KITTI's `tracklet_labels.xml` to extract per-frame 3D bounding boxes (position, size, rotation) for all tracked objects
4. **3D box corners** — Computed 8 corners of each 3D bounding box in Velodyne space
5. **2D box generation** — Projected all 8 corners to image coordinates, computed axis-aligned 2D bounding rectangles
6. **Custom KITTIDataset class** — Full PyTorch Dataset returning image tensors and annotation dicts (`boxes`, `labels`) compatible with Faster R-CNN

### Data Split (Sequence-Level — No Leakage)

The brief explicitly warned against frame-level splitting. Sequence-level splitting was correctly applied:

| Role | Sequences Used | Frames |
|------|---------------|--------|
| **Training** | VideoThree, VideoFive, VideoSeven, Video11, Video12 | ~717 (subsampled) |
| **Validation** | Video16 | 192 (subsampled) |
| **Test** | Video15 | 78 (all frames) |

**Why these sequences:** Selected to ensure all 3 required classes (Car, Pedestrian, Cyclist) appear across training data, with adequate variation in scenes and lighting.

### Data Statistics Computed

Full object count analysis across all sequences:

| Sequence | Role | Frames | Cars | Pedestrians | Cyclists | Total |
|----------|------|--------|------|-------------|----------|-------|
| VideoThree | TRAIN | 154 | 246 | 22 | 154 | 422 |
| VideoFive | TRAIN | 233 | 1033 | 180 | 75 | 1288 |
| VideoSeven | TRAIN | 314 | 822 | 65 | 60 | 947 |
| Video11 | TRAIN | 438 | 1186 | 32 | 40 | 1258 |
| Video12 | TRAIN | 294 | 604 | 31 | 14 | 649 |
| Video16 | VAL | 383 | 2536 | 143 | 79 | 2758 |
| Video15 | TEST | 78 | 144 | 64 | 41 | 249 |
| **TOTAL** | | | **6571** | **537** | **463** | **7571** |

### Pre-Trained Backbone Selected and Justified

**Model chosen: Faster R-CNN with MobileNetV3-Large + FPN backbone**

Justification for this choice:
- Pre-trained on COCO dataset (80 object classes, includes cars and people)
- MobileNetV3 is computationally efficient — practical for CPU-only training
- FPN (Feature Pyramid Network) improves multi-scale detection
- COCO pre-training already contains similar categories (car, person) to KITTI, making fine-tuning effective
- The detection head was replaced to output exactly 4 classes: Background(0), Car(1), Pedestrian(2), Cyclist(3)

### Three Experiments Conducted

#### Experiment 1 — Frozen Backbone (Head-Only Fine-Tuning)
- Entire MobileNetV3 body frozen
- Only FPN, RPN, and detection head trained
- Learning rate: 0.005
- Epochs: 5
- **Trainable params: ~16M / 18.9M total**
- Saved as: `kitti_exp1_frozen.pth`

**Design rationale:** Test whether COCO-pre-trained features alone are sufficient for KITTI — fastest experiment, establishes baseline detection capability.

#### Experiment 2 — Full Fine-Tuning (All Layers)
- Entire model unfrozen — all layers trained
- Lower learning rate: 0.001 (to avoid destroying pre-trained features)
- Epochs: 5
- SGD with momentum 0.9, weight decay 5e-4
- StepLR scheduler (step=3, gamma=0.5)
- Saved as: `kitti_exp2_fullft.pth`

**Design rationale:** Allows backbone to adapt to KITTI's specific appearance (road scenes, German city, particular camera lens) for potentially higher accuracy.

#### Experiment 3 (Bonus) — Temporal Feature Stacking
- Two consecutive frames stacked as a 6-channel input (RGB_t + RGB_{t-1})
- First conv layer modified from 3→6 input channels
- Pre-trained 3-channel weights duplicated and scaled ×0.5 for initialisation
- Image normalisation updated to handle 6-channel input
- Learning rate: 0.002
- Saved as: `kitti_exp3_temporal.pth`

**Design rationale:** KITTI frames are sequential video. Stacking consecutive frames gives the model implicit motion information — moving objects will show displacement between the two frames, helping distinguish them from static background. This directly addresses Hint 4 of the brief (temporal modelling).

### Evaluation Metrics Implemented

Both metrics required by the brief were fully implemented:

**IoU (Intersection over Union):**
- Computed between every predicted box and every ground-truth box of the same class
- IoU ≥ 0.5 threshold used (PASCAL VOC convention) to classify a prediction as True Positive
- Mean IoU reported across all True Positives

**mAP (mean Average Precision):**
- AP computed per class using 11-point interpolation (PASCAL VOC 2007 protocol)
- Predictions sorted by confidence score before computing precision-recall curve
- mAP = average of Car AP, Pedestrian AP, Cyclist AP

### Visual Outputs Produced

| Output Image | What It Shows | Brief Requirement Met |
|-------------|---------------|----------------------|
| `Training GT Boxes.png` | 4 training images with correct bounding boxes | Data loader verification |
| `Test GT Boxes.png` | 4 test images with correct bounding boxes | Data loader verification |
| `task2_training_curves.png` | Loss curves across 5 epochs for Exp1 & Exp2 | Training process logging |
| `task2_map_comparison.png` | Per-class AP bar chart: Exp1 vs Exp2 | Performance comparison |
| `task2_predictions_exp1.png` | GT vs predicted boxes on test images (Exp1) | Visual predictions vs GT |
| `task2_predictions_exp2.png` | GT vs predicted boxes on test images (Exp2) | Visual predictions vs GT |
| `task2_predictions_exp3.png` | GT vs predicted boxes on test images (Exp3) | Temporal model results |
| `task2_all_val_loss.png` | All 3 experiments' validation loss comparison | Final experiment comparison |

### Brief Requirements Checklist — Task 2

| Requirement | Status |
|-------------|--------|
| Custom PyTorch Dataset class for KITTI | ✓ Done — `KITTIDataset` |
| Parse label/annotation files | ✓ Done — full XML tracklet parser |
| Visualise ground-truth boxes on images | ✓ Done — before any training |
| Pre-trained backbone selected and justified | ✓ Done — Faster R-CNN / MobileNetV3 |
| Model head replaced for target classes | ✓ Done — 4-class predictor |
| Document frozen vs fine-tuned layers | ✓ Done — both experiments |
| Train and log loss curves | ✓ Done — per epoch |
| IoU computed | ✓ Done — mean IoU reported |
| mAP computed per class | ✓ Done — Car, Pedestrian, Cyclist |
| Visual: predictions vs ground truth | ✓ Done — all 3 experiments |
| Minimum 5 sequences used | ✓ Done — 7 sequences total |
| At least 3 object classes | ✓ Done — Car, Pedestrian, Cyclist |
| Sequence-level data split (no leakage) | ✓ Done — explained and justified |
| Temporal modelling explored | ✓ Done — Experiment 3 (6-channel stacking) |

---

---

# Deliverables Summary

According to the brief, three items must be submitted:

## 1. Technical Report (PDF, ~3000 words)

Must include these sections:

| Section | What to Write |
|---------|--------------|
| **Cover Sheet** | Full name, student ID, word count |
| **Introduction** | Background, motivation, scope of both tasks |
| **Literature Review** | Key research papers on CNNs, object detection, KITTI, transfer learning |
| **Methodology** | How each task was approached, design decisions with rationale |
| **Results and Discussion** | Present all results, interpret findings, justify best model |
| **Conclusion and Future Works** | What was concluded, what could be done with more time |
| **References** | All cited sources in Harvard style |
| **Appendix** | Full Python code for both tasks (unbroken, plain text) |

**Important:** Code must appear in the appendix as a single unbroken plain-text entry — failure to include this results in a 10-mark deduction.

## 2. Python Code

Submit via Canvas as `.ipynb`, `.py`, or `.zip`:
- `task1_mnist_cnn.ipynb` (or `.py`)
- `task2_kitti_detection.ipynb` (or `.py`)

## 3. Video Presentation (max 15 minutes, ~1080p, ~500MB, .mp4)

| Section | Time | What to Cover |
|---------|------|--------------|
| **Introduction** | ~2 min | Introduce yourself, high-level overview of both tasks |
| **Task 1** | ~4 min | MNIST loading, CNN architectures, results, learning curves, best model |
| **Task 2** | ~6 min | KITTI loading, backbone choice, training process, loss curves, bounding box visuals, temporal modelling |
| **Conclusion & Future Work** | ~3 min | Key findings, what could be improved, why it matters |

**Critical notes for video:**
- Upload the actual `.mp4` file to Canvas — links will NOT be accepted
- File must be fully uploaded before the deadline
- Test a short clip first to verify resolution and file size

---

---

# Key Academic Points to Address in the Report

Based on the assignment brief, these are the specific arguments you should make:

## For Task 1:

**Why Batch Normalisation helps:**
Batch Norm normalises layer inputs, reducing internal covariate shift, which stabilises and accelerates training (Ioffe and Szegedy, 2015). This explains why Model D outperforms Models A–C.

**Why ELU may outperform ReLU:**
ELU (Exponential Linear Unit) produces smooth negative outputs (unlike ReLU which zeros them), reducing the "dying ReLU" problem and improving gradient flow (Clevert et al., 2016).

**Why deeper networks help:**
More convolutional layers build hierarchical feature representations — early layers detect edges, deeper layers detect shapes and patterns (LeCun et al., 1998).

**How to justify best model:**
Compare final test accuracy, macro F1 score, and learning curve stability across all 5 models. The best model is the one with highest accuracy AND stable training (low gap between train and test curves).

## For Task 2:

**Why MobileNetV3 was chosen over ResNet-50:**
MobileNetV3 is approximately 3× faster with only a small accuracy trade-off, making it practical for CPU-based fine-tuning experiments (Howard et al., 2019).

**Why sequence-level splitting matters:**
Consecutive frames share near-identical visual content. Frame-level random splits would allow the model to effectively "memorise" scenes from the test set via similar training frames, inflating performance metrics (Geiger et al., 2012).

**Why temporal stacking helps:**
Static per-frame models cannot capture motion. Stacking consecutive frames gives implicit optical flow information — displacement of objects between frames helps distinguish moving objects (cars, pedestrians) from static background (Simonyan and Zisserman, 2014 — Two-Stream Networks concept).

**Why freeze vs full fine-tuning matters:**
Frozen backbone preserves COCO-learned features and trains faster. Full fine-tuning adapts features to KITTI's specific visual domain (German roads, specific camera optics, weather conditions) but risks overfitting on limited data and requires careful learning rate selection (Yosinski et al., 2014).

---

---

# Summary: What This Project Achieved

| Goal | Achievement |
|------|-------------|
| CNN from scratch on MNIST | 5 architectures built, trained, and systematically compared |
| Evidence-based architecture search | Progression from baseline → filters → depth → BN → ELU justified at each step |
| Best MNIST model identified | Automatically selected by highest test accuracy with full evaluation |
| Real-world object detection | Faster R-CNN fine-tuned on KITTI dashcam data |
| Full annotation pipeline | 3D LiDAR → 2D camera projection implemented from scratch |
| Multiple fine-tuning strategies | 3 experiments: frozen backbone, full fine-tuning, temporal stacking |
| Detection metrics | IoU and mAP per class computed on held-out test sequence |
| Temporal modelling | 6-channel frame stacking implemented as bonus experiment |
| No data leakage | Strict sequence-level split enforced throughout |
| All visual requirements | Ground-truth visualisation, prediction vs GT comparison, loss curves, mAP charts |

---

*CET3013 Deep Learning — Assessment 2 | University of Sunderland*
