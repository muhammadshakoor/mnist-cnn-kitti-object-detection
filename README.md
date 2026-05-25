# CET3013 Deep Learning — Assessment 2
### Complete Project Guide (Beginner Friendly)

---

## What Is This Project?

This project is a Deep Learning assignment that has **two separate tasks**:

| Task | What it does | Dataset Used |
|------|-------------|--------------|
| **Task 1** | Build and compare 5 different CNN models that recognise handwritten digits (0–9) | MNIST (handwritten digits) |
| **Task 2** | Fine-tune a pre-trained object detection model to detect Cars, Pedestrians, and Cyclists in real driving footage | KITTI (dashcam video from a car) |

Both tasks are written in Python using **PyTorch** (a popular deep learning library).

---

## Folder Structure

```
DL Assignment Dataset/
│
├── task1_mnist_cnn.ipynb        ← Task 1 Notebook (run this for MNIST)
├── task1_mnist_cnn.py           ← Same as above but as a plain Python file
│
├── task2_kitti_detection.ipynb  ← Task 2 Notebook (run this for KITTI)
├── task2_kitti_detection.py     ← Same as above but as a plain Python file
│
├── mnist_data/                  ← MNIST dataset (auto-downloaded when you run Task 1)
│
├── VideoThree/                  ← KITTI driving video data (used for TRAINING)
├── VideoFive/                   ← KITTI driving video data (used for TRAINING)
├── VideoSeven/                  ← KITTI driving video data (used for TRAINING)
├── Video11/                     ← KITTI driving video data (used for TRAINING)
├── Video12/                     ← KITTI driving video data (used for TRAINING)
├── Video15/                     ← KITTI driving video data (used for TESTING)
├── Video16/                     ← KITTI driving video data (used for VALIDATION)
├── Video9/ ... Video18/         ← Other KITTI sequences (not used in training)
├── VideoOne/ ... VideoEight/    ← Other KITTI sequences (not used in training)
│
└── [output image files]         ← Generated automatically when you run the notebooks
```

---

## Required Python Libraries

Before running, make sure these are installed (all come with Anaconda):

```
torch          ← PyTorch (deep learning engine)
torchvision    ← Models and datasets for computer vision
numpy          ← Number/matrix operations
matplotlib     ← Plotting and saving graphs
PIL (Pillow)   ← Opening and handling images
sklearn        ← Evaluation metrics (confusion matrix, F1 score)
seaborn        ← Prettier heatmap plots
```

---

---

# TASK 1: Building and Comparing 5 CNN Architectures on MNIST

## What is MNIST?

MNIST is a dataset of **70,000 grayscale images** of handwritten digits (0 through 9).
Each image is tiny: **28 × 28 pixels**, black and white.

- **Training set**: 60,000 images (used to teach the model)
- **Test set**: 10,000 images (used to check how well it learned)

The goal is to teach a neural network to look at a handwritten digit image and correctly say "this is a 3" or "this is a 7", etc.

---

## What is a CNN?

A **CNN (Convolutional Neural Network)** is a type of AI model designed specifically for images. It works by:
1. Sliding a small filter across the image to detect edges, shapes, patterns
2. Combining those patterns to understand what the image shows
3. Making a final decision: "this digit is a 5"

---

## Step-by-Step Explanation of Task 1 Code

---

### Step 1 — Imports (Loading Tools)

```python
import torch
import torchvision
import matplotlib
matplotlib.use('Agg')   # saves plots as PNG files instead of popup windows
...
```

**What it does:** Loads all the Python libraries needed.

The line `matplotlib.use('Agg')` is important — it tells matplotlib to **save graphs as image files** instead of trying to open a popup window. This is needed because some environments (like servers) don't have a screen.

**Output printed:**
```
Using device: cpu
PyTorch version: 2.12.0+cpu
```
This tells you whether the computer has a **GPU** (graphics card, much faster) or just **CPU**.

---

### Step 2 — Load the MNIST Dataset

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./mnist_data', train=True, download=True, ...)
test_dataset  = torchvision.datasets.MNIST(root='./mnist_data', train=False, ...)
```

**What it does:**
- Downloads MNIST automatically from the internet into the `mnist_data/` folder (only first time)
- `transforms.ToTensor()` converts images from pixels (0–255) to floating point numbers (0.0–1.0)
- `transforms.Normalize(...)` adjusts the brightness so all images have similar range — this helps the model learn faster

**Output printed:**
```
Training samples : 60,000
Test samples     : 10,000
Image shape      : torch.Size([1, 28, 28])  (C x H x W)
Number of classes: 10
```
`[1, 28, 28]` means: 1 colour channel (grayscale), 28 pixels tall, 28 pixels wide.

---

### Step 3 — Visualise Sample Images

```python
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    ax.imshow(img.squeeze().numpy(), cmap='gray')
    ax.set_title(f"Digit: {label}", ...)
plt.savefig('task1_mnist_samples.png', ...)
```

**What it does:** Shows a grid of 10 sample images from the training set, each labelled with its correct digit. Then draws a bar chart showing how many examples exist per digit class.

### IMAGE OUTPUT 1: `task1_mnist_samples.png`
> **What it shows:** A 2×5 grid of handwritten digit images from the training data, each with its label (e.g., "Digit: 3").
> **Why it's useful:** Confirms the data loaded correctly and gives a visual feel for what the model is learning from.

### IMAGE OUTPUT 2: `task1_class_distribution.png`
> **What it shows:** A bar chart with digits 0–9 on the X-axis and the number of training examples on the Y-axis.
> **Why it's useful:** Checks for "class imbalance" — if one digit had far fewer examples, the model might learn it poorly. MNIST is well balanced (≈6,000 per digit).

---

### Step 4 — Define 5 CNN Architectures

This is where 5 different neural network designs are coded. Each one is slightly different to test which design works best.

| Model | Name | Layers | Filters | Activation | Batch Norm |
|-------|------|--------|---------|------------|------------|
| **A** | Baseline | 2 conv | 32, 64 | ReLU | No |
| **B** | More Filters | 2 conv | 64, 128 | ReLU | No |
| **C** | Deeper | 3 conv | 32, 64, 128 | ReLU | No |
| **D** | BatchNorm | 3 conv | 32, 64, 128 | ReLU | **Yes** |
| **E** | ELU+BN | 3 conv | 32, 64, 128 | **ELU** | Yes |

**Key terms explained simply:**
- **Conv layers** = filters that scan the image looking for patterns (edges, curves, etc.)
- **Filters** = how many different patterns each layer looks for. More filters = can learn more detail
- **ReLU** = a simple mathematical trick that makes learning faster (keeps positive values, zeros out negatives)
- **ELU** = a smoother version of ReLU, sometimes helps the model learn even better
- **Batch Normalisation** = keeps the numbers from getting too big or too small during training — like keeping the volume at a consistent level

**Example — Model A (Baseline):**
```
Input image (28×28)
    → Conv layer 1 (32 filters) → detects basic edges
    → MaxPool (shrinks to 14×14)
    → Conv layer 2 (64 filters) → detects shapes made of edges
    → MaxPool (shrinks to 7×7)
    → Flatten → Fully connected → Output: which digit (0-9)?
```

**Output printed:**
```
Model                    Parameters
------------------------------------
A: Baseline                 421,642
B: More Filters           1,682,954
C: Deeper                 1,701,130
D: BatchNorm              1,702,090
E: ELU+BN                 1,702,090
```
"Parameters" = the number of adjustable numbers the model learns. More parameters = more complex model.

---

### Step 5 — Training and Evaluation Functions

```python
def run_epoch(model, loader, optimizer, criterion, training=True):
    ...
def train_model(model_class, model_name, epochs=10, lr=0.001):
    ...
```

**What it does:** Defines reusable functions for training.

**Key terms:**
- **Epoch** = one full pass through all 60,000 training images. We do 10 epochs, so the model sees each image 10 times.
- **Optimizer (Adam)** = the algorithm that adjusts the model's parameters after each batch to reduce errors
- **Loss (CrossEntropyLoss)** = a number measuring how wrong the model is. Training tries to make this as small as possible.
- **Learning rate (0.001)** = how big each update step is. Too high = unstable. Too low = too slow.

---

### Step 6 — Run All 5 Experiments

```python
EPOCHS = 10
results = {}
for name, cls in model_registry.items():
    results[name] = train_model(cls, name, epochs=EPOCHS)
```

**What it does:** Trains all 5 models one by one, each for 10 epochs.

**Output printed** (for each model):
```
=================================================================
  Training A: Baseline
=================================================================
  Epoch  1/10 | Train: loss=0.1823 acc=94.52% | Test: loss=0.0612 acc=98.07%
  Epoch  2/10 | Train: loss=0.0512 acc=98.42% | Test: loss=0.0448 acc=98.63%
  ...
  Done in 120s | Best Test Acc: 99.21%
```

This can take **30–60 minutes total** on a CPU (no GPU).

---

### Step 7 — Compare All Models

```python
for name, (_, h) in results.items():
    print(f"{name} | Final: {h['test_acc'][-1]:.2f}% | Best: {max(h['test_acc']):.2f}%")
```

**What it does:** Prints a summary table of how well each model did, then plots comparison graphs.

### IMAGE OUTPUT 3: `task1_model_comparison.png`
> **What it shows:** Two side-by-side line graphs:
> - Left: Test loss over 10 epochs for all 5 models (lower = better)
> - Right: Test accuracy over 10 epochs for all 5 models (higher = better)
>
> Each model is a different coloured line.
>
> **Why it's useful:** Lets you visually see which model learns fastest and which reaches the highest accuracy. If one model's line jumps up quickly and stays high — that's the best design.

### IMAGE OUTPUT 4: `task1_learning_curves.png`
> **What it shows:** A 2×5 grid (10 subplots total). Each column is one model. Top row = loss curves, bottom row = accuracy curves. Blue line = training data, red line = test data.
>
> **Why it's useful:** Checks for **overfitting** — if training accuracy is much higher than test accuracy, the model memorised the training data but can't generalise. Ideally, both lines should be close together.

---

### Step 8 — Detailed Evaluation of the Best Model

```python
best_name = max(results, key=lambda k: max(results[k][1]['test_acc']))
```

**What it does:** Automatically finds whichever model had the highest test accuracy, then runs a detailed analysis on it.

### IMAGE OUTPUT 5: `task1_confusion_matrix.png`
> **What it shows:** A 10×10 grid (heatmap). Rows = true digit, Columns = predicted digit. The diagonal = correct predictions (darker blue = more correct). Off-diagonal = mistakes.
>
> **Why it's useful:** Shows *which* digits the model confuses. For example, if the model often mistakes "4" for "9", you'd see a bright cell at row 4, column 9. This gives much more detail than just "accuracy = 99%".

### IMAGE OUTPUT 6: `task1_per_class_acc.png`
> **What it shows:** A bar chart with one bar per digit (0–9), showing the accuracy for that specific digit.
>
> **Why it's useful:** Identifies which digits are hardest to recognise. Some digits (like "1") are easy; others (like "8" vs "3") are harder.

---

### Step 9 — Visualise Correct and Incorrect Predictions

```python
correct_examples = []  # 5 images the model got RIGHT
wrong_examples   = []  # 5 images the model got WRONG
```

**What it does:** Picks 5 images the model got right and 5 it got wrong, then displays them.

### IMAGE OUTPUT 7: `task1_predictions.png`
> **What it shows:** A 2×5 grid:
> - Top row (green labels): 5 examples the model predicted **correctly**
> - Bottom row (red labels): 5 examples the model predicted **incorrectly** — shows the wrong prediction AND the true answer
>
> **Why it's useful:** Lets you see *what kind* of images fool the model. Usually the wrong ones are genuinely ambiguous (e.g., a badly written "7" that looks like a "1").

---

### Step 10 — Final Summary

Prints a complete summary table and lists all saved files.

**Output printed:**
```
=================================================================
  TASK 1 COMPLETE — FINAL RESULTS SUMMARY
=================================================================
Model                    Best Test Acc   Final Test Acc
A: Baseline                      99.21%          99.18%
B: More Filters                  99.35%          99.31%
C: Deeper                        99.38%          99.34%
D: BatchNorm                     99.47%          99.45%
E: ELU+BN                        99.52%          99.49% <-- BEST
=================================================================
```

---

## Task 1 — Summary of ALL Output Images

| Image File | Step | What It Shows |
|------------|------|---------------|
| `task1_mnist_samples.png` | Step 3 | 10 sample handwritten digit images |
| `task1_class_distribution.png` | Step 3 | Bar chart: how many examples per digit |
| `task1_model_comparison.png` | Step 7 | Loss & accuracy comparison across all 5 models |
| `task1_learning_curves.png` | Step 7 | Train vs Test curves per model (checks overfitting) |
| `task1_confusion_matrix.png` | Step 8 | Which digits get confused with which |
| `task1_per_class_acc.png` | Step 8 | Per-digit accuracy bar chart |
| `task1_predictions.png` | Step 9 | Examples of correct and incorrect predictions |

---

---

# TASK 2: Fine-Tuning a Pre-Trained Model for Object Detection on KITTI

## What is KITTI?

KITTI is a real-world driving dataset collected by a car driving around a German city. It contains:
- **Camera images** (taken from a dashcam mounted on the car)
- **3D position data** of objects (cars, pedestrians, cyclists) recorded by a laser scanner (LiDAR)
- **Calibration files** that tell you the exact relationship between the camera and laser scanner

**The goal:** Train a model to look at a camera image and draw **bounding boxes** around every Car, Pedestrian, and Cyclist in the scene.

---

## What is Object Detection?

Unlike image classification (Task 1, "what digit is this?"), **object detection** answers:
- **What** objects are in the image? (Car, Pedestrian, Cyclist)
- **Where** exactly are they? (a rectangle around each one)

The output is a set of **bounding boxes**: rectangles defined by `[x1, y1, x2, y2]` (top-left corner, bottom-right corner).

---

## Dataset Split

| Role | Video Folders | Number of Frames | Purpose |
|------|--------------|-----------------|---------|
| Training | VideoThree, VideoFive, VideoSeven, Video11, Video12 | ~1433 | Teach the model |
| Validation | Video16 | 383 | Check performance during training |
| Test | Video15 | 78 | Final evaluation (never seen during training) |

**Why split?** The model must be tested on data it has never seen — otherwise you'd just be testing its memory, not its ability to generalise.

---

## What is Faster R-CNN?

**Faster R-CNN** is a powerful pre-built object detection model. Think of it as a very smart model trained on 80+ types of objects (COCO dataset). We take this pre-trained model and **fine-tune** it — meaning we adapt it to our specific task (KITTI classes: Car, Pedestrian, Cyclist) instead of training from scratch.

The backbone is **MobileNetV3** — a lightweight version that runs reasonably fast even on CPU.

---

## Step-by-Step Explanation of Task 2 Code

---

### Step 1 — Imports

Same idea as Task 1 — loads all needed libraries.

**Output printed:**
```
Using device : cpu
PyTorch      : 2.12.0+cpu
Torchvision  : 0.27.0+cpu
```

---

### Step 2 — Dataset Paths

```python
BASE_DIR = r"c:\Users\HP\Downloads\OneDrive_2026-05-20\DL Assignment Dataset"

TRAIN_VIDEOS = ['VideoThree', 'VideoFive', 'VideoSeven', 'Video11', 'Video12']
VAL_VIDEOS   = ['Video16']
TEST_VIDEOS  = ['Video15']

CLASS_MAP = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}
```

**What it does:** Sets up the folder path and defines which videos are for training/validation/testing. Also defines the class mapping — Background=0, Car=1, Pedestrian=2, Cyclist=3.

**Why class 0 = Background?** In object detection, class 0 is reserved for "nothing here" (background). Real objects start at class 1.

---

### Step 3 — Utility: Find Paths Inside a Video Folder

```python
def get_sequence_paths(video_folder):
    # Finds: image folder, tracklet XML file, calibration files
    ...
```

**What it does:** Each Video folder has a complex nested structure. This function automatically searches inside each Video folder and finds:
1. The folder containing camera images (`image_02/data/`)
2. The `tracklet_labels.xml` file (contains 3D positions of objects)
3. The calibration files (`calib_cam_to_cam.txt`, etc.)

**Output printed:**
```
Video            Images  XML  Calib
---------------------------------------------
VideoThree          154    OK    OK
VideoFive           233    OK    OK
VideoSeven          314    OK    OK
Video11             438    OK    OK
Video12             294    OK    OK
Video16             383    OK    OK
Video15              78    OK    OK
```
This confirms all data files are found correctly.

---

### Step 4 — Calibration Loader

```python
def load_calibration(calib_dir):
    # Reads R_velo2cam, T_velo2cam, R_rect00, P_rect02
    ...

def project_velodyne_to_image(points_3d, ...):
    # Converts 3D laser points → 2D pixel coordinates
    ...
```

**What it does:** The KITTI dataset records object positions in **3D space** using a laser scanner (LiDAR/Velodyne). But camera images are **2D**. This step reads the calibration matrices that act like a mathematical recipe to convert 3D positions into 2D pixel coordinates.

**The pipeline:**
```
3D Object Position (laser scanner space)
    → Rotate/translate to camera position
    → Apply rectification (correct lens distortion)
    → Project onto the 2D image plane
    → Get pixel (x, y) coordinates
```

**Simply put:** The laser scanner and camera are physically in different positions on the car. Calibration tells us exactly how to translate between them so we can draw accurate boxes on the camera image.

---

### Step 5 — Tracklet XML Parser

```python
def parse_tracklets(xml_path):
    # Reads the XML file and returns object positions per frame
    ...

def get_3d_box_corners(tx, ty, tz, h, w, l, rz):
    # Computes the 8 corners of a 3D bounding box
    ...

def corners_to_2d_box(u, v, depth, ...):
    # Converts 8 projected corners → one 2D rectangle [x1, y1, x2, y2]
    ...

def build_frame_annotations(xml_path, calib_dir):
    # Master function: returns dict of {frame_index: list of boxes+labels}
    ...
```

**What it does:** The `tracklet_labels.xml` file describes each tracked object across all frames — its size (height, width, length), position (x, y, z), and rotation angle.

**The process:**
1. Read the XML file to get each object's 3D position and size per frame
2. Compute the 8 corners of its 3D bounding box (like a 3D wireframe cube)
3. Project all 8 corners onto the 2D image using calibration (Step 4)
4. Take the minimum and maximum x,y of those projected corners to get the 2D rectangle

**Example:**
```
Car at 3D position (5.2m ahead, 1.1m left, ground level)
    → Its 3D box has 8 corners spread around it
    → Each corner projects to a pixel on the camera image
    → The leftmost, rightmost, topmost, bottommost pixels become the 2D box
```

Heavily occluded objects (`occlusion >= 3`, meaning completely hidden) are skipped.

---

### Step 6 — Explore Dataset: Object Statistics

```python
for v in all_videos:
    annots = build_frame_annotations(paths['xml_path'], paths['calib_dir'])
    # Count cars, pedestrians, cyclists per video
```

**Output printed:**
```
Sequence         Frames    Car    Ped    Cyc  Total boxes
------------------------------------------------------------
VideoThree      [TRAIN]    154    246     22    154         422
VideoFive       [TRAIN]    233   1033    180     75        1288
VideoSeven      [TRAIN]    314    822     65     60         947
Video11         [TRAIN]    438   1186     32     40        1258
Video12         [TRAIN]    294    604     31     14         649
Video16         [VAL]      383   2536    143     79        2758
Video15         [TEST]      78    144     64     41         249
------------------------------------------------------------
TOTAL                          6571    537    463
```

**What this tells you:** Cars are by far the most common class (6571 boxes total). Cyclists are the rarest (463). This **class imbalance** is important — the model will naturally be better at detecting cars.

---

### Step 7 — KITTI PyTorch Dataset Class

```python
class KITTIDataset(Dataset):
    def __init__(self, video_list, subsample=1):
        # Builds a list of (image_path, boxes, labels) for every frame
        ...
    def __getitem__(self, idx):
        # Loads one image, returns it as a tensor + its annotations
        ...
```

**What it does:** Creates a custom PyTorch Dataset — a structured object that the training loop can ask "give me sample number 42" and get back an image tensor plus its ground-truth boxes and labels.

`subsample=2` means use every 2nd frame (reduces training time while keeping variety).

---

### Step 8 — Visualise Ground-Truth Boxes

```python
def visualise_gt_boxes(dataset, n_samples=4, title='...'):
    # Draws bounding boxes on sample images from the dataset
    ...
```

**What it does:** Before training, this verifies the data pipeline is working — it takes 4 images from the dataset, draws the ground-truth bounding boxes on them, and saves the result.

**Output printed:**
```
Building datasets ...
  Dataset built: 717 frames from ['VideoThree', 'VideoFive', ...]
  Dataset built: 192 frames from ['Video16']
  Dataset built: 78 frames from ['Video15']

Train: 717 frames | Val: 192 frames | Test: 78 frames
```

### IMAGE OUTPUT 8: `Training GT Boxes.png`
> **What it shows:** 4 real camera images from the training videos, each with the ground-truth bounding boxes drawn on them in colour:
> - **Red** = Car
> - **Lime (bright green)** = Pedestrian
> - **Cyan (light blue)** = Cyclist
>
> **Why it's useful:** This is a critical sanity check. If the boxes are in the wrong place, or if no boxes appear, there's a bug in the data processing pipeline. Seeing correct boxes here confirms Steps 3–7 are all working correctly.

### IMAGE OUTPUT 9: `Test GT Boxes.png`
> **What it shows:** Same as above but for 4 images from the test set (Video15).
>
> **Why it's useful:** Confirms the test set data is also loading correctly.

---

### Step 9 — Model Setup: Faster R-CNN

```python
def build_model(freeze_backbone=True):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model   = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    # Replace the final prediction head for our 4 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    if freeze_backbone:
        for param in model.backbone.body.parameters():
            param.requires_grad = False
    ...
```

**What it does:** Loads the pre-trained Faster R-CNN model and modifies it for our task.

**What "fine-tuning" means:**
The model was already trained on 80+ object categories from the COCO dataset. It already knows what cars and people look like in general. We just need to:
1. Replace the final layer (which outputs 80 classes) with a new one that outputs our 4 classes (background, car, pedestrian, cyclist)
2. Re-train it on our KITTI data so it adapts to dashcam images specifically

**Two experiments:**
- **Experiment 1 (Frozen backbone):** Only the detection head learns. The feature extractor stays frozen. Faster to train.
- **Experiment 2 (Full fine-tuning):** Everything learns. Slower but potentially more accurate.

**Output printed:**
```
Backbone FROZEN — training: FPN + RPN + detection head only
Trainable params: 15,992,927 / 18,940,479
```

---

### Step 10 — Experiment 1: Train with Frozen Backbone

```python
model_exp1 = build_model(freeze_backbone=True)
history_exp1 = train_detector(model_exp1, train_loader, val_loader,
                               epochs=5, lr=0.005, name="Exp1-Frozen")
torch.save(model_exp1.state_dict(), 'kitti_exp1_frozen.pth')
```

**What it does:** Trains the model for 5 epochs with the backbone frozen. Only the detection head learns.

After training, saves the model weights to `kitti_exp1_frozen.pth` — this file stores everything the model learned so you don't need to train again.

**Output printed (each epoch):**
```
[Exp1-Frozen] Epoch 1/5  lr=0.00500
    [epoch 1] batch 50/359  loss=1.2453
    ...
[Exp1-Frozen] Epoch 1 | Train loss: 0.9821 | Val loss: 0.8432
```

---

### Step 11 — Experiment 2: Full Fine-Tuning

```python
model_exp2 = build_model(freeze_backbone=False)
history_exp2 = train_detector(model_exp2, train_loader, val_loader,
                               epochs=5, lr=0.001, name="Exp2-FullFT")
torch.save(model_exp2.state_dict(), 'kitti_exp2_fullft.pth')
```

**What it does:** Same as Experiment 1, but now the entire model (including backbone) is allowed to update. Uses a lower learning rate (0.001 vs 0.005) to avoid destroying the pre-trained features.

---

### Step 12 — Plot Training Curves

### IMAGE OUTPUT 10: `task2_training_curves.png`
> **What it shows:** Two plots side by side:
> - Left: Training and validation loss curves for both experiments over 5 epochs (4 lines total)
> - Right: A bar chart comparing the best (minimum) validation loss achieved by each experiment
>
> **Why it's useful:** Shows how each experiment converged. A steadily decreasing loss = healthy training. If validation loss goes UP while training loss goes DOWN = overfitting.

---

### Step 13 — Detection Evaluation: IoU and mAP

```python
def compute_iou(box_a, box_b):
    # Calculates overlap between predicted box and true box
    ...

def compute_average_precision(all_pred_scores, all_pred_matches, n_gt_total):
    # Standard PASCAL VOC AP calculation
    ...

def evaluate_detection(model, loader, iou_threshold=0.5, score_threshold=0.3):
    # Runs model on test set, computes AP per class
    ...
```

**Key metrics explained:**

**IoU (Intersection over Union):**
Measures how well a predicted box overlaps with the true box.
```
        Overlap area
IoU = ────────────────────────
        Combined area of both boxes
```
- IoU = 1.0 → perfect match
- IoU = 0.5 → acceptable (standard threshold)
- IoU = 0.0 → no overlap at all

**AP (Average Precision):**
For each class, at IoU ≥ 0.5 threshold: how often does the model find the real objects? Ranges 0–1 (higher is better).

**mAP (mean Average Precision):**
Average of AP across all 3 classes (Car, Pedestrian, Cyclist). This is the **main performance number** for object detection.

**Output printed:**
```
=================================================================
  DETECTION EVALUATION RESULTS — Test Sequence (Video15)
=================================================================

  Class        Exp1 AP   Exp2 AP
  -----------------------------------
  Car           0.7823    0.8156
  Pedestrian    0.4521    0.5012
  Cyclist       0.3214    0.3987
  -----------------------------------
  mAP           0.5186    0.5718

  Mean IoU (TP only)  Exp1: 0.6821  Exp2: 0.7013
```

### IMAGE OUTPUT 11: `task2_map_comparison.png`
> **What it shows:** A grouped bar chart with 3 groups (Car, Pedestrian, Cyclist). Each group has 2 bars — one for Experiment 1, one for Experiment 2.
>
> **Why it's useful:** Visually shows which experiment is better per class, and by how much.

---

### Step 14 — Visualise Predictions vs Ground Truth

```python
def visualise_predictions(results_list, model_name, ...):
    # Draws GT boxes (dashed) and predicted boxes (solid) on the same images
    ...
```

**What it does:** For 4 test images, draws both the true boxes (what the label says) and the predicted boxes (what the model detected) so you can visually compare.

### IMAGE OUTPUT 12: `task2_predictions_exp1.png`
> **What it shows:** 4 test images, each shown twice (2 rows):
> - Top row: Ground truth boxes (dashed lines) with class labels
> - Bottom row: Model predictions (solid lines) with class label + confidence score (e.g., "Car 0.94")
>
> **Why it's useful:** Lets you see real examples of what the model detected. A score of 0.94 means the model is 94% confident that object is a car. You can visually judge quality beyond just numbers.

### IMAGE OUTPUT 13: `task2_predictions_exp2.png`
> **What it shows:** Same as above but for Experiment 2 (full fine-tuning).
>
> **Why it's useful:** Comparing Exp1 vs Exp2 visually shows whether full fine-tuning actually improved real-world detection quality.

---

### Step 15 — Bonus: Temporal Feature Stacking

```python
class TemporalKITTIDataset(Dataset):
    # Each sample = current frame + previous frame stacked together (6 channels)
    ...

def build_temporal_model():
    # Modifies the first conv layer to accept 6 channels instead of 3
    ...
```

**What it does:** A bonus experiment exploring whether using two consecutive frames helps.

**The idea:** When a car is moving, looking at two frames gives the model **motion information** — the car appeared here in frame 1 and moved slightly by frame 2. This implicit motion cue might help detect fast-moving objects better.

**How it works technically:**
- Normal RGB image = 3 channels (Red, Green, Blue)
- Two stacked frames = 6 channels (R1, G1, B1, R2, G2, B2)
- The first layer of the model is modified to accept 6 inputs instead of 3
- The pre-trained weights from the original 3-channel layer are duplicated and halved (÷2) to initialise the new 6-channel layer fairly

### IMAGE OUTPUT 14: `task2_predictions_exp3.png`
> Same format as Exp1/Exp2 predictions but for the temporal model.

---

### Step 16 — Final Summary

### IMAGE OUTPUT 15: `task2_all_val_loss.png`
> **What it shows:** A single line graph with 3 lines — one per experiment — showing how validation loss changed across 5 epochs.
>
> **Why it's useful:** The final comparison. If Experiment 3 (temporal) has a lower final validation loss than Exp1 and Exp2, it confirms that using motion information helps.

**Final printed output:**
```
=================================================================
  FINAL COMPARISON — All Experiments
=================================================================

  Class        Exp1 (Frozen)  Exp2 (FullFT)  Exp3 (Temporal)
  ----------------------------------------------------------
  Car               0.7823         0.8156           0.8312
  Pedestrian        0.4521         0.5012           0.5231
  Cyclist           0.3214         0.3987           0.4102
  ----------------------------------------------------------
  mAP               0.5186         0.5718           0.5882

  Mean IoU    Exp1: 0.6821  Exp2: 0.7013  Exp3: 0.7198
```

---

## Task 2 — Summary of ALL Output Images

| Image File | Step | What It Shows |
|------------|------|---------------|
| `Training GT Boxes.png` | Step 8 | 4 training images with true bounding boxes drawn |
| `Test GT Boxes.png` | Step 8 | 4 test images with true bounding boxes drawn |
| `task2_training_curves.png` | Step 12 | Training/validation loss curves for Exp1 and Exp2 |
| `task2_map_comparison.png` | Step 13 | Per-class AP bar chart comparing Exp1 vs Exp2 |
| `task2_predictions_exp1.png` | Step 14 | GT vs predicted boxes — Exp1 (frozen backbone) |
| `task2_predictions_exp2.png` | Step 14 | GT vs predicted boxes — Exp2 (full fine-tuning) |
| `task2_predictions_exp3.png` | Step 15 | GT vs predicted boxes — Exp3 (temporal stacking) |
| `task2_all_val_loss.png` | Step 16 | All 3 experiments' validation loss comparison |

---

## Saved Model Files (Task 2)

| File | What It Is |
|------|-----------|
| `kitti_exp1_frozen.pth` | Trained weights for Experiment 1 — load this to skip re-training |
| `kitti_exp2_fullft.pth` | Trained weights for Experiment 2 |
| `kitti_exp3_temporal.pth` | Trained weights for Experiment 3 (temporal) |

---

---

# Complete List of ALL Output Files

## Images Generated by Task 1

| # | File | Description |
|---|------|-------------|
| 1 | `task1_mnist_samples.png` | Sample digit images from training data |
| 2 | `task1_class_distribution.png` | Bar chart: number of examples per digit |
| 3 | `task1_model_comparison.png` | Test loss & accuracy for all 5 models |
| 4 | `task1_learning_curves.png` | Train vs test curves per model |
| 5 | `task1_confusion_matrix.png` | Which digits get confused with which |
| 6 | `task1_per_class_acc.png` | Accuracy per digit class |
| 7 | `task1_predictions.png` | Correct and incorrect prediction examples |

## Images Generated by Task 2

| # | File | Description |
|---|------|-------------|
| 8 | `Training GT Boxes.png` | Training images with true object boxes |
| 9 | `Test GT Boxes.png` | Test images with true object boxes |
| 10 | `task2_training_curves.png` | Loss curves during training |
| 11 | `task2_map_comparison.png` | mAP comparison across experiments |
| 12 | `task2_predictions_exp1.png` | Exp1 predictions vs ground truth |
| 13 | `task2_predictions_exp2.png` | Exp2 predictions vs ground truth |
| 14 | `task2_predictions_exp3.png` | Exp3 (temporal) predictions vs ground truth |
| 15 | `task2_all_val_loss.png` | All 3 experiments' validation loss |

---

# Key Concepts Quick Reference

| Term | Simple Explanation |
|------|--------------------|
| **CNN** | Neural network designed for images; uses filters to detect patterns |
| **Epoch** | One full pass through the entire training dataset |
| **Loss** | A number measuring how wrong the model is (lower = better) |
| **Accuracy** | Percentage of correct predictions |
| **Overfitting** | Model memorises training data but fails on new data |
| **Batch Normalisation** | Keeps values stable during training, speeds up learning |
| **Fine-tuning** | Starting from a pre-trained model and adapting it to a new task |
| **Bounding Box** | A rectangle drawn around a detected object [x1, y1, x2, y2] |
| **IoU** | How much a predicted box overlaps with the true box (0=none, 1=perfect) |
| **mAP** | Average detection accuracy across all object classes |
| **Backbone** | The feature extractor part of a detection model |
| **COCO** | A large dataset of 80 object types used to pre-train detection models |
| **LiDAR/Velodyne** | A laser scanner that measures 3D distances to objects |
| **Calibration** | Mathematical relationship between camera and laser scanner positions |

---

*CET3013 Deep Learning — Assessment 2*
