# How to Run This Project on Any Laptop or PC
### CET3013 Deep Learning — Complete Setup Guide (Step by Step)

---

## What You Need Before Starting

| Requirement | Details |
|-------------|---------|
| Operating System | Windows 10 or Windows 11 (64-bit) |
| RAM | Minimum 8 GB (16 GB recommended) |
| Storage | At least 5 GB free space |
| Internet | Required for first-time setup only |
| GPU | Not required — runs on CPU (training will be slow but works) |

---

## Overview of Steps

```
STEP 1 → Install Anaconda (Python manager)
STEP 2 → Copy the project files to the new PC
STEP 3 → Open Anaconda Prompt
STEP 4 → Install all required packages
STEP 5 → Launch Jupyter Notebook
STEP 6 → Run Task 1 (MNIST)
STEP 7 → Run Task 2 (KITTI Detection)
```

---

---

## STEP 1 — Install Anaconda

Anaconda is a Python manager that includes Jupyter Notebook and makes installing packages easy.

### 1.1 Download Anaconda

1. Open any web browser
2. Go to: **https://www.anaconda.com/download**
3. Click **Download** (it will detect your OS automatically)
4. The file will be named something like: `Anaconda3-2024.xx-Windows-x86_64.exe`

### 1.2 Install Anaconda

1. Double-click the downloaded `.exe` file
2. Click **Next** → **I Agree** → **Next**
3. On "Installation Type" → select **Just Me** → click **Next**
4. Leave the install path as default → click **Next**
5. On "Advanced Options":
   - **Check** the box: "Add Anaconda3 to my PATH environment variable"
   - **Check** the box: "Register Anaconda3 as my default Python"
6. Click **Install** and wait (takes 5–10 minutes)
7. Click **Finish** when done

### 1.3 Verify Installation

1. Press `Windows key` → type `Anaconda Prompt` → press Enter
2. A black terminal window opens
3. Type this and press Enter:
```
python --version
```
You should see something like:
```
Python 3.12.x
```
If you see this, Anaconda is installed correctly. ✓

---

---

## STEP 2 — Copy the Project Files to the New PC

You need to copy the **entire project folder** to the new PC.

### What to Copy

Copy the entire folder named `DL Assignment Dataset`. It must contain:

```
DL Assignment Dataset/
│
├── task1_mnist_cnn.ipynb          ← REQUIRED
├── task1_mnist_cnn.py             ← REQUIRED
├── task2_kitti_detection.ipynb    ← REQUIRED
├── task2_kitti_detection.py       ← REQUIRED
│
├── Video9/                        ← REQUIRED (KITTI data)
├── Video10/                       ← REQUIRED
├── Video11/                       ← REQUIRED
├── Video12/                       ← REQUIRED
├── Video13/                       ← REQUIRED
├── Video14/                       ← REQUIRED
├── Video15/                       ← REQUIRED (test data)
├── Video16/                       ← REQUIRED (validation data)
├── Video17/                       ← REQUIRED
├── Video18/                       ← REQUIRED
├── VideoOne/                      ← REQUIRED
├── VideoTwo/                      ← REQUIRED
├── VideoThree/                    ← REQUIRED (training data)
├── VideoFour/                     ← REQUIRED
├── VideoFive/                     ← REQUIRED (training data)
├── VideoSix/                      ← REQUIRED
├── VideoSeven/                    ← REQUIRED (training data)
├── VideoEight/                    ← REQUIRED
│
└── mnist_data/                    ← OPTIONAL (will auto-download if missing)
```

### How to Copy

**Option A — USB Drive:**
1. Plug in a USB drive on the original PC
2. Copy the entire `DL Assignment Dataset` folder to the USB
3. Plug the USB into the new PC
4. Copy the folder to: `C:\Users\YourName\Downloads\`

**Option B — OneDrive / Google Drive:**
1. Upload the entire folder to OneDrive or Google Drive
2. On the new PC, sign in and download the folder

**Option C — External Hard Drive:** Same process as USB.

**Option D — Download KITTI data fresh from the official website (if you cloned from GitHub):**

> The KITTI dataset is publicly available. The code uses specific drives from the 2011-09-26 date.
> The GitHub repository does **not** include the Video folders — you must download them separately.

1. Go to: **https://www.cvlibs.net/datasets/kitti/raw_data.php** (requires free registration)
2. Download the following drives (synced+rectified data + tracklets for each):

| Folder Name | KITTI Drive | Download Files |
|-------------|-------------|----------------|
| VideoOne | 2011_09_26_drive_0001 | `_sync.zip` + `_tracklets.zip` |
| VideoTwo | 2011_09_26_drive_0002 | `_sync.zip` + `_tracklets.zip` |
| VideoThree *(training)* | 2011_09_26_drive_0005 | `_sync.zip` + `_tracklets.zip` |
| VideoFour | 2011_09_26_drive_0009 | `_sync.zip` + `_tracklets.zip` |
| VideoFive *(training)* | 2011_09_26_drive_0011 | `_sync.zip` + `_tracklets.zip` |
| VideoSix | 2011_09_26_drive_0013 | `_sync.zip` + `_tracklets.zip` |
| VideoSeven *(training)* | 2011_09_26_drive_0014 | `_sync.zip` + `_tracklets.zip` |
| VideoEight | 2011_09_26_drive_0017 | `_sync.zip` + `_tracklets.zip` |
| Video9 | 2011_09_26_drive_0018 | `_sync.zip` + `_tracklets.zip` |
| Video10 | 2011_09_26_drive_0048 | `_sync.zip` + `_tracklets.zip` |
| Video11 *(training)* | 2011_09_26_drive_0051 | `_sync.zip` + `_tracklets.zip` |
| Video12 *(training)* | 2011_09_26_drive_0056 | `_sync.zip` + `_tracklets.zip` |
| Video13 | 2011_09_26_drive_0057 | `_sync.zip` + `_tracklets.zip` |
| Video14 | 2011_09_26_drive_0059 | `_sync.zip` + `_tracklets.zip` |
| Video15 *(test)* | 2011_09_26_drive_0060 | `_sync.zip` + `_tracklets.zip` |
| Video16 *(validation)* | 2011_09_26_drive_0084 | `_sync.zip` + `_tracklets.zip` |
| Video17 | 2011_09_26_drive_0091 | `_sync.zip` + `_tracklets.zip` |
| Video18 | 2011_09_26_drive_0093 | `_sync.zip` + `_tracklets.zip` |

3. Also download the calibration file for this date:
   - On the same page, find and download **`2011_09_26_calib.zip`**

4. Extract each `_sync.zip` into its Video folder. The inner folder structure must be kept exactly as extracted — do NOT flatten it.

5. Extract each `_tracklets.zip` into the same Video folder.

6. Extract `2011_09_26_calib.zip` and copy the `2011_09_26/` calibration folder into **each** Video folder.

> **Minimum download** (only training/val/test videos): VideoThree, VideoFive, VideoSeven, Video11, Video12, Video15, Video16.
> Total download size for these 7 sequences: approximately 1.5 GB.

> **Important:** Keep the folder structure exactly as it is. Do NOT rename or move any inner folders.

---

---

## STEP 3 — Update the File Path in the Notebook

The notebooks have a path hardcoded to the original PC. You must update it.

### For Task 2 notebook only:

1. Find where you saved the project on the new PC (e.g., `C:\Users\JohnPC\Downloads\DL Assignment Dataset`)
2. Open `task2_kitti_detection.ipynb` in Jupyter (see Step 5 first)
3. Find this line (it is in **Step 2** of the notebook, near the top):

```python
BASE_DIR = r"c:\Users\HP\Downloads\OneDrive_2026-05-20\DL Assignment Dataset"
```

4. Change it to match where YOU saved the folder. Examples:

```python
# If saved in Downloads folder:
BASE_DIR = r"C:\Users\YourName\Downloads\DL Assignment Dataset"

# If saved on Desktop:
BASE_DIR = r"C:\Users\YourName\Desktop\DL Assignment Dataset"

# If saved on D: drive:
BASE_DIR = r"D:\DL Assignment Dataset"
```

> **How to find your exact path:**
> - Open File Explorer
> - Navigate to the `DL Assignment Dataset` folder
> - Click the address bar at the top — it shows the full path
> - Copy that path and paste it into `BASE_DIR`

Task 1 does **not** need any path changes (MNIST downloads automatically).

---

---

## STEP 4 — Install Required Python Packages

This is the most important step. All packages must be installed before running anything.

### 4.1 Open Anaconda Prompt

Press `Windows key` → search for **Anaconda Prompt** → click to open it.

> **Important:** Always use **Anaconda Prompt**, NOT regular Command Prompt or PowerShell.

### 4.2 Install PyTorch (the Deep Learning Engine)

Copy and paste this command exactly, then press Enter:

```
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```

This installs PyTorch for CPU (no GPU required). It will download about 200–300 MB. Wait for it to complete — it may take 5–10 minutes.

You will see:
```
Successfully installed torch-2.2.0 torchvision-0.17.0 ...
```

### 4.3 Install All Other Required Packages

Copy and paste each command one at a time, pressing Enter after each:

```
pip install numpy
```
```
pip install matplotlib
```
```
pip install Pillow
```
```
pip install scikit-learn
```
```
pip install seaborn
```
```
pip install jupyter
```
```
pip install jupyterlab
```

Wait for each one to say `Successfully installed` before typing the next.

### 4.4 Verify Everything Installed Correctly

Type this command and press Enter:

```
python -c "import torch; import torchvision; import numpy; import matplotlib; import sklearn; import seaborn; print('ALL OK')"
```

If you see `ALL OK` — all packages are installed correctly. ✓

If you see an error like `ModuleNotFoundError: No module named 'seaborn'` — run `pip install seaborn` again.

---

---

## STEP 5 — Launch Jupyter Notebook

### 5.1 Navigate to the Project Folder

In Anaconda Prompt, type the following (replace the path with where you saved the project):

```
cd "C:\Users\YourName\Downloads\DL Assignment Dataset"
```

Press Enter. The prompt should now show your folder path.

### 5.2 Start Jupyter Notebook

Type this and press Enter:

```
jupyter notebook
```

After a few seconds, your web browser will open automatically showing the Jupyter file browser.

You should see all the project files listed:
- `task1_mnist_cnn.ipynb`
- `task2_kitti_detection.ipynb`
- All the Video folders

> **If the browser does NOT open automatically:**
> Look in the Anaconda Prompt window. You will see a URL like:
> `http://localhost:8888/?token=abc123...`
> Copy that URL and paste it into your browser manually.

---

---

## STEP 6 — Run Task 1: MNIST CNN

### 6.1 Open the Notebook

In the Jupyter browser, click on **`task1_mnist_cnn.ipynb`**

The notebook will open in a new tab with all the code cells visible.

### 6.2 Select the Kernel

At the top right of the notebook, you should see a kernel selector.
- If it shows **Python 3 (ipykernel)** or **base** — you are ready ✓
- If it shows **No Kernel** or **Not Connected**:
  1. Click on it
  2. Select **Python 3 (ipykernel)**

### 6.3 Run All Cells

**Method 1 (Recommended — run all at once):**
- Click the menu: **Kernel** → **Restart & Run All**
- Click **Restart and Run All Cells** in the popup

**Method 2 (Run one cell at a time):**
- Click on the first cell
- Press **Shift + Enter** to run it and move to the next
- Repeat for each cell

### 6.4 What to Expect

- **First run:** MNIST dataset will download automatically (about 11 MB) into `mnist_data/` folder
- **Training time:** About 30–90 minutes on CPU for all 5 models × 10 epochs
- **While running:** You will see epoch-by-epoch progress printed below each cell:

```
=================================================================
  Training A: Baseline
=================================================================
  Epoch  1/10 | Train: loss=0.1823 acc=94.52% | Test: loss=0.0612 acc=98.07%
  Epoch  2/10 | Train: loss=0.0512 acc=98.42% | Test: loss=0.0448 acc=98.63%
  ...
```

### 6.5 Output Files Created

After Task 1 completes, these image files will appear in your project folder:

```
task1_mnist_samples.png
task1_class_distribution.png
task1_model_comparison.png
task1_learning_curves.png
task1_confusion_matrix.png
task1_per_class_acc.png
task1_predictions.png
```

---

---

## STEP 7 — Run Task 2: KITTI Object Detection

### 7.1 Open the Notebook

In Jupyter, click on **`task2_kitti_detection.ipynb`**

### 7.2 Update BASE_DIR First (Important!)

Before running anything, find the cell in **Step 2** that contains:

```python
BASE_DIR = r"c:\Users\HP\Downloads\OneDrive_2026-05-20\DL Assignment Dataset"
```

Change it to your actual path (as explained in Step 3 above).

### 7.3 Run All Cells

- Click **Kernel** → **Restart & Run All**

### 7.4 What to Expect

Task 2 is much slower than Task 1. Here is the timeline:

| Stage | Estimated Time (CPU) |
|-------|---------------------|
| Steps 1–8 (loading data, visualising) | 5–10 minutes |
| Experiment 1 training (5 epochs) | 2–4 hours |
| Experiment 2 training (5 epochs) | 3–5 hours |
| Experiment 3 / Bonus (5 epochs) | 3–5 hours |
| Evaluation and plots | 10–20 minutes |
| **Total** | **8–15 hours** |

> **Tip:** You can run the notebook overnight. Jupyter keeps running even if you close the browser tab — just don't close the Anaconda Prompt window.

### 7.5 Watching Progress

During training you will see output like:
```
[Exp1-Frozen] Epoch 1/5  lr=0.00500
    [epoch 1] batch 50/359  loss=1.2453
    [epoch 1] batch 100/359  loss=0.9821
    ...
[Exp1-Frozen] Epoch 1 | Train loss: 0.8432 | Val loss: 0.7918
```

### 7.6 Output Files Created

```
Training GT Boxes.png
Test GT Boxes.png
task2_training_curves.png
task2_map_comparison.png
task2_predictions_exp1.png
task2_predictions_exp2.png
task2_predictions_exp3.png
task2_all_val_loss.png
kitti_exp1_frozen.pth       ← saved model weights
kitti_exp2_fullft.pth       ← saved model weights
kitti_exp3_temporal.pth     ← saved model weights
```

---

---

## Common Errors and How to Fix Them

### Error 1 — `ModuleNotFoundError: No module named 'torch'`

**Cause:** PyTorch was not installed correctly.

**Fix:** Open Anaconda Prompt and run:
```
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```
Then restart the Jupyter kernel: **Kernel → Restart & Run All**

---

### Error 2 — `ModuleNotFoundError: No module named 'seaborn'`

**Cause:** Seaborn not installed.

**Fix:**
```
pip install seaborn
```

---

### Error 3 — `FileNotFoundError: [Errno 2] No such file or directory`

**Cause:** `BASE_DIR` path is wrong in Task 2.

**Fix:** Update `BASE_DIR` in Step 2 of the notebook to match your actual folder path. Double-check using File Explorer.

---

### Error 4 — `WARNING: no image_02/data found in VideoThree`

**Cause:** The Video folders are not in the same folder as the notebook, OR the folder was renamed.

**Fix:**
- Make sure the folder structure is exactly: `DL Assignment Dataset/VideoThree/...`
- Do NOT rename any folders
- Make sure `BASE_DIR` points to the folder that directly CONTAINS the Video folders

---

### Error 5 — Jupyter Notebook shows a blank page / cells don't appear

**Cause:** Browser extension conflict or rendering issue.

**Fix:**
1. Press `Ctrl + Shift + P` in the browser → Clear cache
2. Or try a different browser (Chrome, Firefox, Edge)
3. Or add `?` to the Jupyter URL: `http://localhost:8888/tree?`

---

### Error 6 — Training is very slow / seems stuck

**Cause:** Normal behaviour on CPU. Deep learning training is slow without a GPU.

**What to expect:** One epoch of Task 2 training takes 20–40 minutes on CPU.

**Tip:** Check that something IS happening by looking at Anaconda Prompt — it will show batch progress every 50 batches.

---

### Error 7 — `KernelDied` or kernel keeps restarting

**Cause:** Not enough RAM. The KITTI model requires 4–6 GB of RAM.

**Fix:**
- Close all other programs (Chrome tabs, etc.) to free up RAM
- In Task 2 notebook, find `SUBSAMPLE = 2` and change it to `SUBSAMPLE = 4` to use fewer frames and less memory

---

### Error 8 — `pip` command not found

**Cause:** Using regular Command Prompt instead of Anaconda Prompt.

**Fix:** Close the window. Search for **Anaconda Prompt** in the Start menu and use that instead.

---

---

## Quick Checklist Before Running

Use this checklist every time you set up on a new PC:

- [ ] Anaconda is installed and Anaconda Prompt opens
- [ ] `python --version` shows Python 3.x in Anaconda Prompt
- [ ] All packages installed (`torch`, `torchvision`, `numpy`, `matplotlib`, `sklearn`, `seaborn`, `PIL`)
- [ ] `python -c "import torch; print('ALL OK')"` prints `ALL OK`
- [ ] Project folder copied with all Video subfolders intact
- [ ] `BASE_DIR` updated in `task2_kitti_detection.ipynb` Step 2
- [ ] Jupyter launched from the project folder using Anaconda Prompt
- [ ] Kernel set to `base (Python 3)` in the notebook

---

---

## Folder Structure Check

After copying the project, open File Explorer and verify the folder looks exactly like this:

```
DL Assignment Dataset\
    ├── task1_mnist_cnn.ipynb
    ├── task2_kitti_detection.ipynb
    ├── Video15\
    │       └── 2011_09_26_drive_XXXX_sync\
    │               └── 2011_09_26\
    │                       └── 2011_09_26_drive_XXXX_sync\
    │                               └── image_02\
    │                                       └── data\
    │                                               ├── 0000000000.png
    │                                               ├── 0000000001.png
    │                                               └── ...
    ├── Video15\
    │       └── 2011_09_26_drive_XXXX_tracklets\
    │               └── 2011_09_26\
    │                       └── 2011_09_26_drive_XXXX_sync\
    │                               └── tracklet_labels.xml
    └── Video15\
            └── 2011_09_26\
                    ├── calib_cam_to_cam.txt
                    ├── calib_imu_to_velo.txt
                    └── calib_velo_to_cam.txt
```

Each Video folder MUST have:
1. A folder with `.png` images inside `image_02/data/`
2. A `tracklet_labels.xml` file
3. Calibration `.txt` files (`calib_cam_to_cam.txt`, `calib_velo_to_cam.txt`)

---

---

## Summary: Commands to Copy-Paste

Here are all Anaconda Prompt commands in one place for easy reference:

```
# Step 1: Install PyTorch
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install other packages
pip install numpy matplotlib Pillow scikit-learn seaborn jupyter jupyterlab

# Step 3: Verify installation
python -c "import torch; import torchvision; import numpy; import matplotlib; import sklearn; import seaborn; print('ALL OK')"

# Step 4: Navigate to project folder (change path to match yours)
cd "C:\Users\YourName\Downloads\DL Assignment Dataset"

# Step 5: Launch Jupyter
jupyter notebook
```

---

*CET3013 Deep Learning — Assessment 2 Setup Guide*
