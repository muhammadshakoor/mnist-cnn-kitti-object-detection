# %% [markdown]
# # Task 2: Fine-Tuning a Pre-Trained Model for Object Detection on KITTI
#
# **CET3013 Deep Learning — Assessment 2**
#
# We fine-tune a pre-trained Faster R-CNN (MobileNetV3 backbone) for 2D object detection
# on selected KITTI raw sequences. Labels come from 3D tracklet XML files, which we project
# to 2D bounding boxes using the provided calibration matrices.
#
# **Sequence split (no frame-level leakage):**
# | Role       | Sequences                              | Frames |
# |------------|----------------------------------------|--------|
# | Training   | VideoThree, VideoFive, VideoSeven,     |        |
# |            | Video11, Video12                       | ~1433  |
# | Validation | Video16                                |  383   |
# | Test       | Video15                                |   78   |
#
# **Object classes:** Background=0  Car=1  Pedestrian=2  Cyclist=3

# %% [markdown]
# ## Step 1: Imports

# %%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # fix OpenMP conflict on Windows+Anaconda

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — saves plots as PNG files instead of popup windows

import glob
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
import torchvision.ops as ops
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device : {device}")
print(f"PyTorch      : {torch.__version__}")
print(f"Torchvision  : {torchvision.__version__}")

# %% [markdown]
# ## Step 2: Dataset Paths
#
# Set BASE_DIR to the folder that contains VideoOne, VideoTwo … Video18.

# %%
BASE_DIR = r"c:\Users\HP\Downloads\OneDrive_2026-05-20\DL Assignment Dataset"

# Mapping: folder name → drive number (auto-detected below)
TRAIN_VIDEOS = ['VideoThree', 'VideoFive', 'VideoSeven', 'Video11', 'Video12']
VAL_VIDEOS   = ['Video16']
TEST_VIDEOS  = ['Video15']

# Class mapping — all other object types are treated as background
CLASS_MAP = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}
CLASS_NAMES = {0: 'Background', 1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}
NUM_CLASSES = 4  # background + 3

# Colour per class for visualisation
CLASS_COLOURS = {1: 'red', 2: 'lime', 3: 'cyan'}

# %% [markdown]
# ## Step 3: Utility — Find Paths Inside a Video Folder

# %%
def get_sequence_paths(video_folder):
    """
    Given a video folder (e.g. VideoFive), automatically find:
      - image directory  (image_02/data/)
      - tracklet XML
      - calibration directory (contains calib_*.txt)
    Returns a dict of paths, or None if any are missing.
    """
    base = os.path.join(BASE_DIR, video_folder)

    # Find image data dir
    img_dirs = glob.glob(os.path.join(base, '**', 'image_02', 'data'), recursive=True)
    if not img_dirs:
        print(f"WARNING: no image_02/data found in {video_folder}")
        return None

    # Find tracklet XML
    xml_files = glob.glob(os.path.join(base, '**', 'tracklet_labels.xml'), recursive=True)
    if not xml_files:
        print(f"WARNING: no tracklet_labels.xml found in {video_folder}")
        return None

    # Find calibration directory (contains calib_cam_to_cam.txt)
    calib_files = glob.glob(os.path.join(base, '**', 'calib_cam_to_cam.txt'), recursive=True)
    if not calib_files:
        print(f"WARNING: no calib_cam_to_cam.txt found in {video_folder}")
        return None

    return {
        'img_dir':   img_dirs[0],
        'xml_path':  xml_files[0],
        'calib_dir': os.path.dirname(calib_files[0]),
        'video':     video_folder,
    }


# Quick check — print what was found
all_videos = TRAIN_VIDEOS + VAL_VIDEOS + TEST_VIDEOS
print(f"{'Video':<15} {'Images':>7}  XML  Calib")
print("-" * 45)
for v in all_videos:
    paths = get_sequence_paths(v)
    if paths:
        n_imgs = len(glob.glob(os.path.join(paths['img_dir'], '*.png')))
        print(f"{v:<15} {n_imgs:>7}    OK    OK")
    else:
        print(f"{v:<15}  MISSING")

# %% [markdown]
# ## Step 4: Calibration Loader
#
# We need two calibration files:
# - `calib_velo_to_cam.txt`  — rotation + translation: Velodyne LiDAR → Camera 0
# - `calib_cam_to_cam.txt`   — R_rect_00 (rectification) and P_rect_02 (projection for camera 2)

# %%
def load_calibration(calib_dir):
    """
    Load and return the matrices needed to project 3D velodyne points to image pixels.

    Returns:
        R_velo2cam : (3,3)  rotation from velodyne to cam0
        T_velo2cam : (3,1)  translation from velodyne to cam0
        R_rect00   : (3,3)  rectification matrix for camera 0
        P_rect02   : (3,4)  projection matrix for camera 2 (left colour)
    """
    # --- velo_to_cam ---
    velo_file = os.path.join(calib_dir, 'calib_velo_to_cam.txt')
    velo_data = {}
    with open(velo_file) as f:
        for line in f:
            line = line.strip()
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            try:
                velo_data[key.strip()] = np.fromstring(val, sep=' ')
            except Exception:
                pass
    R_velo2cam = velo_data['R'].reshape(3, 3)
    T_velo2cam = velo_data['T'].reshape(3, 1)

    # --- cam_to_cam ---
    cam_file = os.path.join(calib_dir, 'calib_cam_to_cam.txt')
    cam_data = {}
    with open(cam_file) as f:
        for line in f:
            line = line.strip()
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            try:
                cam_data[key.strip()] = np.fromstring(val, sep=' ')
            except Exception:
                pass
    R_rect00 = cam_data['R_rect_00'].reshape(3, 3)
    P_rect02 = cam_data['P_rect_02'].reshape(3, 4)

    return R_velo2cam, T_velo2cam, R_rect00, P_rect02


def project_velodyne_to_image(points_3d, R_velo2cam, T_velo2cam, R_rect00, P_rect02):
    """
    Project N×3 points in Velodyne space to 2D image coordinates.

    Pipeline:  Velo → Cam0  → Rect  → Image (camera 2)

    Returns:
        u, v    : pixel coordinates (N,)
        depth   : depth in rectified camera space (N,)  — keep only depth > 0
    """
    N = points_3d.shape[0]
    # Step 1: Velodyne → Camera 0
    X_cam = (R_velo2cam @ points_3d.T + T_velo2cam).T            # (N,3)
    # Step 2: Rectification
    X_rect = (R_rect00 @ X_cam.T).T                               # (N,3)
    # Step 3: Homogeneous → Project with P_rect_02 (3×4)
    X_hom = np.hstack([X_rect, np.ones((N, 1))])                  # (N,4)
    x_img = (P_rect02 @ X_hom.T).T                                # (N,3)
    depth = x_img[:, 2]
    u = x_img[:, 0] / np.maximum(depth, 1e-6)
    v = x_img[:, 1] / np.maximum(depth, 1e-6)
    return u, v, depth

# %% [markdown]
# ## Step 5: Tracklet XML Parser
#
# The KITTI raw tracklet format stores 3D bounding boxes as:
# - Dimensions: h (height), w (width), l (length)
# - Per-frame pose: tx, ty, tz (center in Velodyne space), rz (yaw)
# - `tz` is the **bottom** of the bounding box (ground contact z-coordinate)
# - `first_frame`: the frame index where this tracklet begins

# %%
def parse_tracklets(xml_path):
    """
    Parse KITTI tracklet XML file.

    Returns list of dicts:
        {objectType, h, w, l, first_frame, poses: [{tx,ty,tz,rz,occlusion}, ...]}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracklets_node = root.find('tracklets')
    if tracklets_node is None:
        return []

    tracklets = []
    for item in tracklets_node.findall('item'):
        obj_type    = item.findtext('objectType', 'Unknown').strip()
        h           = float(item.findtext('h', 0))
        w           = float(item.findtext('w', 0))
        l           = float(item.findtext('l', 0))
        first_frame = int(item.findtext('first_frame', 0))
        poses = []
        poses_node = item.find('poses')
        if poses_node is not None:
            for pose in poses_node.findall('item'):
                poses.append({
                    'tx':         float(pose.findtext('tx', 0)),
                    'ty':         float(pose.findtext('ty', 0)),
                    'tz':         float(pose.findtext('tz', 0)),
                    'rz':         float(pose.findtext('rz', 0)),
                    'occlusion':  int(pose.findtext('occlusion', -1)),
                    'truncation': int(pose.findtext('truncation', -1)),
                })
        tracklets.append({
            'objectType': obj_type,
            'h': h, 'w': w, 'l': l,
            'first_frame': first_frame,
            'poses': poses,
        })
    return tracklets


def get_3d_box_corners(tx, ty, tz, h, w, l, rz):
    """
    Compute the 8 corners of a 3D bounding box in Velodyne coordinates.

    Convention used here:
      - tx, ty  = center of box in x-y (Velodyne)
      - tz      = z-coordinate of the BOTTOM face (ground level)
      - center_z = tz + h/2
      - rz      = yaw rotation around the z-axis (up)

    Returns corners as (8,3) array.
    """
    cos_rz, sin_rz = np.cos(rz), np.sin(rz)
    Rz = np.array([[cos_rz, -sin_rz, 0],
                   [sin_rz,  cos_rz, 0],
                   [0,       0,      1]])

    # 8 corners in the local (object) frame, centred at origin
    hl, hw, hh = l / 2, w / 2, h / 2
    corners_local = np.array([
        [-hl, -hw, -hh], [-hl,  hw, -hh],
        [ hl,  hw, -hh], [ hl, -hw, -hh],
        [-hl, -hw,  hh], [-hl,  hw,  hh],
        [ hl,  hw,  hh], [ hl, -hw,  hh],
    ])  # (8,3)

    center = np.array([tx, ty, tz + hh])          # true 3D centre
    return (Rz @ corners_local.T).T + center       # (8,3)


def corners_to_2d_box(u, v, depth, img_w=1242, img_h=375):
    """
    Given projected corner coordinates, return a 2D bounding box [x1,y1,x2,y2].
    Only uses corners that are in front of the camera (depth > 0.1).
    Returns None if no valid corners exist or box area < 4 pixels.
    """
    valid = depth > 0.1
    if valid.sum() < 2:
        return None
    u_v, v_v = u[valid], v[valid]
    x1, y1 = float(u_v.min()), float(v_v.min())
    x2, y2 = float(u_v.max()), float(v_v.max())
    # Clip to image bounds
    x1 = max(0.0, min(x1, img_w - 1))
    y1 = max(0.0, min(y1, img_h - 1))
    x2 = max(0.0, min(x2, img_w - 1))
    y2 = max(0.0, min(y2, img_h - 1))
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return None
    return [x1, y1, x2, y2]


def build_frame_annotations(xml_path, calib_dir, img_w=1242, img_h=375):
    """
    Build a dictionary mapping frame_index → list of {box:[x1,y1,x2,y2], label:int}.

    Parses tracklet XML, projects 3D boxes to 2D, filters invalid boxes.
    """
    tracklets          = parse_tracklets(xml_path)
    R_velo2cam, T_velo2cam, R_rect00, P_rect02 = load_calibration(calib_dir)
    frame_annotations  = {}

    for trk in tracklets:
        label = CLASS_MAP.get(trk['objectType'], 0)
        if label == 0:
            continue   # skip unknown/background classes

        for pose_idx, pose in enumerate(trk['poses']):
            frame_idx = trk['first_frame'] + pose_idx

            # Skip heavily occluded objects (occlusion == 3 means fully occluded)
            if pose['occlusion'] >= 3:
                continue

            # 3D corners in Velodyne space
            corners = get_3d_box_corners(
                pose['tx'], pose['ty'], pose['tz'],
                trk['h'], trk['w'], trk['l'], pose['rz']
            )

            # Project to image
            u, v, depth = project_velodyne_to_image(
                corners, R_velo2cam, T_velo2cam, R_rect00, P_rect02
            )

            box_2d = corners_to_2d_box(u, v, depth, img_w, img_h)
            if box_2d is None:
                continue

            if frame_idx not in frame_annotations:
                frame_annotations[frame_idx] = []
            frame_annotations[frame_idx].append({'box': box_2d, 'label': label})

    return frame_annotations

# %% [markdown]
# ## Step 6: Explore Dataset — Object Statistics

# %%
print("Analysing selected sequences ...\n")
print(f"{'Sequence':<15} {'Frames':>7} {'Car':>6} {'Ped':>6} {'Cyc':>6} {'Total boxes':>12}")
print("-" * 60)

total_stats = {1: 0, 2: 0, 3: 0}
for v in all_videos:
    paths = get_sequence_paths(v)
    if paths is None:
        continue
    annots = build_frame_annotations(paths['xml_path'], paths['calib_dir'])
    n_frames = len(glob.glob(os.path.join(paths['img_dir'], '*.png')))
    counts = {1: 0, 2: 0, 3: 0}
    for frame_annots in annots.values():
        for a in frame_annots:
            counts[a['label']] += 1
            total_stats[a['label']] += 1
    role = 'TRAIN' if v in TRAIN_VIDEOS else ('VAL' if v in VAL_VIDEOS else 'TEST')
    total_boxes = sum(counts.values())
    print(f"{v:<15} [{role}]  {n_frames:>5}  {counts[1]:>5}  {counts[2]:>5}  {counts[3]:>5}  {total_boxes:>10}")

print("-" * 60)
print(f"{'TOTAL':<15}        {'':>5}  {total_stats[1]:>5}  {total_stats[2]:>5}  {total_stats[3]:>5}")

# %% [markdown]
# ## Step 7: KITTI PyTorch Dataset Class

# %%
class KITTIDataset(Dataset):
    """
    PyTorch Dataset for KITTI object detection.

    Each item returns:
        image  : FloatTensor (3, H, W), pixel values in [0, 1]
        target : dict with keys:
                   'boxes'  FloatTensor (N, 4)   in [x1, y1, x2, y2] format
                   'labels' Int64Tensor (N,)
    """

    def __init__(self, video_list, subsample=1):
        """
        video_list : list of video folder names (e.g. ['VideoThree', 'VideoFive'])
        subsample  : keep every Nth frame (1 = all frames, 2 = every 2nd, etc.)
        """
        self.samples = []   # list of (img_path, boxes_list, labels_list)

        for v in video_list:
            paths = get_sequence_paths(v)
            if paths is None:
                continue

            annots   = build_frame_annotations(paths['xml_path'], paths['calib_dir'])
            img_paths = sorted(glob.glob(os.path.join(paths['img_dir'], '*.png')))

            for idx, img_path in enumerate(img_paths):
                if idx % subsample != 0:
                    continue
                frame_annots = annots.get(idx, [])
                boxes  = [a['box']   for a in frame_annots]
                labels = [a['label'] for a in frame_annots]
                self.samples.append((img_path, boxes, labels))

        print(f"  Dataset built: {len(self.samples)} frames from {video_list}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, boxes, labels = self.samples[idx]

        # Load image and convert to float tensor
        img = Image.open(img_path).convert('RGB')
        img_tensor = TF.to_tensor(img)            # (3,H,W), values in [0,1]

        # Build target dict
        if len(boxes) > 0:
            boxes_t  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        target = {'boxes': boxes_t, 'labels': labels_t}
        return img_tensor, target


def collate_fn(batch):
    """Custom collate — Faster R-CNN expects lists of images and targets, not stacked tensors."""
    return list(zip(*batch))

# %% [markdown]
# ## Step 8: Visualise Ground-Truth Boxes
#
# Before training, always verify that the data loader is working correctly
# by overlaying the projected 2D bounding boxes on the actual images.

# %%
SUBSAMPLE = 2   # use every 2nd frame for training (adjust to 1 for all frames)

print("Building datasets ...")
train_dataset = KITTIDataset(TRAIN_VIDEOS, subsample=SUBSAMPLE)
val_dataset   = KITTIDataset(VAL_VIDEOS,   subsample=SUBSAMPLE)
test_dataset  = KITTIDataset(TEST_VIDEOS,  subsample=1)          # use all test frames

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,  collate_fn=collate_fn, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

print(f"\nTrain: {len(train_dataset)} frames | Val: {len(val_dataset)} frames | Test: {len(test_dataset)} frames")

# %%
def visualise_gt_boxes(dataset, n_samples=4, title='Ground-Truth Bounding Boxes'):
    """Plot n_samples images from the dataset with GT boxes overlaid."""
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)
    fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 4))
    if n_samples == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        img_t, target = dataset[idx]
        img_np = img_t.permute(1, 2, 0).numpy()  # (H,W,3)
        ax.imshow(img_np)

        for box, lbl in zip(target['boxes'].numpy(), target['labels'].numpy()):
            x1, y1, x2, y2 = box
            colour = CLASS_COLOURS.get(lbl, 'white')
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=colour, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 2, CLASS_NAMES.get(lbl, '?'),
                    color=colour, fontsize=8, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.4, pad=1))

        ax.set_title(f"Frame index: {idx}", fontsize=9)
        ax.axis('off')

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = title.lower().replace(' ', '_') + '.png'
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fname}")


# Visualise ground-truth on a few training samples
print("\nVisualising ground-truth bounding boxes on training samples ...")
visualise_gt_boxes(train_dataset, n_samples=4, title='Training GT Boxes')
print("\nVisualising ground-truth bounding boxes on test samples ...")
visualise_gt_boxes(test_dataset, n_samples=4, title='Test GT Boxes')

# %% [markdown]
# ## Step 9: Model Setup — Faster R-CNN with MobileNetV3 Backbone
#
# **Choice justification:**
# MobileNetV3 is a lightweight, efficient backbone designed for real-time inference.
# Compared to ResNet-50, it is ~3× faster with a small accuracy trade-off, making it
# practical for fine-tuning experiments on CPU. The FPN (Feature Pyramid Network)
# improves detection across multiple scales.
#
# **Fine-tuning strategy (2 experiments):**
# - Experiment 1: Freeze entire backbone.body, train only FPN + RPN + detection head
# - Experiment 2: Unfreeze the last layer block of backbone.body, giving more flexibility

# %%
def build_model(freeze_backbone=True):
    """
    Load Faster R-CNN (MobileNetV3) pre-trained on COCO and replace the detection
    head for NUM_CLASSES (background + Car + Pedestrian + Cyclist).
    """
    # Load pre-trained weights
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model   = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    # Replace the box predictor head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
    )

    if freeze_backbone:
        # Freeze all backbone body parameters
        for param in model.backbone.body.parameters():
            param.requires_grad = False
        print("Backbone FROZEN — training: FPN + RPN + detection head only")
    else:
        # Unfreeze everything (full fine-tuning)
        for param in model.parameters():
            param.requires_grad = True
        print("Full model UNFROZEN — fine-tuning all layers")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")
    return model.to(device)


# %%
def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        if (batch_idx + 1) % 50 == 0:
            print(f"    [epoch {epoch}] batch {batch_idx+1}/{len(loader)}  loss={losses.item():.4f}")

    return total_loss / len(loader)


def evaluate_loss(model, loader):
    """Compute average loss on a loader (model stays in train mode for loss computation)."""
    model.train()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict  = model(images, targets)
            total_loss += sum(loss_dict.values()).item()
    return total_loss / max(len(loader), 1)


def train_detector(model, train_loader, val_loader, epochs=5, lr=0.005, name="Exp"):
    """Full training loop with loss tracking."""
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    history   = {'train_loss': [], 'val_loss': []}

    import time
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        print(f"\n  [{name}] Epoch {epoch}/{epochs}  lr={scheduler.get_last_lr()[0]:.5f}")
        tr_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        va_loss = evaluate_loss(model, val_loader)
        scheduler.step()
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        print(f"  [{name}] Epoch {epoch} | Train loss: {tr_loss:.4f} | Val loss: {va_loss:.4f}")

    print(f"\n  Training done in {time.time()-t0:.0f}s")
    return history

# %% [markdown]
# ## Step 10: Experiment 1 — Frozen Backbone (Head-Only Fine-Tuning)
#
# Freeze the entire MobileNetV3 body, only train the FPN, RPN, and detection head.
# This is the fastest experiment and often already produces good results when
# the pre-training data (COCO) shares similar categories (car, person) with KITTI.

# %%
EPOCHS = 5

print("=" * 65)
print("  EXPERIMENT 1: Frozen Backbone — Head-Only Fine-Tuning")
print("=" * 65)

model_exp1 = build_model(freeze_backbone=True)
history_exp1 = train_detector(
    model_exp1, train_loader, val_loader,
    epochs=EPOCHS, lr=0.005, name="Exp1-Frozen"
)

# Save model
torch.save(model_exp1.state_dict(), 'kitti_exp1_frozen.pth')
print("Model saved: kitti_exp1_frozen.pth")

# %% [markdown]
# ## Step 11: Experiment 2 — Full Fine-Tuning (All Layers Unfrozen)
#
# Unfreeze the entire model and fine-tune with a lower learning rate.
# This gives the backbone the opportunity to adapt its features to KITTI's
# appearance (road scenes, specific camera, lighting conditions).

# %%
print("=" * 65)
print("  EXPERIMENT 2: Full Fine-Tuning (All Layers Unfrozen)")
print("=" * 65)

model_exp2 = build_model(freeze_backbone=False)
history_exp2 = train_detector(
    model_exp2, train_loader, val_loader,
    epochs=EPOCHS, lr=0.001, name="Exp2-FullFT"
)

torch.save(model_exp2.state_dict(), 'kitti_exp2_fullft.pth')
print("Model saved: kitti_exp2_fullft.pth")

# %% [markdown]
# ## Step 12: Plot Training Curves

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ep = range(1, EPOCHS + 1)

ax1.plot(ep, history_exp1['train_loss'], 'o-', label='Exp1 Train', color='royalblue')
ax1.plot(ep, history_exp1['val_loss'],   's--', label='Exp1 Val',   color='royalblue', alpha=0.6)
ax1.plot(ep, history_exp2['train_loss'], 'o-', label='Exp2 Train', color='tomato')
ax1.plot(ep, history_exp2['val_loss'],   's--', label='Exp2 Val',   color='tomato', alpha=0.6)
ax1.set_title('Total Training Loss', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.bar(['Exp1\n(Frozen)', 'Exp2\n(FullFT)'],
        [min(history_exp1['val_loss']), min(history_exp2['val_loss'])],
        color=['royalblue', 'tomato'], edgecolor='black')
ax2.set_title('Best Validation Loss', fontsize=13, fontweight='bold')
ax2.set_ylabel('Min Validation Loss')

plt.suptitle('Faster R-CNN Training on KITTI', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('task2_training_curves.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: task2_training_curves.png")

# %% [markdown]
# ## Step 13: Detection Evaluation — IoU and mAP
#
# We evaluate on the held-out test sequence (Video15) using:
# - **IoU** (Intersection over Union): measures overlap between predicted and GT boxes
# - **mAP** (mean Average Precision): standard object detection metric, computed
#   per class at IoU threshold 0.5 (PASCAL VOC convention)

# %%
def compute_iou(box_a, box_b):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1); inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2); inter_y2 = min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a     = (xa2 - xa1) * (ya2 - ya1)
    area_b     = (xb2 - xb1) * (yb2 - yb1)
    union      = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def compute_average_precision(all_pred_scores, all_pred_matches, n_gt_total):
    """
    Compute AP for a single class using the 11-point interpolation (PASCAL VOC 2007).

    all_pred_scores  : list of confidence scores for all detections
    all_pred_matches : list of 1/0 (True positive or False positive)
    n_gt_total       : total number of GT boxes for this class
    """
    if n_gt_total == 0:
        return 0.0
    # Sort by confidence (descending)
    order   = np.argsort(all_pred_scores)[::-1]
    matches = np.array(all_pred_matches)[order]

    tp = np.cumsum(matches)
    fp = np.cumsum(1 - matches)
    recalls    = tp / n_gt_total
    precisions = tp / (tp + fp + 1e-9)

    # 11-point interpolation
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_rec = precisions[recalls >= thr]
        ap += (prec_at_rec.max() if len(prec_at_rec) > 0 else 0.0) / 11.0
    return float(ap)


def evaluate_detection(model, loader, iou_threshold=0.5, score_threshold=0.3):
    """
    Run inference on the loader and compute per-class AP and mean IoU.

    Returns:
        ap_per_class   : dict {class_id: AP value}
        mean_iou       : float
        all_results    : list of (image_tensor, gt_target, predictions) for visualisation
    """
    model.eval()

    # Accumulators per class
    pred_scores  = {c: [] for c in CLASS_MAP.values()}
    pred_matches = {c: [] for c in CLASS_MAP.values()}
    gt_counts    = {c: 0  for c in CLASS_MAP.values()}
    all_ious     = []
    all_results  = []

    with torch.no_grad():
        for images, targets in loader:
            images_dev = [img.to(device) for img in images]
            preds      = model(images_dev)

            for img_t, target, pred in zip(images, targets, preds):
                gt_boxes  = target['boxes'].numpy()
                gt_labels = target['labels'].numpy()

                # Filter predictions by score
                keep       = pred['scores'].cpu().numpy() >= score_threshold
                pred_boxes  = pred['boxes'].cpu().numpy()[keep]
                pred_labels = pred['labels'].cpu().numpy()[keep]
                pred_scores_arr = pred['scores'].cpu().numpy()[keep]

                # Track GT counts
                for lbl in gt_labels:
                    if lbl in gt_counts:
                        gt_counts[lbl] += 1

                # Track which GT boxes have been matched (prevent double-matching)
                matched_gt = set()

                for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores_arr):
                    if pl not in pred_scores:
                        continue
                    pred_scores[pl].append(ps)

                    # Find best-matching GT box of the same class
                    best_iou, best_gt_idx = 0.0, -1
                    for gi, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if gl != pl or gi in matched_gt:
                            continue
                        iou = compute_iou(pb, gb)
                        if iou > best_iou:
                            best_iou, best_gt_idx = iou, gi

                    if best_iou >= iou_threshold:
                        pred_matches[pl].append(1)
                        matched_gt.add(best_gt_idx)
                        all_ious.append(best_iou)
                    else:
                        pred_matches[pl].append(0)

                all_results.append((img_t, target, pred))

    # Compute AP per class
    ap_per_class = {}
    for c in CLASS_MAP.values():
        ap_per_class[c] = compute_average_precision(
            pred_scores[c], pred_matches[c], gt_counts[c]
        )

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    return ap_per_class, mean_iou, all_results


# %%
print("Evaluating Experiment 1 (Frozen Backbone) on test set ...")
ap1, miou1, results1 = evaluate_detection(model_exp1, test_loader)

print("\nEvaluating Experiment 2 (Full Fine-Tuning) on test set ...")
ap2, miou2, results2 = evaluate_detection(model_exp2, test_loader)

# Print results
print("\n" + "=" * 65)
print("  DETECTION EVALUATION RESULTS — Test Sequence (Video15)")
print("=" * 65)
print(f"\n  {'Class':<12} {'Exp1 AP':>10} {'Exp2 AP':>10}")
print(f"  {'-'*35}")
for c in CLASS_MAP.values():
    print(f"  {CLASS_NAMES[c]:<12} {ap1[c]:>10.4f} {ap2[c]:>10.4f}")
mAP1 = np.mean(list(ap1.values()))
mAP2 = np.mean(list(ap2.values()))
print(f"  {'-'*35}")
print(f"  {'mAP':<12} {mAP1:>10.4f} {mAP2:>10.4f}")
print(f"\n  Mean IoU (TP only)  Exp1: {miou1:.4f}  Exp2: {miou2:.4f}")
print("=" * 65)

# %%
# Bar chart comparing mAP per class and overall
class_labels = [CLASS_NAMES[c] for c in CLASS_MAP.values()]
x = np.arange(len(class_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, [ap1[c] for c in CLASS_MAP.values()], width,
               label='Exp1: Frozen Backbone', color='royalblue', edgecolor='black')
bars2 = ax.bar(x + width/2, [ap2[c] for c in CLASS_MAP.values()], width,
               label='Exp2: Full Fine-Tune',  color='tomato',     edgecolor='black')
for bar in bars1 + bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}',
            ha='center', va='bottom', fontsize=8)
ax.set_xlabel('Object Class'); ax.set_ylabel('Average Precision (AP)')
ax.set_title('Per-Class AP Comparison — KITTI Test Set', fontsize=13, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(class_labels)
ax.legend(); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('task2_map_comparison.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: task2_map_comparison.png")

# %% [markdown]
# ## Step 14: Visualise Predictions vs Ground Truth

# %%
def visualise_predictions(results_list, model_name='Model', n_samples=4,
                           score_threshold=0.3, save_name='predictions.png'):
    """
    Draw GT boxes (green dashed) and predicted boxes (solid colour) side-by-side.
    Samples are chosen to spread across the test set.
    """
    indices = np.linspace(0, len(results_list) - 1, n_samples, dtype=int)
    fig, axes = plt.subplots(2, n_samples, figsize=(5 * n_samples, 7))

    for col, idx in enumerate(indices):
        img_t, target, pred = results_list[idx]
        img_np = img_t.permute(1, 2, 0).numpy()

        for row in range(2):
            axes[row, col].imshow(img_np)
            axes[row, col].axis('off')

        # --- Row 0: Ground Truth ---
        for box, lbl in zip(target['boxes'].numpy(), target['labels'].numpy()):
            x1, y1, x2, y2 = box
            c = CLASS_COLOURS.get(lbl, 'white')
            axes[0, col].add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=c, facecolor='none', linestyle='--'))
            axes[0, col].text(x1, y1-2, CLASS_NAMES.get(lbl, '?'),
                              color=c, fontsize=7, fontweight='bold',
                              bbox=dict(facecolor='black', alpha=0.4, pad=1))
        axes[0, col].set_title(f'GT (frame {idx})', fontsize=8)

        # --- Row 1: Predictions ---
        keep        = pred['scores'].cpu().numpy() >= score_threshold
        pred_boxes  = pred['boxes'].cpu().numpy()[keep]
        pred_labels = pred['labels'].cpu().numpy()[keep]
        pred_sc     = pred['scores'].cpu().numpy()[keep]
        for box, lbl, sc in zip(pred_boxes, pred_labels, pred_sc):
            x1, y1, x2, y2 = box
            c = CLASS_COLOURS.get(lbl, 'white')
            axes[1, col].add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=c, facecolor='none'))
            axes[1, col].text(x1, y1-2, f'{CLASS_NAMES.get(lbl,"?")} {sc:.2f}',
                              color=c, fontsize=7, fontweight='bold',
                              bbox=dict(facecolor='black', alpha=0.4, pad=1))
        axes[1, col].set_title(f'Pred (score≥{score_threshold})', fontsize=8)

    axes[0, 0].set_ylabel('Ground Truth', fontsize=10, fontweight='bold', color='green')
    axes[1, 0].set_ylabel('Predictions',  fontsize=10, fontweight='bold', color='royalblue')
    plt.suptitle(f'Detection Results — {model_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_name, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_name}")


visualise_predictions(results1, model_name='Exp1: Frozen Backbone',
                      save_name='task2_predictions_exp1.png')
visualise_predictions(results2, model_name='Exp2: Full Fine-Tuning',
                      save_name='task2_predictions_exp2.png')

# %% [markdown]
# ## Step 15: Bonus — Temporal Feature Stacking
#
# KITTI frames are sequential. We can exploit temporal information by stacking
# two consecutive frames as a 6-channel input (frame_t and frame_{t-1}).
# This gives the network implicit motion cues without a complex architecture.
#
# NOTE: This experiment modifies the first convolutional layer of the backbone,
# making it incompatible with COCO pre-trained weights for the first layer only.
# All other layers retain their pre-trained weights.

# %%
class TemporalKITTIDataset(Dataset):
    """
    Like KITTIDataset but each sample stacks the current frame with the previous frame.
    Input to model: 6-channel tensor (frame_{t} RGB + frame_{t-1} RGB).
    """
    def __init__(self, video_list, subsample=1):
        self.samples = []
        for v in video_list:
            paths = get_sequence_paths(v)
            if paths is None:
                continue
            annots    = build_frame_annotations(paths['xml_path'], paths['calib_dir'])
            img_paths = sorted(glob.glob(os.path.join(paths['img_dir'], '*.png')))

            for idx in range(1, len(img_paths)):  # start from 1 (need prev frame)
                if idx % subsample != 0:
                    continue
                frame_annots = annots.get(idx, [])
                boxes  = [a['box']   for a in frame_annots]
                labels = [a['label'] for a in frame_annots]
                self.samples.append((img_paths[idx], img_paths[idx - 1], boxes, labels))

        print(f"  Temporal dataset: {len(self.samples)} pairs from {video_list}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cur_path, prev_path, boxes, labels = self.samples[idx]
        cur_t  = TF.to_tensor(Image.open(cur_path).convert('RGB'))
        prev_t = TF.to_tensor(Image.open(prev_path).convert('RGB'))
        img_6ch = torch.cat([cur_t, prev_t], dim=0)   # (6, H, W)

        if len(boxes) > 0:
            boxes_t  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        return img_6ch, {'boxes': boxes_t, 'labels': labels_t}


def build_temporal_model():
    """
    Modify Faster R-CNN MobileNetV3 to accept 6-channel input.
    The first conv layer is re-initialised (averages 3-ch weights and doubles them).
    """
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model   = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    # Replace detection head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
    )

    # Modify first conv layer: in_channels 3 → 6
    # Strategy: duplicate the existing 3-ch weights and scale by 0.5 to preserve normalisation
    first_conv = model.backbone.body.features[0][0]           # Conv2d(3,16,...)
    old_weight = first_conv.weight.data                        # (16,3,3,3)
    new_weight = torch.cat([old_weight, old_weight], dim=1) * 0.5  # (16,6,3,3)

    model.backbone.body.features[0][0] = nn.Conv2d(
        6, first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None
    )
    model.backbone.body.features[0][0].weight.data = new_weight
    print("Temporal model: first conv layer modified 3-ch → 6-ch")
    return model.to(device)

import torch.nn as nn   # needed for nn.Conv2d above

# Build temporal datasets
print("Building temporal datasets ...")
temp_train_ds = TemporalKITTIDataset(TRAIN_VIDEOS, subsample=SUBSAMPLE)
temp_val_ds   = TemporalKITTIDataset(VAL_VIDEOS,   subsample=SUBSAMPLE)
temp_test_ds  = TemporalKITTIDataset(TEST_VIDEOS,  subsample=1)

temp_train_loader = DataLoader(temp_train_ds, batch_size=2, shuffle=True,  collate_fn=collate_fn, num_workers=0)
temp_val_loader   = DataLoader(temp_val_ds,   batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)
temp_test_loader  = DataLoader(temp_test_ds,  batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

# %%
print("=" * 65)
print("  EXPERIMENT 3 (BONUS): Temporal Feature Stacking")
print("=" * 65)

model_exp3  = build_temporal_model()
history_exp3 = train_detector(
    model_exp3, temp_train_loader, temp_val_loader,
    epochs=EPOCHS, lr=0.002, name="Exp3-Temporal"
)
torch.save(model_exp3.state_dict(), 'kitti_exp3_temporal.pth')
print("Model saved: kitti_exp3_temporal.pth")

# %%
print("Evaluating Experiment 3 (Temporal) on test set ...")
ap3, miou3, results3 = evaluate_detection(model_exp3, temp_test_loader)

# Final comparison table
print("\n" + "=" * 75)
print("  FINAL COMPARISON — All Experiments")
print("=" * 75)
print(f"\n  {'Class':<12} {'Exp1 (Frozen)':>14} {'Exp2 (FullFT)':>14} {'Exp3 (Temporal)':>16}")
print(f"  {'-'*58}")
for c in CLASS_MAP.values():
    print(f"  {CLASS_NAMES[c]:<12} {ap1[c]:>14.4f} {ap2[c]:>14.4f} {ap3[c]:>16.4f}")
mAP3 = np.mean(list(ap3.values()))
print(f"  {'-'*58}")
print(f"  {'mAP':<12} {mAP1:>14.4f} {mAP2:>14.4f} {mAP3:>16.4f}")
print(f"\n  Mean IoU    Exp1: {miou1:.4f}  Exp2: {miou2:.4f}  Exp3: {miou3:.4f}")
print("=" * 75)

# %%
visualise_predictions(results3, model_name='Exp3: Temporal Stacking',
                      save_name='task2_predictions_exp3.png')

# %% [markdown]
# ## Step 16: Final Summary

# %%
# Combined training loss plot
fig, ax = plt.subplots(figsize=(10, 5))
ep = range(1, EPOCHS + 1)
ax.plot(ep, history_exp1['val_loss'], 'o-', label='Exp1: Frozen Backbone', color='royalblue')
ax.plot(ep, history_exp2['val_loss'], 's-', label='Exp2: Full Fine-Tune',  color='tomato')
ax.plot(ep, history_exp3['val_loss'], '^-', label='Exp3: Temporal Stack',  color='seagreen')
ax.set_title('Validation Loss — All Experiments', fontsize=13, fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Validation Loss')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task2_all_val_loss.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: task2_all_val_loss.png")

print("\n" + "=" * 65)
print("  TASK 2 COMPLETE")
print("=" * 65)
print("\nSaved model files:")
print("  kitti_exp1_frozen.pth")
print("  kitti_exp2_fullft.pth")
print("  kitti_exp3_temporal.pth")
print("\nSaved figures:")
for f in ['task2_training_curves.png', 'task2_map_comparison.png',
          'task2_predictions_exp1.png', 'task2_predictions_exp2.png',
          'task2_predictions_exp3.png', 'task2_all_val_loss.png',
          'Training GT Boxes.png', 'Test GT Boxes.png']:
    print(f"  {f}")
