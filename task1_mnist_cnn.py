# %% [markdown]
# # Task 1: Building and Training a CNN from Scratch on MNIST
#
# **CET3013 Deep Learning — Assessment 2**
#
# We build 5 CNN architectures and compare them systematically on MNIST.
# Architecture variants: Baseline → More Filters → Deeper → BatchNorm → ELU+BatchNorm

# %% [markdown]
# ## Step 1: Imports

# %%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # fix OpenMP conflict on Windows+Anaconda

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — saves plots as PNG files instead of popup windows

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import time
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

# %% [markdown]
# ## Step 2: Load MNIST Dataset
#
# MNIST: 70,000 grayscale 28×28 images, 10 digit classes (0–9)
# - Training: 60,000 images
# - Test:     10,000 images
#
# Normalisation constants (mean=0.1307, std=0.3081) are the known MNIST statistics.

# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./mnist_data', train=True,  download=True, transform=transform)
test_dataset  = torchvision.datasets.MNIST(root='./mnist_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=0)

print(f"Training samples : {len(train_dataset):,}")
print(f"Test samples     : {len(test_dataset):,}")
print(f"Image shape      : {train_dataset[0][0].shape}  (C x H x W)")
print(f"Number of classes: {len(train_dataset.classes)}")

# %% [markdown]
# ## Step 3: Visualise Sample Images

# %%
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    ax.imshow(img.squeeze().numpy(), cmap='gray')
    ax.set_title(f"Digit: {label}", fontsize=12)
    ax.axis('off')
plt.suptitle('Sample MNIST Images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('task1_mnist_samples.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: task1_mnist_samples.png")

# Class distribution
labels_all = [lbl for _, lbl in train_dataset]
counts = np.bincount(labels_all)
plt.figure(figsize=(8, 4))
plt.bar(range(10), counts, color='steelblue', edgecolor='black')
for i, c in enumerate(counts):
    plt.text(i, c + 50, str(c), ha='center', fontsize=9)
plt.xlabel('Digit Class'); plt.ylabel('Count')
plt.title('Class Distribution in Training Set', fontsize=13)
plt.xticks(range(10))
plt.tight_layout()
plt.savefig('task1_class_distribution.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 4: Define 5 CNN Architectures
#
# | Model | Depth | Filters        | Activation | BatchNorm |
# |-------|-------|----------------|------------|-----------|
# | A     | 2 conv | 32, 64        | ReLU       | No        |
# | B     | 2 conv | 64, 128       | ReLU       | No        |
# | C     | 3 conv | 32, 64, 128   | ReLU       | No        |
# | D     | 3 conv | 32, 64, 128   | ReLU       | Yes       |
# | E     | 3 conv | 32, 64, 128   | ELU        | Yes       |

# %%
class BaselineCNN(nn.Module):
    """Model A — 2 conv layers, 32/64 filters, ReLU, MaxPool (baseline)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28×28→28×28
        self.pool1 = nn.MaxPool2d(2, 2)                            # →14×14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # →14×14
        self.pool2 = nn.MaxPool2d(2, 2)                            # →7×7
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class LargerFiltersCNN(nn.Module):
    """Model B — same depth as A but doubled filter counts (64/128)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 7 * 7, 256)
        self.fc2   = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DeeperCNN(nn.Module):
    """Model C — 3 conv layers (32→64→128), deeper feature hierarchy."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)                            # after conv1+conv2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 7 * 7, 256)
        self.fc2   = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class BatchNormCNN(nn.Module):
    """Model D — 3 conv layers with Batch Normalisation before each ReLU."""
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv2d(1,  32, kernel_size=3, padding=1)
        self.bn1    = nn.BatchNorm2d(32)
        self.conv2  = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2    = nn.BatchNorm2d(64)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.conv3  = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3    = nn.BatchNorm2d(128)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.fc1    = nn.Linear(128 * 7 * 7, 256)
        self.bn_fc  = nn.BatchNorm1d(256)
        self.fc2    = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        return self.fc2(x)


class EluBatchNormCNN(nn.Module):
    """Model E — same as D but ELU activation (smooth negative gradients)."""
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv2d(1,  32, kernel_size=3, padding=1)
        self.bn1    = nn.BatchNorm2d(32)
        self.conv2  = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2    = nn.BatchNorm2d(64)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.conv3  = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3    = nn.BatchNorm2d(128)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.fc1    = nn.Linear(128 * 7 * 7, 256)
        self.bn_fc  = nn.BatchNorm1d(256)
        self.fc2    = nn.Linear(256, 10)

    def forward(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = self.pool1(F.elu(self.bn2(self.conv2(x))))
        x = self.pool2(F.elu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.elu(self.bn_fc(self.fc1(x)))
        return self.fc2(x)


# Print parameter counts
model_registry = {
    'A: Baseline':     BaselineCNN,
    'B: More Filters': LargerFiltersCNN,
    'C: Deeper':       DeeperCNN,
    'D: BatchNorm':    BatchNormCNN,
    'E: ELU+BN':       EluBatchNormCNN,
}
print(f"{'Model':<22} {'Parameters':>12}")
print("-" * 36)
for name, cls in model_registry.items():
    params = sum(p.numel() for p in cls().parameters() if p.requires_grad)
    print(f"{name:<22} {params:>12,}")

# %% [markdown]
# ## Step 5: Training and Evaluation Functions

# %%
def run_epoch(model, loader, optimizer, criterion, training=True):
    """Run one epoch — training or evaluation."""
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            _, pred = outputs.max(1)
            total   += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def train_model(model_class, model_name, epochs=10, lr=0.001):
    print(f"\n{'='*65}")
    print(f"  Training {model_name}")
    print(f"{'='*65}")
    model     = model_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history   = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    t0        = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, training=True)
        te_loss, te_acc = run_epoch(model, test_loader,  optimizer, criterion, training=False)
        for key, val in zip(history, [tr_loss, te_loss, tr_acc, te_acc]):
            history[key].append(val)
        print(f"  Epoch {epoch:2d}/{epochs} | "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.2f}% | "
              f"Test:  loss={te_loss:.4f} acc={te_acc:.2f}%")
    print(f"  Done in {time.time()-t0:.0f}s | Best Test Acc: {max(history['test_acc']):.2f}%")
    return model, history

# %% [markdown]
# ## Step 6: Run All 5 Experiments
#
# Each model trains for 10 epochs. Expected time: ~3–8 minutes per model on CPU.

# %%
EPOCHS = 10
results = {}

for name, cls in model_registry.items():
    results[name] = train_model(cls, name, epochs=EPOCHS)

# %% [markdown]
# ## Step 7: Compare All Models

# %%
# Summary table
print(f"\n{'='*60}")
print(f"{'Model':<22} {'Final Test Acc':>15} {'Best Test Acc':>15}")
print(f"{'='*60}")
for name, (_, h) in results.items():
    print(f"{name:<22} {h['test_acc'][-1]:>14.2f}% {max(h['test_acc']):>14.2f}%")
print(f"{'='*60}")

# %%
# Side-by-side loss and accuracy comparison
epochs_range = range(1, EPOCHS + 1)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for (name, (_, h)), c in zip(results.items(), colors):
    ax1.plot(epochs_range, h['test_loss'], label=name, color=c, linewidth=2)
    ax2.plot(epochs_range, h['test_acc'],  label=name, color=c, linewidth=2)

ax1.set_title('Test Loss per Epoch',     fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

ax2.set_title('Test Accuracy per Epoch', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.suptitle('CNN Architecture Comparison — MNIST', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('task1_model_comparison.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: task1_model_comparison.png")

# %%
# Individual train vs test curves (check for overfitting)
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
for col, (name, (_, h)) in enumerate(results.items()):
    # Loss row
    axes[0, col].plot(epochs_range, h['train_loss'], label='Train', color='royalblue')
    axes[0, col].plot(epochs_range, h['test_loss'],  label='Test',  color='tomato')
    axes[0, col].set_title(name, fontsize=9, fontweight='bold')
    axes[0, col].legend(fontsize=8); axes[0, col].grid(True, alpha=0.3)
    axes[0, col].set_xlabel('Epoch')
    # Accuracy row
    axes[1, col].plot(epochs_range, h['train_acc'], label='Train', color='royalblue')
    axes[1, col].plot(epochs_range, h['test_acc'],  label='Test',  color='tomato')
    axes[1, col].legend(fontsize=8); axes[1, col].grid(True, alpha=0.3)
    axes[1, col].set_xlabel('Epoch')

axes[0, 0].set_ylabel('Loss',         fontsize=11)
axes[1, 0].set_ylabel('Accuracy (%)', fontsize=11)
plt.suptitle('Train vs Test Curves per Architecture', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('task1_learning_curves.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: task1_learning_curves.png")

# %% [markdown]
# ## Step 8: Detailed Evaluation of the Best Model

# %%
best_name   = max(results, key=lambda k: max(results[k][1]['test_acc']))
best_model, best_history = results[best_name]
print(f"Best model         : {best_name}")
print(f"Best test accuracy : {max(best_history['test_acc']):.2f}%")

# Collect all predictions on the test set
best_model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        _, pred = best_model(images).max(1)
        all_preds.extend(pred.cpu().numpy())
        all_true.extend(labels.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)

# Confusion matrix
cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title(f'Confusion Matrix — {best_name}', fontsize=13, fontweight='bold')
plt.ylabel('True Label', fontsize=11)
plt.xlabel('Predicted Label', fontsize=11)
plt.tight_layout()
plt.savefig('task1_confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: task1_confusion_matrix.png")

# Classification report and F1
print("\nClassification Report:")
print(classification_report(all_true, all_preds,
                             target_names=[f"Digit {i}" for i in range(10)]))
macro_f1 = f1_score(all_true, all_preds, average='macro')
print(f"Macro F1 Score: {macro_f1:.4f}")

# %%
# Per-class accuracy bar chart
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
plt.figure(figsize=(9, 5))
bars = plt.bar(range(10), per_class_acc, color='steelblue', edgecolor='black')
for bar, acc in zip(bars, per_class_acc):
    plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
             f'{acc:.1f}%', ha='center', fontsize=9)
plt.xlabel('Digit Class', fontsize=11); plt.ylabel('Accuracy (%)', fontsize=11)
plt.title(f'Per-Class Accuracy — {best_name}', fontsize=13, fontweight='bold')
plt.xticks(range(10)); plt.ylim(min(per_class_acc) - 2, 101)
plt.tight_layout()
plt.savefig('task1_per_class_acc.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 9: Visualise Correct and Incorrect Predictions

# %%
best_model.eval()
correct_examples, wrong_examples = [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs  = best_model(images.to(device))
        _, preds = outputs.max(1)
        for img, pred, true in zip(images, preds.cpu(), labels):
            if pred == true and len(correct_examples) < 5:
                correct_examples.append((img.squeeze().numpy(), pred.item(), true.item()))
            elif pred != true and len(wrong_examples) < 5:
                wrong_examples.append((img.squeeze().numpy(), pred.item(), true.item()))
        if len(correct_examples) == 5 and len(wrong_examples) == 5:
            break

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, (img, pred, true) in enumerate(correct_examples):
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].set_title(f"Pred:{pred}  True:{true}", color='green', fontsize=10)
    axes[0, i].axis('off')
for i, (img, pred, true) in enumerate(wrong_examples):
    axes[1, i].imshow(img, cmap='gray')
    axes[1, i].set_title(f"Pred:{pred}  True:{true}", color='red', fontsize=10)
    axes[1, i].axis('off')
axes[0, 0].set_ylabel('Correct',   fontsize=12, color='green')
axes[1, 0].set_ylabel('Incorrect', fontsize=12, color='red')
plt.suptitle(f'Sample Predictions — {best_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('task1_predictions.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: task1_predictions.png")

# %% [markdown]
# ## Step 10: Final Summary

# %%
print("\n" + "=" * 65)
print("  TASK 1 COMPLETE — FINAL RESULTS SUMMARY")
print("=" * 65)
print(f"{'Model':<22} {'Best Test Acc':>13} {'Final Test Acc':>15}")
print("-" * 52)
for name, (_, h) in results.items():
    marker = " <-- BEST" if name == best_name else ""
    print(f"{name:<22} {max(h['test_acc']):>12.2f}% {h['test_acc'][-1]:>14.2f}%{marker}")
print("=" * 65)
print(f"\nBest model        : {best_name}")
print(f"Best test accuracy: {max(best_history['test_acc']):.2f}%")
print(f"Macro F1 score    : {macro_f1:.4f}")
print("\nOutput files saved:")
for f in ['task1_mnist_samples.png', 'task1_class_distribution.png',
          'task1_model_comparison.png', 'task1_learning_curves.png',
          'task1_confusion_matrix.png', 'task1_per_class_acc.png',
          'task1_predictions.png']:
    print(f"  {f}")

# %%
