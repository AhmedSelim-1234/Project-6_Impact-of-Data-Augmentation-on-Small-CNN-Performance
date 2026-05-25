"""
lr_sensitivity_analysis.py
==========================
Hyperparameter Sensitivity Analysis — Learning Rate
Baseline: 'none' augmentation (no augmentation, original LR = 0.001)

Experiments
-----------
  Exp 1 (Baseline) : LR = 0.001  | augmentation = 'none'
  Exp 2 (Lower LR) : LR = 0.0001 | augmentation = 'none'
  Exp 3 (Higher LR): LR = 0.01   | augmentation = 'none'
  Exp 4 (Very High): LR = 0.05   | augmentation = 'none'

Why these values?
  • 0.0001 → conservative; often under-fits within 30 epochs
  • 0.001  → your original setting (baseline)
  • 0.01   → 10× baseline; tests faster convergence vs instability
  • 0.05   → aggressive; likely diverges or oscillates — useful error-analysis case

All other hyper-parameters are frozen (same as train.py):
  BATCH_SIZE   = 64
  WEIGHT_DECAY = 0.0005
  NUM_EPOCHS   = 30
  DATA         = 100 % of CIFAR-10
"""

import numpy as np
import copy
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt

from simple_model import SimpleCNN
from dataset import get_dataloader

# ─────────────────────────────────────────
# Device
# ─────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─────────────────────────────────────────
# Frozen hyper-parameters
# ─────────────────────────────────────────
NUM_EPOCHS   = 30
BATCH_SIZE   = 64
WEIGHT_DECAY = 0.0005
NUM_CLASSES  = 10
PERCENTAGE   = 1.0          # use full dataset for a fair comparison
AUGMENTATION = 'none'       # baseline augmentation — keep constant

# ─────────────────────────────────────────
# Learning-rate grid  (≥ 3 experiments required)
# ─────────────────────────────────────────
LR_EXPERIMENTS = {
    'LR=0.0001 (lower)' : 0.0001,   # Exp 2
    'LR=0.001  (baseline)': 0.001,  # Exp 1  ← baseline
    'LR=0.01   (higher)' : 0.01,    # Exp 3
    'LR=0.05   (very high)': 0.05,  # Exp 4
}
BASELINE_KEY = 'LR=0.001  (baseline)'

# ─────────────────────────────────────────
# Reuse train / validation helpers from train.py
# ─────────────────────────────────────────
def train_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(train_loader.dataset)


def validate_epoch(model, val_loader, loss_fn):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(val_loader.dataset), 100.0 * correct / total


def training_loop(model, train_loader, val_loader, loss_fn, optimizer):
    model.to(device)
    best_acc, best_state, best_epoch = 0.0, None, 0
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(NUM_EPOCHS):
        t_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        v_loss, v_acc = validate_epoch(model, val_loader, loss_fn)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        val_accs.append(v_acc)
        print(f"  Epoch [{epoch+1:>2}/{NUM_EPOCHS}]  "
              f"Train Loss: {t_loss:.4f}  "
              f"Val Loss: {v_loss:.4f}  "
              f"Val Acc: {v_acc:.2f}%")
        if v_acc > best_acc:
            best_acc   = v_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    if best_state:
        print(f"  ↳ Best = {best_acc:.2f}% @ epoch {best_epoch}")
        model.load_state_dict(best_state)
    return model, [train_losses, val_losses, val_accs]


# ─────────────────────────────────────────
# Run all LR experiments
# ─────────────────────────────────────────
os.makedirs('results/lr_sensitivity', exist_ok=True)

all_metrics = {}   # label → [train_losses, val_losses, val_accs]
all_best    = {}   # label → best accuracy

train_loader, val_loader = get_dataloader(
    PERCENTAGE,
    batch_size=BATCH_SIZE,
    augmentation_combo_name=AUGMENTATION,
)

for label, lr in LR_EXPERIMENTS.items():
    print(f"\n{'='*55}")
    print(f"  Experiment: {label}")
    print(f"{'='*55}")

    model     = SimpleCNN(num_classes=NUM_CLASSES)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    _, metrics = training_loop(model, train_loader, val_loader, loss_fn, optimizer)

    best_acc = max(metrics[2])
    all_metrics[label] = metrics
    all_best[label]    = best_acc

    # Persist to disk
    safe_key = label.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '')
    with open(f'results/lr_sensitivity/{safe_key}.json', 'w') as f:
        json.dump({
            'lr': lr, 'label': label,
            'train_losses': metrics[0],
            'val_losses':   metrics[1],
            'val_accs':     metrics[2],
            'best_accuracy': best_acc,
        }, f, indent=2)
    print(f"  Saved → results/lr_sensitivity/{safe_key}.json")


# ─────────────────────────────────────────
# ① Learning Curves — all LRs on one plot
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
epochs = range(1, NUM_EPOCHS + 1)

for (label, metrics), color in zip(all_metrics.items(), colors):
    ls = '--' if label == BASELINE_KEY else '-'
    lw = 2.5  if label == BASELINE_KEY else 1.8
    axes[0].plot(epochs, metrics[0], color=color, ls=ls, lw=lw, label=label)
    axes[1].plot(epochs, metrics[2], color=color, ls=ls, lw=lw, label=label)

axes[0].set_title('Training Loss vs Epoch',     fontsize=13, fontweight='bold')
axes[1].set_title('Validation Accuracy vs Epoch', fontsize=13, fontweight='bold')
for ax in axes:
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
axes[0].set_ylabel('Loss')
axes[1].set_ylabel('Accuracy (%)')

plt.suptitle('Learning Rate Sensitivity Analysis\n'
             f'(Baseline: none augmentation, fixed at 100% data)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/lr_sensitivity/lr_learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved → results/lr_sensitivity/lr_learning_curves.png")


# ─────────────────────────────────────────
# ② Bar chart — best accuracy per LR
# ─────────────────────────────────────────
labels = list(all_best.keys())
accs   = list(all_best.values())
bar_colors = [('#3498db' if l == BASELINE_KEY else '#bdc3c7') for l in labels]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, accs, color=bar_colors, edgecolor='black', linewidth=0.7, width=0.5)

for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

baseline_acc = all_best[BASELINE_KEY]
ax.axhline(baseline_acc, color='#3498db', ls='--', lw=1.5, label=f'Baseline ({baseline_acc:.2f}%)')
ax.set_title('Best Validation Accuracy by Learning Rate', fontsize=13, fontweight='bold')
ax.set_ylabel('Best Accuracy (%)')
ax.set_ylim(max(0, min(accs) - 5), min(100, max(accs) + 5))
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=15, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig('results/lr_sensitivity/lr_bar_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved → results/lr_sensitivity/lr_bar_chart.png")


# ─────────────────────────────────────────
# ③ Delta table — comparison vs baseline
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("LEARNING RATE SENSITIVITY — COMPARISON VS BASELINE")
print("=" * 60)
print(f"{'Experiment':<30} {'Best Acc':>10} {'Δ vs Baseline':>15} {'Verdict':>12}")
print("-" * 60)

for label, acc in all_best.items():
    delta   = acc - baseline_acc
    verdict = 'BASELINE' if label == BASELINE_KEY else \
              ('↑ Better'  if delta > 0.5  else
               '↓ Worse'   if delta < -0.5 else
               '≈ Similar')
    print(f"  {label:<28} {acc:>8.2f}%  {delta:>+10.2f}%  {verdict:>10}")
print("=" * 60)


# ─────────────────────────────────────────
# ④ Error analysis — per-epoch instability
# ─────────────────────────────────────────
print("\n─── Error / Instability Analysis ───")
for label, metrics in all_metrics.items():
    accs_arr = np.array(metrics[2])
    # Oscillation: std of accuracy changes epoch-to-epoch
    delta_accs = np.diff(accs_arr)
    instability = np.std(delta_accs)
    # Peak-to-final gap: did model regress after its peak?
    gap = max(accs_arr) - accs_arr[-1]
    print(f"  {label}")
    print(f"    Best  acc : {max(accs_arr):.2f}%   "
          f"Final acc: {accs_arr[-1]:.2f}%   "
          f"Peak→Final gap: {gap:.2f}%")
    print(f"    Epoch-to-epoch std (instability): {instability:.4f}")
    if instability > 1.5:
        print(f"    ⚠ High oscillation detected — LR may be too large")
    if gap > 3.0:
        print(f"    ⚠ Model regressed after peak — consider LR scheduling")
    print()

# Save summary JSON
with open('results/lr_sensitivity/summary.json', 'w') as f:
    json.dump(all_best, f, indent=2)
print("Summary saved → results/lr_sensitivity/summary.json")
