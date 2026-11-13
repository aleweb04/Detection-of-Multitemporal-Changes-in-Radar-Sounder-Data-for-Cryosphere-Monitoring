#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train — ShallowASPP_BiLSTM con sola MSE Loss
=============================================
Loss totale:
    L_total = L_recon  (MSE)

Struttura:
    ASPP → Conv1 → Conv2 → Conv3 → Bi-ConvLSTM → CNN 1×1 → output
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Importa il modello con Bi-ConvLSTM ────────────────────────────────────────
from models.model_biconvlstm_aspp import ShallowASPP_BiLSTM


# ── Dataset ───────────────────────────────────────────────────────────────────
class RadarDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2 = self.pairs[idx]
        img1 = np.load(p1).astype(np.float32)
        data = np.stack([img1, img1], axis=0)          # (2, H, W)

        if self.transform:
            aug  = self.transform(image=data.transpose(1, 2, 0))
            data = aug["image"]                        # (2, H, W) tensor
        else:
            data = torch.from_numpy(data)

        # autoencoder: input == target (training su t1 solo)
        return data, data


# ── Hyperparametri ────────────────────────────────────────────────────────────
DATA_DIR   = '/app/processed_data'
LR         = 1e-3
EPOCHS     = 60
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

RESULTS_DIR = f'./radargram_shallow/ASPP_BiLSTM_lr{LR}_ep{EPOCHS}_bs{BATCH_SIZE}'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Raccolta coppie (_2014, _2016) ────────────────────────────────────────────
pairs = []
for exp in os.listdir(DATA_DIR):
    d = os.path.join(DATA_DIR, exp)
    if os.path.isdir(d):
        files = sorted(f for f in os.listdir(d) if f.endswith('_2014.npy'))
        for f in files:
            p1 = os.path.join(d, f)
            p2 = p1.replace('_2014.npy', '_2016.npy')
            if os.path.exists(p2):
                pairs.append((p1, p2))
print(f"Total pairs: {len(pairs)}")

# ── Split 60/20/20 ────────────────────────────────────────────────────────────
n       = len(pairs)
n_train = int(0.6 * n)
n_val   = int(0.2 * n)
n_test  = n - n_train - n_val
train_pairs, val_pairs, test_pairs = random_split(pairs, [n_train, n_val, n_test])

with open(os.path.join(RESULTS_DIR, 'test_pairs.json'), 'w') as fh:
    json.dump([list(x) for x in test_pairs], fh, indent=2)

# ── DataLoader ────────────────────────────────────────────────────────────────
transform    = A.Compose([ToTensorV2()])
train_ds     = RadarDataset(train_pairs, transform)
val_ds       = RadarDataset(val_pairs,   transform)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=0
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=0
)

# ── Modello ───────────────────────────────────────────────────────────────────
model = ShallowASPP_BiLSTM(in_channels=2, out_channels=2, features=64).to(DEVICE)

if DEVICE == 'cuda':
    print("CUDA device in use. If you hit OOMs consider:\n"
          " - Lowering BATCH_SIZE (export BATCH_SIZE=2)\n"
          " - Enabling mixed precision training (AMP)\n"
          " - Setting PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:64 or expandable_segments:True\n"
          " - torch.cuda.empty_cache() between large allocations")

# ── Ottimizzatore / Scheduler / Criterio ─────────────────────────────────────
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.MSELoss()

# ── Training loop ─────────────────────────────────────────────────────────────
best_val = float('inf')
history  = {'train': [], 'val': []}

for epoch in range(1, EPOCHS + 1):

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    model.train()
    run_loss = count = 0.0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        run_loss += loss.item() * x.size(0)
        count    += x.size(0)

    history['train'].append(run_loss / count)

    # ── VALIDATE ──────────────────────────────────────────────────────────────
    model.eval()
    run_loss = count = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out  = model(x)
            loss = criterion(out, y)

            run_loss += loss.item() * x.size(0)
            count    += x.size(0)

    history['val'].append(run_loss / count)

    print(
        f"Epoch {epoch:>3}/{EPOCHS} | "
        f"Train Loss: {history['train'][-1]:.6f} | "
        f"Val Loss:   {history['val'][-1]:.6f}"
    )

    scheduler.step()

    # salva best model
    if history['val'][-1] < best_val:
        best_val = history['val'][-1]
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pth'))
        print("  ** Best model saved **")

# ── Plot delle loss ───────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), history['train'], label='Train')
plt.plot(range(1, EPOCHS + 1), history['val'],   label='Val')
plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)')
plt.title('Loss Totale')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'loss_plot.png'), dpi=150)
plt.close()

# ── Esempi da val set ─────────────────────────────────────────────────────────
os.makedirs(os.path.join(RESULTS_DIR, 'examples'), exist_ok=True)
model.eval()
inputs, targets = next(iter(val_loader))
with torch.no_grad():
    outputs = model(inputs.to(DEVICE)).cpu().numpy()
inputs = inputs.numpy()

for i in range(min(6, inputs.shape[0])):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(inputs[i, 0],  cmap='jet'); axs[0].set_title('In Ch0 (2014)'); axs[0].axis('off')
    axs[1].imshow(outputs[i, 0], cmap='jet'); axs[1].set_title('Out Ch0');        axs[1].axis('off')
    diff = np.abs(outputs[i, 0] - inputs[i, 0])
    axs[2].imshow(diff,          cmap='hot'); axs[2].set_title('Diff');            axs[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'examples', f'example_{i}.png'))
    plt.close()

print(f"\nTraining complete!  Best Val Loss: {best_val:.6f}")
print(f"Results: {RESULTS_DIR}")
