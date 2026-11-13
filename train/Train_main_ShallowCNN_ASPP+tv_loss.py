#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train — ShallowASPP_Refined con TV Loss sulle feature interne
=============================================================
Loss totale:
    L_total = L_recon  +  LAMBDA_TV × L_TV

Dove L_TV è calcolata sull'output del blocco `refinement` (vedi architettura),
catturato tramite un forward hook → i gradienti fluiscono correttamente.

Struttura:
    ASPP → [Refinement (Conv-BN-ReLU × 2)] --TV--> Conv 1×1 → output
                        └─── L_TV applicata qui
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.aspp import ASPP
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------- MODEL DEFINITION: ShallowCNN (no down/up-sampling) ----------
class ShallowASPP_Refined(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, features=64):
        super().__init__()

        # Modulo per l'estrazione multi-scala
        self.aspp = ASPP(
            in_channels=in_channels,
            out_channels=features,
            atrous_rates=[6, 12, 18]
        )

        # Modulo di rifinitura locale (due layer come nel tuo Shallow originale)
        self.refinement = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),

            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

        # Layer finale
        self.classifier = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # 1. Guarda l'immagine a tutte le scale (ASPP)
        x = self.aspp(x)
        # 2. Rifinisci localmente i dettagli
        x = self.refinement(x)
        # 3. Ritorna ai 2 canali
        out = self.classifier(x)
        return out


# ---------- TV LOSS sulle feature map interne ----------
def tv_loss(features: torch.Tensor) -> torch.Tensor:
    """
    Total Variation Loss: penalizza le discontinuità spaziali nelle feature map.
    Applicata DOPO il blocco refinement (prima del Conv 1×1 finale),
    non sull'output finale.

    Args:
        features: (B, C, H, W)  —  feature map interna con gradienti attivi
    Returns:
        Scalare: media di |Δh| + |Δw|
    """
    diff_h = features[:, :, 1:, :] - features[:, :, :-1, :]
    diff_w = features[:, :, :, 1:] - features[:, :, :, :-1]
    return torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_w))


# ---------- DATASET DEFINITION ----------
class RadarDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2 = self.pairs[idx]
        # Carichiamo SOLO la patch del 2014
        img1 = np.load(p1).astype(np.float32)

        # Duplichiamo la patch del 2014 per riempire i 2 canali richiesti
        data = np.stack([img1, img1], axis=0)  # (2, H, W)

        if self.transform:
            # Albumentations vuole HWC
            aug = self.transform(image=data.transpose(1, 2, 0))
            data = aug["image"]               # (2, H, W)
        else:
            data = torch.from_numpy(data)
        # autoencoder: input == target
        return data, data


# ---------- TRAINING SETUP ----------
DATA_DIR    = '/app/processed_data'
LR          = 1e-3
EPOCHS      = 100
BATCH_SIZE  = 8
LAMBDA_TV   = 1e-3          # peso della TV loss — regola qui (suggerito: 1e-3 ÷ 1e-2)
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

RESULTS_DIR = f'./radargram_shallow/ASPP_TV_lr{LR}_ep{EPOCHS}_bs{BATCH_SIZE}'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1) Raccogli tutte le coppie (_2014, _2016)
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

# 2) Split 60/20/20
n       = len(pairs)
n_train = int(0.6 * n)
n_val   = int(0.2 * n)
n_test  = n - n_train - n_val
train_pairs, val_pairs, test_pairs = random_split(pairs, [n_train, n_val, n_test])

with open(os.path.join(RESULTS_DIR, 'test_pairs.json'), 'w') as fh:
    json.dump([list(x) for x in test_pairs], fh, indent=2)

# 3) Trasformazioni e DataLoader
transform    = A.Compose([ToTensorV2()])
train_ds     = RadarDataset(train_pairs, transform)
val_ds       = RadarDataset(val_pairs,   transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

# 4) Model / Forward Hook / Optim / Scheduler / Loss
model = ShallowASPP_Refined(in_channels=2, out_channels=2).to(DEVICE)

# ── Forward hook sul blocco refinement ──────────────────────────────────────
# Cattura l'output del refinement CON gradienti attivi.
# Il tensore viene sovrascritto ad ogni forward → nessun leak di memoria.
_refinement_feat: dict = {}

def _refinement_hook(module, inp, output):
    _refinement_feat['out'] = output   # (B, features, H, W) con grad_fn

model.refinement.register_forward_hook(_refinement_hook)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.MSELoss()

# 5) Training loop
best_val = float('inf')
history  = {'train_total': [], 'train_recon': [], 'train_tv': [],
            'val_total':   [], 'val_recon':   [], 'val_tv':   []}

for epoch in range(1, EPOCHS + 1):

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    model.train()
    run_total = run_recon = run_tv = count = 0.0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        out = model(x)                              # forward → hook scatta qui

        # Loss di ricostruzione (sull'output finale)
        loss_recon = criterion(out, y)

        # TV loss sull'output del refinement (feature interne)
        loss_tv = tv_loss(_refinement_feat['out'])

        loss = loss_recon + LAMBDA_TV * loss_tv
        loss.backward()
        optimizer.step()

        b = x.size(0)
        run_total += loss.item()       * b
        run_recon += loss_recon.item() * b
        run_tv    += loss_tv.item()    * b
        count     += b

    history['train_total'].append(run_total / count)
    history['train_recon'].append(run_recon / count)
    history['train_tv'].append(run_tv    / count)

    # ── VALIDATE ──────────────────────────────────────────────────────────────
    model.eval()
    run_total = run_recon = run_tv = count = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)                          # hook scatta anche qui

            loss_recon = criterion(out, y)
            loss_tv    = tv_loss(_refinement_feat['out'])
            loss       = loss_recon + LAMBDA_TV * loss_tv

            b = x.size(0)
            run_total += loss.item()       * b
            run_recon += loss_recon.item() * b
            run_tv    += loss_tv.item()    * b
            count     += b

    history['val_total'].append(run_total / count)
    history['val_recon'].append(run_recon / count)
    history['val_tv'].append(run_tv    / count)

    print(
        f"Epoch {epoch:>3}/{EPOCHS} | "
        f"Train  total={history['train_total'][-1]:.6f} "
        f"recon={history['train_recon'][-1]:.6f} "
        f"tv={history['train_tv'][-1]:.6f} | "
        f"Val  total={history['val_total'][-1]:.6f} "
        f"recon={history['val_recon'][-1]:.6f} "
        f"tv={history['val_tv'][-1]:.6f}"
    )

    scheduler.step()

    # salva best model (basato su val_total)
    if history['val_total'][-1] < best_val:
        best_val = history['val_total'][-1]
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pth'))
        print("  ** Best model saved **")

# 6) Plot delle loss
epochs_range = range(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(epochs_range, history['train_total'], label='Train')
axes[0].plot(epochs_range, history['val_total'],   label='Val')
axes[0].set_title('Loss Totale'); axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(epochs_range, history['train_recon'], label='Train')
axes[1].plot(epochs_range, history['val_recon'],   label='Val')
axes[1].set_title('L_recon (MSE)'); axes[1].set_xlabel('Epoch')
axes[1].legend()

axes[2].plot(epochs_range, history['train_tv'], label='Train')
axes[2].plot(epochs_range, history['val_tv'],   label='Val')
axes[2].set_title(f'L_TV  (λ={LAMBDA_TV})'); axes[2].set_xlabel('Epoch')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'loss_plot.png'), dpi=150)
plt.close()

# 7) Esempi da val set
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
