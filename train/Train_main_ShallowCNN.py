#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

# ---------- MODEL DEFINITION: ShallowCNN (no down/up-sampling) ----------
class ShallowCNN(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),

            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features), 
            nn.ReLU(inplace=True),

            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),

            nn.Conv2d(features, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # mantiene altezza/larghezza invariate
        return self.net(x)

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
            aug = self.transform(image=data.transpose(1,2,0))
            data = aug["image"]               # (2, H, W)
        else:
            data = torch.from_numpy(data)
        # autoencoder: input == target
        return data, data

# ---------- TRAINING SETUP ----------
DATA_DIR    = '/app/processed_data'
RESULTS_DIR = './radargram_shallow/Train14-14pairs_cache_shallowCNN_1e-3_100_32'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparams
LR         = 1e-3
EPOCHS     = 100
BATCH_SIZE = 32
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1) Raccogli tutte le coppie (_2014, _2016)
# Le raccogliamo comunque in coppie così ti trovi il file test_pairs.json già pronto con entrambi i path
pairs = []
for exp in os.listdir(DATA_DIR):
    d = os.path.join(DATA_DIR, exp)
    if os.path.isdir(d):
        files = sorted(f for f in os.listdir(d) if f.endswith('_2014.npy'))
        for f in files:
            p1 = os.path.join(d, f)
            p2 = p1.replace('_2014.npy','_2016.npy')
            if os.path.exists(p2):
                pairs.append((p1,p2))
print(f"Total pairs: {len(pairs)}")

# 2) Split 60/20/20
n = len(pairs)
n_train = int(0.6*n)
n_val   = int(0.2*n)
n_test  = n - n_train - n_val
train_pairs, val_pairs, test_pairs = random_split(pairs, [n_train,n_val,n_test])

# Salva la lista di test per il futuro (qui avrai sia i path del 2014 che del 2016 per quando farai testing)
with open(os.path.join(RESULTS_DIR,'test_pairs.json'),'w') as f:
    json.dump([list(x) for x in test_pairs], f, indent=2)

# 3) Trasformazioni (solo ToTensor)
transform = A.Compose([ToTensorV2()])

train_ds = RadarDataset(train_pairs, transform)
val_ds   = RadarDataset(val_pairs,   transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

# 4) Model / Optim / Scheduler / Loss
model     = ShallowCNN(in_channels=2, out_channels=2).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.MSELoss()

# 5) Training loop
best_val = float('inf')
train_losses, val_losses = [], []

for epoch in range(1, EPOCHS+1):
    #  train
    model.train()
    running, count = 0.0, 0
    for x,y in train_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
        count   += x.size(0)
    train_loss = running / count
    train_losses.append(train_loss)

    #  validate
    model.eval()
    running, count = 0.0, 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            running += loss.item() * x.size(0)
            count   += x.size(0)
    val_loss = running / count
    val_losses.append(val_loss)

    print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
    scheduler.step()

    # salva best
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR,'best_model.pth'))
        print("** Best model saved **")

# 6) Plot delle loss
plt.figure(figsize=(8,5))
plt.plot(range(1,EPOCHS+1), train_losses, label='Train')
plt.plot(range(1,EPOCHS+1), val_losses,   label='Val')
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,'loss_plot.png'))
plt.close()

# 7) Esempi da val set
os.makedirs(os.path.join(RESULTS_DIR,'examples'), exist_ok=True)
model.eval()
inputs, targets = next(iter(val_loader))
with torch.no_grad():
    outputs = model(inputs.to(DEVICE)).cpu().numpy()
inputs = inputs.numpy()
for i in range(min(6, inputs.shape[0])):
    fig, axs = plt.subplots(1,3,figsize=(12,4))
    axs[0].imshow(inputs[i,0], cmap='jet'); axs[0].set_title("In Ch0 (2014)"); axs[0].axis('off')
    axs[1].imshow(outputs[i,0],cmap='jet'); axs[1].set_title("Out Ch0");axs[1].axis('off')
    diff = np.abs(outputs[i,0] - inputs[i,0])
    axs[2].imshow(diff,     cmap='hot'); axs[2].set_title("Diff");  axs[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,'examples',f'example_{i}.png'))
    plt.close()

print(f"\nTraining complete! Best Val Loss: {best_val:.6f}")
print(f"Results directory: {RESULTS_DIR}")
