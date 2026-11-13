# -*- coding: utf-8 -*-
import os, glob, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt

import torch.nn as nn


from utils.aspp import ASPP

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


# --- Configurazione ---
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR    = os.getcwd()
DATA_DIR    = os.path.join(BASE_DIR, "processed_data", "esp3")
CSV_PATH    = os.path.join(DATA_DIR, "patches_metadata.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "test_results_shallowCNN_FINALE")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Carica modello ShallowCNN ---
model = ShallowASPP_Refined(in_channels=2, out_channels=2, features=64).to(DEVICE)
model.load_state_dict(torch.load(
    os.path.join(BASE_DIR, "ASPP_cache_shallowCNN_1e-3_64_8", "best_model.pth"),
    map_location=DEVICE
))
model.eval()

# --- Carica metadata e prendi primi 6 ID unici ---
df = pd.read_csv(CSV_PATH)
ids = df['id'].unique()[:6]

summary = []
for pid in ids:
    # carica patch 2014 e 2016
    p14 = np.load(os.path.join(DATA_DIR, f"{pid}_2014.npy"))
    p16 = np.load(os.path.join(DATA_DIR, f"{pid}_2016.npy"))

    # inference 2014
    inp14 = torch.from_numpy(np.stack([p14, p14],0)[None]).float().to(DEVICE)
    with torch.no_grad():
        out14 = model(inp14).squeeze(0).cpu().numpy()  # (2,H,W)

    # inference 2016
    inp16 = torch.from_numpy(np.stack([p16, p16],0)[None]).float().to(DEVICE)
    with torch.no_grad():
        out16 = model(inp16).squeeze(0).cpu().numpy()

    # calcola diffs tra canali di output
    diff14 = np.abs(out14[1] - out14[0])
    diff16 = np.abs(out16[1] - out16[0])

    # salva metriche
    summary.append({
        'id': pid,
        'mse_diff14': float((diff14**2).mean()),
        'mse_diff16': float((diff16**2).mean()),
    })

    # disegna plot 2�5
    fig, ax = plt.subplots(2, 5, figsize=(25, 10))
    # Row 0 = 2014
    arrs0 = [
        (p14,        "In2014 Ch0",  'jet'),
        (p14,        "In2014 Ch1",  'jet'),
        (out14[0],   "Out2014 Ch0", 'jet'),
        (out14[1],   "Out2014 Ch1", 'jet'),
        (diff14,     "Diff14 |C1-C0|", 'hot'),
    ]
    for col, (img, ttl, cmap) in enumerate(arrs0):
        im = ax[0, col].imshow(img, cmap=cmap, aspect='auto')
        ax[0, col].set_title(ttl)
        ax[0, col].axis('off')
        fig.colorbar(im, ax=ax[0, col])

    # Row 1 = 2016
    arrs1 = [
        (p16,        "In2016 Ch0",  'jet'),
        (p16,        "In2016 Ch1",  'jet'),
        (out16[0],   "Out2016 Ch0", 'jet'),
        (out16[1],   "Out2016 Ch1", 'jet'),
        (diff16,     "Diff16 |C1-C0|", 'hot'),
    ]
    for col, (img, ttl, cmap) in enumerate(arrs1):
        im = ax[1, col].imshow(img, cmap=cmap, aspect='auto')
        ax[1, col].set_title(ttl)
        ax[1, col].axis('off')
        fig.colorbar(im, ax=ax[1, col])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{pid}.png"))
    plt.close(fig)

# salva CSV riassuntivo
pd.DataFrame(summary).to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
print(f"Test completati. Risultati in {RESULTS_DIR}")
