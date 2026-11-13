# -*- coding: utf-8 -*-
"""
ASPP_SSIM_CVA_kmeans_overlay.py

Combina:
  - Modello ShallowASPP_Refined (ASPP + refinement, come auto_AMP_color_ASPP)
  - Stitching globale feature
  - Selezione canali alla S²CVA (varianza → SSIM reciprocità), ma SOLO come filtro
  - Proiezione CVA classica (PCA a 2 componenti) sui canali selezionati
  - KMeans auto-k con Silhouette Score
  - Overlay finale a colori su patch stitchate

Output:
  - PNG overlay (immagine 2014 + cluster colorati)
  - PNG overlay modulo CVA (tutti i pixel, scala di grigi)
  - PNG scatter polare (cluster + centroidi)
  - PNG scatter densità KDE (LogNorm + PowerNorm)
  - .npy centroidi
  - Istogrammi selezioni canali (varianza + SSIM)
"""

import os
import glob
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, PowerNorm
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

import CVA_config
from models.utils.aspp import ASPP


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRI
# ─────────────────────────────────────────────────────────────────────────────
N_VARIANCE = 16        # canali trattenuti dopo step varianza
N_FINAL = 4            # canali finali dopo SSIM (filtro, non proiezione)
AMPLITUDE_THRESHOLD = 0.5  # soglia sul modulo CVA normalizzato per overlay
PATCH_SIZE = 256
OVERLAP = 128
MAX_PATCHES = 64

CLUSTER_COLORS = [
    (1.0, 0.0, 0.0, 0.6), (0.0, 0.5, 1.0, 0.6), (0.0, 0.8, 0.2, 0.6),
    (1.0, 0.8, 0.0, 0.6), (0.8, 0.0, 0.8, 0.6), (0.0, 0.9, 0.9, 0.6),
    (1.0, 0.5, 0.0, 0.6), (0.5, 0.5, 0.5, 0.6)
]


# ─────────────────────────────────────────────────────────────────────────────
# MODELLO (da scatterCVA_kmeans_patchwise_TRY / auto_AMP_color_ASPP)
# ─────────────────────────────────────────────────────────────────────────────

class ShallowASPP_Refined(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, features=64):
        super().__init__()
        self.aspp = ASPP(in_channels=in_channels, out_channels=features,
                         atrous_rates=[6, 12, 18])
        self.refinement = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.aspp(x)
        x = self.refinement(x)
        return self.classifier(x)


def extract_features_aspp(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    feat_aspp = model.aspp(x)
    features = model.refinement(feat_aspp)
    return features[0].detach().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm2d(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def stitch_patches(patches: List[np.ndarray], patch_size: int = 256,
                   overlap: int = 128) -> np.ndarray:
    if not patches:
        raise ValueError("Nessuna patch da stitchare")

    C, H, W = patches[0].shape
    step = patch_size - overlap
    n_patches = len(patches)

    total_width = (n_patches - 1) * step + patch_size
    big_image = np.zeros((C, H, total_width), dtype=np.float32)

    for i, patch in enumerate(patches):
        start = i * step
        big_image[:, :, start:start + patch_size] += patch

        if i > 0:
            overlap_region = big_image[:, :, start:start + overlap]
            overlap_region /= 2.0

    return big_image


# ─────────────────────────────────────────────────────────────────────────────
# SELEZIONE CANALI (filtro alla S²CVA, ma la proiezione è CVA classica)
# ─────────────────────────────────────────────────────────────────────────────

def select_variance_channels_global(diff: np.ndarray,
                                    n_retained: int) -> Tuple[np.ndarray, np.ndarray]:
    C = diff.shape[0]
    diff_flat = diff.reshape(C, -1)
    variances = np.var(diff_flat, axis=1)

    k = min(n_retained, C)
    top_idx = np.argsort(variances)[-k:]

    return np.sort(top_idx), variances


def select_channels_by_ssim_diversity(diff: np.ndarray, candidate_idx: np.ndarray,
                                       n_final: int) -> Tuple[np.ndarray, np.ndarray]:
    """Filtra i canali candidati: tiene i piu' diversi tra loro (minor SSIM reciproca).
    Questo è lo STEP 2 della S²CVA, usato qui come filtro prima della CVA."""
    n_cand = len(candidate_idx)
    similarity_matrix = np.zeros((n_cand, n_cand))

    for i in range(n_cand):
        for j in range(n_cand):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                di = _norm2d(diff[candidate_idx[i]].astype(np.float64))
                dj = _norm2d(diff[candidate_idx[j]].astype(np.float64))
                similarity_matrix[i, j] = ssim(di, dj, data_range=1.0, win_size=3)

    avg_similarity = similarity_matrix.mean(axis=1)
    diversity_scores = 1.0 - avg_similarity

    thresh = threshold_otsu(diversity_scores)
    otsu_idx = np.where(diversity_scores >= thresh)[0]

    if len(otsu_idx) > n_final:
        otsu_idx = np.argsort(diversity_scores)[-n_final:]

    return np.sort(candidate_idx[otsu_idx]), diversity_scores


# ─────────────────────────────────────────────────────────────────────────────
# CVA PROJECTION (PCA a 2 componenti)
# ─────────────────────────────────────────────────────────────────────────────

def cva_project(diff_sel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    B, N = diff_sel.shape
    flat = diff_sel.T  # (N, B)
    coords2 = PCA(n_components=2).fit_transform(flat)
    dx = coords2[:, 0]
    dy = coords2[:, 1]

    module = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    return dx, dy, module, angle


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-K KMEANS
# ─────────────────────────────────────────────────────────────────────────────

def auto_kmeans_clustering(xy_all: np.ndarray):
    print(f"  KMeans AUTO-K su {len(xy_all)} punti...")

    nn = NearestNeighbors(n_neighbors=15).fit(xy_all)
    dists, _ = nn.kneighbors(xy_all)
    density_mask = np.min(dists[:, 1:], axis=1) < np.percentile(dists[:, 1:], 30)
    xy_dense = xy_all[density_mask]
    print(f"  Punti densi: {len(xy_dense)} / {len(xy_all)}")

    best_k = 4
    best_score = -1
    best_centers = None
    best_labels = None

    for k in range(3, 8):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=42)
        labels_k = kmeans.fit_predict(xy_dense)
        score = silhouette_score(xy_dense, labels_k)
        print(f"    k={k}  silhouette={score:.3f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_centers = kmeans.cluster_centers_
            best_labels = labels_k

    print(f"  Miglior K: {best_k} (silhouette={best_score:.3f})")

    labels_full = np.full(len(xy_all), -1)
    labels_full[density_mask] = best_labels

    return best_k, best_centers, best_labels, labels_full, best_score, density_mask


# ─────────────────────────────────────────────────────────────────────────────
# ISTOGRAMMI
# ─────────────────────────────────────────────────────────────────────────────

def plot_variance_histogram(variances, selected_idx, name, out_dir):
    C = len(variances)
    order = np.argsort(variances)
    sorted_variances = variances[order]
    selected_set = set(selected_idx)
    colors = ["#E8593C" if ch in selected_set else "#B4B2A9" for ch in order]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(C), sorted_variances, color=colors, width=0.85)
    ax.set_xticks(range(C))
    ax.set_xticklabels(order, rotation=90, fontsize=8)
    ax.set_xlabel("Numero Feature (ID Canale)")
    ax.set_ylabel("Varianza")
    ax.set_title(f"Varianza globale (rosso = feature filtrate) — {name}")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"hist_variance_global_{name}.png"), dpi=150)
    plt.close(fig)


def plot_ssim_histogram(change_scores, candidate_idx, selected_idx, name, out_dir):
    order = np.argsort(change_scores)
    sorted_scores = change_scores[order]
    sorted_candidates = candidate_idx[order]
    selected_set = set(selected_idx)
    colors = ["#1D9E75" if ch in selected_set else "#B4B2A9" for ch in sorted_candidates]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(change_scores)), sorted_scores, color=colors, width=0.7)
    ax.set_xticks(range(len(change_scores)))
    ax.set_xticklabels(sorted_candidates, fontsize=9)
    ax.set_xlabel("Numero Feature Candidata (ID Canale)")
    ax.set_ylabel("1 − SSIM_diff (change score)")
    ax.set_title(f"SSIM diversita' (verde = {len(selected_idx)} canali finali) — {name}")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"hist_ssim_diff_global_{name}.png"), dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SCATTER POLARE
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter_polar(modules_all, angles_all, labels_full, centers_xy,
                       cluster_colors, n_clusters, silhouette, out_path):
    cx, cy = centers_xy[:, 0], centers_xy[:, 1]
    mod_c = np.hypot(cx, cy)
    ang_c = np.arctan2(cy, cx)

    radii = np.array([
        np.std(modules_all[labels_full == k]) if np.any(labels_full == k) else 0.0
        for k in range(n_clusters)
    ])

    colors_plot = [(r, g, b) for r, g, b, _ in cluster_colors]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='polar')

    for k in range(n_clusters):
        mask_k = labels_full == k
        ax.scatter(angles_all[mask_k], modules_all[mask_k],
                   s=3, alpha=0.35, color=colors_plot[k], label=f'Cluster {k}')

    for k in range(n_clusters):
        r_inner = max(0, mod_c[k] - radii[k])
        r_outer = mod_c[k] + radii[k]
        if mod_c[k] > 1e-6:
            half_arc = np.arcsin(min(1.0, radii[k] / mod_c[k]))
        else:
            half_arc = np.pi
        ang_arc = np.linspace(ang_c[k] - half_arc, ang_c[k] + half_arc, 120)

        r_arc = np.full_like(ang_arc, r_outer)
        ax.plot(ang_arc, r_arc, '-', color=colors_plot[k], linewidth=1.5, alpha=0.8)
        r_arc = np.full_like(ang_arc, r_inner)
        ax.plot(ang_arc, r_arc, '-', color=colors_plot[k], linewidth=1.0, alpha=0.4)
        for edge_ang in [ang_c[k] - half_arc, ang_c[k] + half_arc]:
            ax.plot([edge_ang, edge_ang], [r_inner, r_outer],
                    '-', color=colors_plot[k], linewidth=1.0, alpha=0.5)

    for k in range(n_clusters):
        ax.scatter(ang_c[k], mod_c[k], s=160, marker='X', color=colors_plot[k],
                   edgecolors='white', linewidths=0.8, zorder=5)

    ax.set_title(f'CVA con filtro SSIM — KMeans AUTO-K (k={n_clusters}, silhouette={silhouette:.3f})', pad=18)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Scatter polare salvato: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

def assign_clusters_to_patch(x_map: np.ndarray, y_map: np.ndarray,
                              module_map: np.ndarray,
                              centers_xy: np.ndarray,
                              cluster_colors: np.ndarray,
                              amp_threshold: float = 0.5) -> np.ndarray:
    H, W = x_map.shape

    module_norm = (module_map - module_map.mean()) / (module_map.std() + 1e-8)
    mask = module_norm >= amp_threshold if amp_threshold is not None else np.ones_like(module_map, dtype=bool)

    overlay = np.zeros((H, W, 4), dtype=float)

    if mask.sum() > 0:
        points = np.column_stack([x_map[mask], y_map[mask]])
        dists = np.sqrt(np.sum((points[:, np.newaxis, :] - centers_xy[np.newaxis, :, :]) ** 2, axis=2))
        labels = np.argmin(dists, axis=1)
        overlay[mask] = cluster_colors[labels]

    return overlay


def stitch_overlay_rightwins(patches_rgba: List[np.ndarray],
                              patch_size: int = 256,
                              overlap: int = 128) -> np.ndarray:
    H, W = patches_rgba[0].shape[:2]
    step = patch_size - overlap
    n_patches = len(patches_rgba)
    total_width = (n_patches - 1) * step + patch_size

    full = np.zeros((H, total_width, 4), dtype=float)
    for i, patch in enumerate(patches_rgba):
        x0 = i * step
        full[:, x0:x0 + W] = patch

    return full


def stitch_grayscale_mean(patches_gray: List[np.ndarray],
                           patch_size: int = 256,
                           overlap: int = 128) -> np.ndarray:
    H, W = patches_gray[0].shape
    step = patch_size - overlap
    n_patches = len(patches_gray)
    total_width = (n_patches - 1) * step + patch_size

    sum_img = np.zeros((H, total_width), dtype=float)
    count_img = np.zeros((H, total_width), dtype=int)

    for i, patch in enumerate(patches_gray):
        x0 = i * step
        sum_img[:, x0:x0 + W] += patch
        count_img[:, x0:x0 + W] += 1

    count_img[count_img == 0] = 1
    return sum_img / count_img


def plot_overlay(full_image: np.ndarray, full_overlay: np.ndarray,
                 cluster_colors: np.ndarray, n_clusters: int, out_path: str):
    n_changed = int((full_overlay[..., 3] > 0).sum())
    total_px = full_overlay.shape[0] * full_overlay.shape[1]
    pct = 100 * n_changed / total_px if total_px > 0 else 0

    fig, ax = plt.subplots(figsize=(max(12, full_image.shape[1] // 40), 7))
    ax.imshow(full_image, cmap='gray', aspect='auto',
              vmin=np.percentile(full_image, 2),
              vmax=np.percentile(full_image, 98))
    ax.imshow(full_overlay, aspect='auto')

    legend_elements = [
        mpatches.Patch(color=tuple(c[:3]), alpha=c[3], label=f'Cluster {i}')
        for i, c in enumerate(cluster_colors)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax.set_title(f'CVA con filtro SSIM — KMeans (k={n_clusters}) — '
                 f'soglia ampiezza >= {AMPLITUDE_THRESHOLD}\n'
                 f'{n_changed} pixel colorati ({pct:.1f}%)')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Overlay salvato: {out_path}")


def plot_module_overlay_normal(module_patches: List[np.ndarray],
                                patch_size: int = 256,
                                overlap: int = 128,
                                out_path: str = "") -> np.ndarray:
    full_module = stitch_grayscale_mean(module_patches, patch_size, overlap)

    fig, ax = plt.subplots(figsize=(max(12, full_module.shape[1] // 40), 7))
    im = ax.imshow(full_module, cmap='gray', aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.7, label="Modulo CVA")
    ax.set_title("CVA con filtro SSIM — Mappa del modulo (tutti i pixel, nessuna soglia)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Modulo normale salvato: {out_path}")

    return full_module


# ─────────────────────────────────────────────────────────────────────────────
# KDE DENSITY SCATTER
# ─────────────────────────────────────────────────────────────────────────────

def _kde_gpu(modules: np.ndarray, angles: np.ndarray,
             n_sample: int = 10_000, device: str = "cpu") -> np.ndarray:
    x = modules * np.cos(angles)
    y = modules * np.sin(angles)
    xy_t = torch.tensor(np.vstack([x, y]), dtype=torch.float32, device=device)
    N = xy_t.shape[1]
    n_s = min(N, n_sample)

    idx_s = torch.randperm(N, device=device)[:n_s]
    S = xy_t[:, idx_s].t()
    X = xy_t.t()
    bw = 1.06 * xy_t.std() * (n_s ** -0.2)

    z_t = torch.zeros(N, device=device)
    batch_size = 50_000
    with torch.no_grad():
        for i in range(0, N, batch_size):
            Xb = X[i : i + batch_size]
            dist_sq = ((Xb.unsqueeze(1) - S.unsqueeze(0)) ** 2).sum(2)
            z_t[i : i + batch_size] = torch.exp(-dist_sq / (2 * bw ** 2)).sum(1)

    return z_t.cpu().numpy()


def plot_scatter_kde(modules_all: np.ndarray, angles_all: np.ndarray,
                     out_dir: str, name: str, device: str = "cpu"):
    z = _kde_gpu(modules_all, angles_all, device=device)
    idx_sort = np.argsort(z)
    a_s = angles_all[idx_sort]
    m_s = modules_all[idx_sort]
    z_s = z[idx_sort]

    vmin_log = max(z_s[z_s > 0].min(), 1e-6)
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={"projection": "polar"})
    sc = ax.scatter(a_s, m_s, c=z_s, s=3, cmap="turbo", alpha=0.9,
                    edgecolors="none", norm=LogNorm(vmin=vmin_log, vmax=z_s.max()))
    plt.colorbar(sc, ax=ax, label="Densità KDE (LogNorm)", pad=0.1)
    ax.set_title(f"CVA+SSIM Density LogNorm — {name}")
    fig.savefig(os.path.join(out_dir, f"cva_ssim_scatter_density_{name}_lognorm.png"), dpi=300)
    plt.close(fig)
    print(f"  Scatter KDE LogNorm salvato.")

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={"projection": "polar"})
    sc = ax.scatter(a_s, m_s, c=z_s, s=2, cmap="turbo", alpha=0.8,
                    edgecolors="none",
                    norm=PowerNorm(gamma=0.5, vmin=z_s.min(), vmax=z_s.max()))
    plt.colorbar(sc, ax=ax, label="Densità KDE (PowerNorm γ=0.5)", pad=0.1)
    ax.set_title(f"CVA+SSIM Density PowerNorm — {name}")
    fig.savefig(os.path.join(out_dir, f"cva_ssim_scatter_density_{name}_powernorm.png"), dpi=300)
    plt.close(fig)
    print(f"  Scatter KDE PowerNorm salvato.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ESP = CVA_config.ESP
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join('/app/patch', CVA_config.PLOTS_SELECTED, ESP)
    OUT_DIR = os.path.join(BASE_DIR, CVA_config.OUT_DIR)
    HIST_DIR = os.path.join(OUT_DIR, "histograms")

    # ── 1. Carica modello ShallowASPP_Refined ─────────────────────────────
    print(f"[1] Caricamento modello ShallowASPP_Refined su {DEVICE}...")
    sh_model = ShallowASPP_Refined(in_channels=2, out_channels=2).to(DEVICE)
    sh_model.load_state_dict(torch.load(
        os.path.join(BASE_DIR, CVA_config.MODEL_NAME, 'best_model.pth'),
        map_location=DEVICE,
        weights_only=False,
    ))
    sh_model.eval()

    # ── 2. Lista patch ────────────────────────────────────────────────────
    patch_files14 = sorted(glob.glob(os.path.join(DATA_DIR, 'patch_*_2014.npy')))[:MAX_PATCHES]
    patch_ids = [os.path.basename(p)[:-9] for p in patch_files14]
    print(f"[2] {len(patch_ids)} patch trovate")

    # ── 3. Feature extraction ─────────────────────────────────────────────
    print("[3] Estrazione feature patchwise (ASPP+Refinement)...")
    all_feat14 = []
    all_feat16 = []

    for pid in patch_ids:
        p14 = np.load(os.path.join(DATA_DIR, f"{pid}_2014.npy"))
        p16 = np.load(os.path.join(DATA_DIR, f"{pid}_2016.npy"))

        inp14 = torch.from_numpy(np.stack([p14, p14], 0))[None].float().to(DEVICE)
        inp16 = torch.from_numpy(np.stack([p16, p16], 0))[None].float().to(DEVICE)

        with torch.no_grad():
            feat14 = extract_features_aspp(sh_model, inp14)
            feat16 = extract_features_aspp(sh_model, inp16)

        all_feat14.append(feat14)
        all_feat16.append(feat16)

    # ── 4. Stitching globale feature ──────────────────────────────────────
    print("[4] Stitching feature in immagine globale...")
    big_feat14 = stitch_patches(all_feat14, PATCH_SIZE, OVERLAP)
    big_feat16 = stitch_patches(all_feat16, PATCH_SIZE, OVERLAP)
    diff_global = big_feat16 - big_feat14
    print(f"    Feature globali shape: {diff_global.shape}")

    # ── 5. Filtro canali: varianza → SSIM, poi CVA ────────────────────────
    print("[5] Filtro canali (Variance → SSIM diversity)...")
    candidate_idx, variances = select_variance_channels_global(diff_global, N_VARIANCE)
    selected_idx, change_scores = select_channels_by_ssim_diversity(
        diff_global, candidate_idx, N_FINAL
    )
    print(f"    Canali selezionati (filtro): {selected_idx.tolist()}")

    # Istogrammi
    plot_variance_histogram(variances, selected_idx, "cva_ssim", HIST_DIR)
    plot_ssim_histogram(change_scores, candidate_idx, selected_idx, "cva_ssim", HIST_DIR)

    # ── 6. CVA per ogni patch + accumulo XY per KMeans ───────────────────
    print("[6] CVA (PCA) per ogni patch...")
    all_modules = []
    all_angles = []
    all_x = []
    all_y = []
    gray_patches = []
    module_map_patches = []

    step = PATCH_SIZE - OVERLAP
    for idx, pid in enumerate(patch_ids):
        diff_patch = diff_global[:, :, idx * step : idx * step + PATCH_SIZE]
        diff_sel = diff_patch[selected_idx].reshape(len(selected_idx), -1).astype(np.float64)

        dx, dy, module_1d, angle_1d = cva_project(diff_sel)

        H, W = diff_patch.shape[1:]
        module_map = module_1d.reshape(H, W)
        x_map = dx.reshape(H, W)
        y_map = dy.reshape(H, W)

        all_modules.append(module_1d)
        all_angles.append(angle_1d)
        all_x.append(x_map.ravel())
        all_y.append(y_map.ravel())
        module_map_patches.append(module_map)

        p14_gray = np.load(os.path.join(DATA_DIR, f"{pid}_2014.npy"))
        gray_patches.append(p14_gray)

    modules_all = np.concatenate(all_modules)
    angles_all = np.concatenate(all_angles)
    xy_all = np.column_stack([np.concatenate(all_x), np.concatenate(all_y)])
    print(f"    Punti totali per clustering: {len(xy_all)}")

    # ── 7. Auto-K KMeans ─────────────────────────────────────────────────
    print("[7] Clustering auto-k KMeans...")
    n_clusters, centers_xy, _, labels_full, silhouette_score_val, density_mask = \
        auto_kmeans_clustering(xy_all)

    colors_rgba = np.array([CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
                            for k in range(n_clusters)])

    # ── 8. Assegna cluster a ogni patch e crea overlay ────────────────────
    print("[8] Assegnazione cluster e creazione overlay...")
    overlays_rgba = []

    for idx in range(len(patch_ids)):
        H, W = gray_patches[idx].shape
        offset_start = sum(len(m) for m in all_modules[:idx])
        offset_end = offset_start + len(all_modules[idx])

        mod_patch = modules_all[offset_start:offset_end].reshape(H, W)
        x_map_patch = all_x[idx].reshape(H, W)
        y_map_patch = all_y[idx].reshape(H, W)

        overlay_patch = assign_clusters_to_patch(
            x_map_patch, y_map_patch, mod_patch,
            centers_xy, colors_rgba,
            amp_threshold=AMPLITUDE_THRESHOLD
        )
        overlays_rgba.append(overlay_patch)

    # ── 9. Stitching finale ───────────────────────────────────────────────
    print("[9] Stitching overlay finale...")
    full_image = stitch_grayscale_mean(gray_patches, PATCH_SIZE, OVERLAP)
    full_overlay = stitch_overlay_rightwins(overlays_rgba, PATCH_SIZE, OVERLAP)

    # ── 10. Salva output ──────────────────────────────────────────────────
    print("[10] Salvataggio output...")

    overlay_path = os.path.join(OUT_DIR, f'cva_ssim_kmeans_autok_overlay_{ESP}.png')
    plot_overlay(full_image, full_overlay, colors_rgba, n_clusters, overlay_path)

    module_normal_path = os.path.join(OUT_DIR, f'cva_ssim_module_normal_{ESP}.png')
    plot_module_overlay_normal(module_map_patches, PATCH_SIZE, OVERLAP, module_normal_path)

    plot_scatter_kde(modules_all, angles_all, OUT_DIR, ESP, DEVICE)

    scatter_path = os.path.join(OUT_DIR, f'cva_ssim_kmeans_autok_scatter_{ESP}.png')
    plot_scatter_polar(modules_all, angles_all, labels_full, centers_xy,
                       colors_rgba, n_clusters, silhouette_score_val, scatter_path)

    kmeans_data = {
        'centers_xy': centers_xy,
        'cluster_colors': colors_rgba,
        'n_clusters': n_clusters,
        'n_retained': N_VARIANCE,
        'percentile_mod': 70.2,
        'silhouette_score': silhouette_score_val
    }
    npy_path = os.path.join(OUT_DIR, f'cva_ssim_kmeans_autok_centers_{ESP}.npy')
    np.save(npy_path, kmeans_data, allow_pickle=True)
    print(f"    Centroidi salvati: {npy_path}")

    print("[DONE] ASPP_SSIM_CVA_kmeans_overlay completato.")
