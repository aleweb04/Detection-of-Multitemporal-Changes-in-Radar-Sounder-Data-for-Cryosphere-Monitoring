# -*- coding: utf-8 -*-
"""
SSMI_s2cva_kmeans_overlay.py

Combina:
  - S²CVA globale (selezione canali variance → SSIM → proiezione eigen)
  - KMeans auto-k con Silhouette Score
  - Overlay finale a colori su patch stitchate

Output:
  - PNG overlay (immagine 2014 + cluster colorati)
  - PNG overlay normale (modulo S²CVA, tutti i pixel, scala di grigi)
  - PNG scatter polare (cluster + centroidi)
  - PNG scatter densità KDE (LogNorm + PowerNorm)
  - .npy centroidi (leggibile da auto_AMP_color_ASPP.py)
  - Istogrammi selezioni canali (varianza + SSIM)
"""

import os
import glob
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, PowerNorm
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

import CVA_config
from models.shallow_cnnASPP import ShallowASPP_Refined



# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRI
# ─────────────────────────────────────────────────────────────────────────────
N_VARIANCE = 16        # canali trattenuti dopo step varianza
N_FINAL = 4            # canali finali dopo SSIM diversità
PERCENTILE_MOD = 70.2  # percentile per soglia PCA (non usato in S²CVA puro)
AMPLITUDE_THRESHOLD = 1.0  # soglia sul modulo S²CVA normalizzato per overlay
PATCH_SIZE = 256
OVERLAP = 128
MAX_PATCHES = 64

CLUSTER_COLORS = [
    (1.0, 0.0, 0.0, 0.6), (0.0, 0.5, 1.0, 0.6), (0.0, 0.8, 0.2, 0.6),
    (1.0, 0.8, 0.0, 0.6), (0.8, 0.0, 0.8, 0.6), (0.0, 0.9, 0.9, 0.6),
    (1.0, 0.5, 0.0, 0.6), (0.5, 0.5, 0.5, 0.6)
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm2d(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def stitch_patches(patches: List[np.ndarray], patch_size: int = 256,
                   overlap: int = 128) -> np.ndarray:
    """Ricostruisce un'unica grande immagine dalle patch con overlap (media)."""
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
# SELEZIONE CANALI — S²CVA GLOBALE (da SSMI_final.py)
# ─────────────────────────────────────────────────────────────────────────────

def select_variance_channels_global(diff: np.ndarray,
                                    n_retained: int) -> Tuple[np.ndarray, np.ndarray]:
    """Step 1: Prende i top n_retained canali per varianza (no Otsu)."""
    C = diff.shape[0]
    diff_flat = diff.reshape(C, -1)
    variances = np.var(diff_flat, axis=1)

    k = min(n_retained, C)
    top_idx = np.argsort(variances)[-k:]

    return np.sort(top_idx), variances


def select_channels_by_diff_ssim(diff: np.ndarray, candidate_idx: np.ndarray,
                                  n_final: int) -> Tuple[np.ndarray, np.ndarray]:
    """Step 2: Dai candidati, seleziona i più diversi (minor SSIM reciproca)."""
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
# S²CVA PROJECTION (da SSMI_final.py)
# ─────────────────────────────────────────────────────────────────────────────

def s2cva_project(diff_sel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Proietta le differenze selezionate sull'autovettore principale.
    Restituisce module, angle (vettori 1D)."""
    B, N = diff_sel.shape
    module = np.sqrt(np.sum(diff_sel ** 2, axis=0))

    # Un solo canale — non serve eigen-decomposition
    if B == 1:
        angle = np.zeros(N)
        angle[diff_sel[0] < 0] = np.pi
        return module, angle

    cov = np.cov(diff_sel)
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        cov = np.eye(B)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    R = eigenvectors[:, -1]

    numerator = diff_sel.T @ R
    denominator = module * np.linalg.norm(R) + 1e-8
    cosine = np.clip(numerator / denominator, -1.0, 1.0)
    angle = np.arccos(cosine)

    return module, angle


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-K KMEANS (da scatterCVA_kmeans_patchwise_TRY.py)
# ─────────────────────────────────────────────────────────────────────────────

def auto_kmeans_clustering(xy_all: np.ndarray):
    """Auto-k KMeans con NearestNeighbors (30% densi) + silhouette (k=3..7)."""
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

    for k in range(2,4):
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

    # Mappa etichette su tutti i punti (-1 = esclusi dal clustering)
    labels_full = np.full(len(xy_all), -1)
    labels_full[density_mask] = best_labels

    return best_k, best_centers, best_labels, labels_full, best_score, density_mask


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING ISTOGRAMMI (da SSMI_final.py)
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
    ax.set_title(f"Varianza globale (rosso = feature filtrate con Otsu) — {name}")
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
    ax.set_title(f"SSIM sulle differenze (verde = {len(selected_idx)} canali S²CVA finali) — {name}")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"hist_ssim_diff_global_{name}.png"), dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING SCATTER POLARE (adattato da scatterCVA)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter_polar(modules_all, angles_all, labels_full, centers_xy,
                       cluster_colors, n_clusters, silhouette, out_path):
    """Scatter polare con cluster colorati, centroidi a X, archi raggio."""
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

    ax.set_title(f'S²CVA KMeans AUTO-K (k={n_clusters}, silhouette={silhouette:.3f})', pad=18)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Scatter polare salvato: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# OVERLAY (da auto_AMP_color_ASPP.py)
# ─────────────────────────────────────────────────────────────────────────────

def assign_clusters_to_patch(x_map: np.ndarray, y_map: np.ndarray,
                              module_map: np.ndarray,
                              centers_xy: np.ndarray,
                              cluster_colors: np.ndarray,
                              amp_threshold: float = 0.5) -> np.ndarray:
    """Assegna cluster ai pixel di una patch → overlay RGBA (H, W, 4)."""
    H, W = x_map.shape

    # Normalizza modulo per soglia
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
    """Stitch patch overlay con strategia right-wins (sovrascrive)."""
    H, W = patches_rgba[0].shape[:2]
    step = patch_size - overlap
    n_patches = len(patches_rgba)
    total_width = (n_patches - 1) * step + patch_size

    full = np.zeros((H, total_width, 4), dtype=float)
    for i, patch in enumerate(patches_rgba):
        x0 = i * step
        full[:, x0:x0 + W] = patch  # right-wins: l'ultima patch vince

    return full


def stitch_grayscale_mean(patches_gray: List[np.ndarray],
                           patch_size: int = 256,
                           overlap: int = 128) -> np.ndarray:
    """Stitch patch in scala di grigi con media nell'overlap."""
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
    """Salva immagine finale con overlay cluster + legenda."""
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
    ax.set_title(f'S²CVA KMeans (k={n_clusters}) — '
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
    """Stitcha le mappe modulo S²CVA (TUTTI i pixel, nessuna soglia)
    e salva un'immagine in scala di grigi."""
    full_module = stitch_grayscale_mean(module_patches, patch_size, overlap)

    fig, ax = plt.subplots(figsize=(max(12, full_module.shape[1] // 40), 7))
    im = ax.imshow(full_module, cmap='gray', aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.7, label="Modulo S²CVA")
    ax.set_title("S²CVA — Mappa del modulo (tutti i pixel, nessuna soglia)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Modulo normale salvato: {out_path}")

    return full_module


# ─────────────────────────────────────────────────────────────────────────────
# KDE DENSITY SCATTER (da SSMI_final)
# ─────────────────────────────────────────────────────────────────────────────

def _kde_gpu(modules: np.ndarray, angles: np.ndarray,
             n_sample: int = 10_000, device: str = "cpu") -> np.ndarray:
    """KDE su coordinate polari per densità scatter."""
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
    """Scatter polare con densità KDE (LogNorm + PowerNorm) — da SSMI_final."""
    z = _kde_gpu(modules_all, angles_all, device=device)
    idx_sort = np.argsort(z)
    a_s = angles_all[idx_sort]
    m_s = modules_all[idx_sort]
    z_s = z[idx_sort]

    # LogNorm
    vmin_log = max(z_s[z_s > 0].min(), 1e-6)
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={"projection": "polar"})
    sc = ax.scatter(a_s, m_s, c=z_s, s=3, cmap="turbo", alpha=0.9,
                    edgecolors="none", norm=LogNorm(vmin=vmin_log, vmax=z_s.max()))
    plt.colorbar(sc, ax=ax, label="Densità KDE (LogNorm)", pad=0.1)
    ax.set_title(f"S²CVA Density LogNorm — {name}")
    fig.savefig(os.path.join(out_dir, f"s2cva_scatter_density_{name}_lognorm.png"), dpi=300)
    plt.close(fig)
    print(f"  Scatter KDE LogNorm salvato.")

    # PowerNorm
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={"projection": "polar"})
    sc = ax.scatter(a_s, m_s, c=z_s, s=2, cmap="turbo", alpha=0.8,
                    edgecolors="none",
                    norm=PowerNorm(gamma=0.5, vmin=z_s.min(), vmax=z_s.max()))
    plt.colorbar(sc, ax=ax, label="Densità KDE (PowerNorm γ=0.5)", pad=0.1)
    ax.set_title(f"S²CVA Density PowerNorm — {name}")
    fig.savefig(os.path.join(out_dir, f"s2cva_scatter_density_{name}_powernorm.png"), dpi=300)
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
    OUT_DIR = os.path.join(BASE_DIR, f"{CVA_config.OUT_DIR}_{AMPLITUDE_THRESHOLD}")
    HIST_DIR = os.path.join(OUT_DIR, "histograms")

    # ── 1. Carica modello ShallowASPP_Refined ──────────────────────────────
    print(f"[1] Caricamento modello ShallowASPP_Refined su {DEVICE}...")
    sh_model = ShallowASPP_Refined(in_channels=2, out_channels=2).to(DEVICE)
    sh_model.load_state_dict(torch.load(
        os.path.join('/app/models/radargram_shallow', CVA_config.MODEL_NAME, 'best_model.pth'),
        map_location=DEVICE,
        weights_only=False,
    ))
    sh_model.eval()

    # ── 2. Lista patch ────────────────────────────────────────────────────
    patch_files14 = sorted(glob.glob(os.path.join(DATA_DIR, 'patch_*_2014.npy')))[:MAX_PATCHES]
    patch_ids = [os.path.basename(p)[:-9] for p in patch_files14]
    print(f"[2] {len(patch_ids)} patch trovate")

    # ── 3. Feature extraction per tutte le patch ──────────────────────────
    print("[3] Estrazione feature patchwise...")
    all_feat14 = []
    all_feat16 = []

    for pid in patch_ids:
        p14 = np.load(os.path.join(DATA_DIR, f"{pid}_2014.npy"))
        p16 = np.load(os.path.join(DATA_DIR, f"{pid}_2016.npy"))

        inp14 = torch.from_numpy(np.stack([p14, p14], 0))[None].float().to(DEVICE)
        inp16 = torch.from_numpy(np.stack([p16, p16], 0))[None].float().to(DEVICE)

        with torch.no_grad():
            feat14 = sh_model.extract_features_aspp(inp14).squeeze(0).cpu().numpy()
            feat16 = sh_model.extract_features_aspp(inp16).squeeze(0).cpu().numpy()

        all_feat14.append(feat14)
        all_feat16.append(feat16)

    # ── 4. Stitching globale feature ──────────────────────────────────────
    print("[4] Stitching feature in immagine globale...")
    big_feat14 = stitch_patches(all_feat14, PATCH_SIZE, OVERLAP)
    big_feat16 = stitch_patches(all_feat16, PATCH_SIZE, OVERLAP)
    diff_global = big_feat16 - big_feat14
    print(f"    Feature globali shape: {diff_global.shape}")

    # ── 5. Selezione canali globale (S²CVA) ───────────────────────────────
    print("[5] Selezione canali globale (Variance → SSIM)...")
    candidate_idx, variances = select_variance_channels_global(diff_global, N_VARIANCE)
    selected_idx, change_scores = select_channels_by_diff_ssim(diff_global, candidate_idx, N_FINAL)
    print(f"    Canali selezionati: {selected_idx.tolist()}")

    # Istogrammi
    plot_variance_histogram(variances, selected_idx, "s2cva_global", HIST_DIR)
    plot_ssim_histogram(change_scores, candidate_idx, selected_idx, "s2cva_global", HIST_DIR)

    # ── 6. S²CVA per ogni patch + accumulo XY per KMeans ─────────────────
    print("[6] S²CVA per ogni patch...")
    all_modules = []
    all_angles = []
    all_x = []
    all_y = []
    gray_patches = []
    module_map_patches = []  # per overlay normale (tutti i pixel)

    step = PATCH_SIZE - OVERLAP  # 128
    for idx, pid in enumerate(patch_ids):
        # Estrai la differenza dall'immagine globale stitchata (come in SSMI_final)
        # così l'overlap è già mediato e la proiezione S²CVA è identica
        diff_patch = diff_global[:, :, idx * step : idx * step + PATCH_SIZE]

        # S²CVA sui canali selezionati
        diff_sel = diff_patch[selected_idx].reshape(len(selected_idx), -1).astype(np.float64)
        module_1d, angle_1d = s2cva_project(diff_sel)

        # Reshape a 2D
        H, W = diff_patch.shape[1:]
        module_map = module_1d.reshape(H, W)
        angle_map = angle_1d.reshape(H, W)

        # Coordinate cartesiane per KMeans
        x_map = module_map * np.cos(angle_map)
        y_map = module_map * np.sin(angle_map)

        # Accumula per clustering
        all_modules.append(module_1d)
        all_angles.append(angle_1d)
        all_x.append(x_map.ravel())
        all_y.append(y_map.ravel())
        module_map_patches.append(module_map)  # per overlay normale

        # Carica immagine 2014 originale per overlay
        p14_gray = np.load(os.path.join(DATA_DIR, f"{pid}_2014.npy"))
        gray_patches.append(p14_gray)

    # Concatena tutti i punti
    modules_all = np.concatenate(all_modules)
    angles_all = np.concatenate(all_angles)
    xy_all = np.column_stack([np.concatenate(all_x), np.concatenate(all_y)])
    print(f"    Punti totali per clustering: {len(xy_all)}")

    # ── 7. Auto-K KMeans ─────────────────────────────────────────────────
    print("[7] Clustering auto-k KMeans...")
    n_clusters, centers_xy, _, labels_full, silhouette_score_val, density_mask = \
        auto_kmeans_clustering(xy_all)

    # Colori per i cluster
    colors_rgba = np.array([CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
                            for k in range(n_clusters)])

    # ── 8. Assegna cluster a ogni patch e crea overlay ────────────────────
    print("[8] Assegnazione cluster e creazione overlay...")
    overlays_rgba = []

    for idx, pid in enumerate(patch_ids):
        H, W = gray_patches[idx].shape

        # Recupera mappe S²CVA per questa patch
        offset_start = sum(len(m) for m in all_modules[:idx])
        offset_end = offset_start + len(all_modules[idx])
        mod_patch = modules_all[offset_start:offset_end].reshape(H, W)
        ang_patch = angles_all[offset_start:offset_end].reshape(H, W)
        x_map_patch = mod_patch * np.cos(ang_patch)
        y_map_patch = mod_patch * np.sin(ang_patch)

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

    # Overlay PNG
    overlay_path = os.path.join(OUT_DIR, f's2cva_kmeans_autok_overlay_{ESP}.png')
    plot_overlay(full_image, full_overlay, colors_rgba, n_clusters, overlay_path)

    # Overlay normale (modulo S²CVA, tutti i pixel, scala di grigi)
    module_normal_path = os.path.join(OUT_DIR, f's2cva_module_normal_{ESP}.png')
    plot_module_overlay_normal(module_map_patches, PATCH_SIZE, OVERLAP, module_normal_path)

    # Scatter KDE densità (LogNorm + PowerNorm)
    plot_scatter_kde(modules_all, angles_all, OUT_DIR, ESP, DEVICE)

    # Scatter polare PNG
    scatter_path = os.path.join(OUT_DIR, f's2cva_kmeans_autok_scatter_{ESP}.png')
    plot_scatter_polar(modules_all, angles_all, labels_full, centers_xy,
                       colors_rgba, n_clusters, silhouette_score_val, scatter_path)

    # Centroidi .npy (compatibile con auto_AMP_color_ASPP.py)
    kmeans_data = {
        'centers_xy': centers_xy,
        'cluster_colors': colors_rgba,
        'n_clusters': n_clusters,
        'n_retained': N_VARIANCE,
        'percentile_mod': PERCENTILE_MOD,
        'silhouette_score': silhouette_score_val
    }
    npy_path = os.path.join(OUT_DIR, f's2cva_kmeans_autok_centers_{ESP}.npy')
    np.save(npy_path, kmeans_data, allow_pickle=True)
    print(f"    Centroidi salvati: {npy_path}")

    print("[DONE] SSMI_s2cva_kmeans_overlay completato.")
