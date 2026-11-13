# -*- coding: utf-8 -*-
"""
ShallowASPP_BiLSTM
==================
Architettura encoder per change detection su patch SAR.

Pipeline (vedi ASPP_CNN_MODIFICATA.png):
  Input (B, C, H, W)
      │
   [ASPP]               ← contesto multi-scala          utils.aspp.ASPP
      │
  [3× ShallowConvBlock] ← raffinamento locale, no pooling
      │
  [Bi-ConvLSTM] NEW     ← contesto sequenziale lungo tracce orizzontali
      │
  [Conv 1×1]            ← proiezione features  (→ self.proj)
      │
  [Classifier 1×1]      ← output finale  (→ self.classifier)

Feature map esportabile: output di self.proj  (features canali, stessa H×W)
"""

import torch
import torch.nn as nn

from models.utils.aspp import ASPP


# ── Blocco Conv Shallow ────────────────────────────────────────────────────────
class ShallowConvBlock(nn.Sequential):
    """Conv 3×3 → BN → ReLU. Nessun pooling: preserva H×W."""
    def __init__(self, channels: int):
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )


# ── ConvLSTM Cell ──────────────────────────────────────────────────────────────
class ConvLSTMCell(nn.Module):
    """
    Cella ConvLSTM — Shi et al., NeurIPS 2015.
    Gate i/f/g/o calcolati in una sola Conv2d su [x ∥ h].
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(4 * hidden_channels)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        raw = self.bn(self.gates(torch.cat([x, h], dim=1)))
        i, f, g, o = raw.chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

    def init_hidden(self, B: int, H: int, W: int, device):
        z = torch.zeros(B, self.hidden_channels, H, W, device=device)
        return z, z.clone()


# ── Bi-ConvLSTM ────────────────────────────────────────────────────────────────
class BiConvLSTM(nn.Module):
    """
    ConvLSTM bidirezionale che scorre lungo le tracce orizzontali (dim W).

    Ogni passo processa una colonna (B, C, H, 1):
      · forward  : colonna 0 → W-1
      · backward : colonna W-1 → 0

    Le due sequenze vengono riassemblate e fuse con Conv 1×1.

    Input : (B, C,       H, W)
    Output: (B, channels, H, W)   — stessa shape dell'input
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.fwd = ConvLSTMCell(channels, channels, kernel_size)
        self.bwd = ConvLSTMCell(channels, channels, kernel_size)
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        dev = x.device
        h_f, c_f = self.fwd.init_hidden(B, H, 1, dev)
        h_b, c_b = self.bwd.init_hidden(B, H, 1, dev)

        fwd_cols, bwd_cols = [], [None] * W
        for w in range(W):
            h_f, c_f = self.fwd(x[:, :, :, w:w+1], h_f, c_f)
            fwd_cols.append(h_f)
        for w in reversed(range(W)):
            h_b, c_b = self.bwd(x[:, :, :, w:w+1], h_b, c_b)
            bwd_cols[w] = h_b

        fwd_map = torch.cat(fwd_cols, dim=3)   # (B, C, H, W)
        bwd_map = torch.cat(bwd_cols, dim=3)
        return self.fusion(torch.cat([fwd_map, bwd_map], dim=1))


# ── Modello principale ─────────────────────────────────────────────────────────
class ShallowASPP_BiLSTM(nn.Module):
    """
    Rete encoder per change detection su patch SAR dual-pol.

    Args:
        in_channels        : canali input (default 2, SAR HH+HV).
        out_channels       : classi output (default 2).
        features           : dimensione interna (default 64).
        atrous_rates       : rate ASPP (default (6, 12, 18)).
        num_shallow_blocks : numero di blocchi shallow (default 3).
        lstm_kernel_size   : kernel della BiConvLSTM (default 3).

    Forward output:
        Tensor (B, out_channels, H, W)  — logit per pixel.

    Feature map intermedia (per CVA esterna):
        model.extract_features(x)  → Tensor (B, features, H, W)
        oppure usa direttamente model.get_features(x) che restituisce
        le features PRIMA del classifier (output di self.proj).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: int = 64,
        atrous_rates: tuple = (6, 12, 18),
        num_shallow_blocks: int = 3,
        lstm_kernel_size: int = 3,
    ):
        super().__init__()

        # 1) ASPP  — usa utils.aspp (tua libreria)
        self.aspp = ASPP(
            in_channels=in_channels,
            out_channels=features,
            atrous_rates=atrous_rates,
        )

        # 2) Shallow blocks — senza pooling
        self.refinement = nn.Sequential(
            *[ShallowConvBlock(features) for _ in range(num_shallow_blocks)]
        )

        # 3) Bi-ConvLSTM ← NUOVA
        self.bi_convlstm = BiConvLSTM(channels=features, kernel_size=lstm_kernel_size)

        # 4) Proiezione 1×1  ← punto di estrazione features
        self.proj = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )

        # 5) Classificatore finale
        self.classifier = nn.Conv2d(features, out_channels, kernel_size=1)

    # ── forward completo (usato in training) ──────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aspp(x)
        x = self.refinement(x)
        x = self.bi_convlstm(x)
        x = self.proj(x)
        return self.classifier(x)

    # ── estrazione feature interne (usato in CVA, interpretabilità) ───────────
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Restituisce la feature map PRIMA del classifier (output di self.proj).
        Shape: (B, features, H, W)  —  stessa H×W dell'input.
        """
        with torch.no_grad():
            x = self.aspp(x)
            x = self.refinement(x)
            x = self.bi_convlstm(x)
            x = self.proj(x)
        return x



# 1. Definizione loss
def tv_loss(features):
    diff_h = features[:, :, 1:, :] - features[:, :, :-1, :]
    diff_w = features[:, :, :, 1:] - features[:, :, :, :-1]
    return torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_w))


# if __name__ == '__main__':
#     # Example training step / usage snippet kept under __main__ guard so importing
#     # this module does not execute demo code (which references variables that
#     # are undefined at import-time). Run this file as a script to execute it.
#     try:
#         # The following is an illustrative example; adapt variables before use.
#         features = model.extract_features(input_data)  # requires model/input_data defined
#         loss_recon = criterion(output, target)         # requires criterion/output/target
#         loss_tv = tv_loss(features)
#         total_loss = loss_recon + lambda_tv * loss_tv
#         total_loss.backward()
#     except NameError:
#         print("Example training snippet skipped: define 'model', 'input_data', 'criterion', etc. to run it.")
