import torch
import torch.nn as nn
# Importiamo solo il modulo ASPP ufficiale
from models.utils.aspp import ASPP

class ShallowASPP_Refined(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, features=64):
        super().__init__()

        # Usiamo l'ASPP ufficiale di torchvision
        self.aspp = ASPP(
            in_channels=in_channels,
            atrous_rates=[6, 12, 18],
            out_channels=features
        )

        # Modulo di rifinitura locale
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
        x = self.aspp(x)
        x = self.refinement(x)
        out = self.classifier(x)
        return out


    def extract_features_aspp(model, x):
        # 1. Passa per l'ASPP
        feat_aspp = model.aspp(x)
        # 2. Passa per il refinement
        features = model.refinement(feat_aspp)

        # Restituisce il layer prima della classificazione (64 canali)
        # Se preferisci, puoi concatenare anche feat_aspp per averne 128!
        return features[0].detach().cpu().numpy()
