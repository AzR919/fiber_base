"""
Main model file

"""

import torch
import torch.nn as nn

class Base_Model(nn.Module):
    """
    Simple conv transformer model

    """
    def __init__(self, d_m6a, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.conv_m6a = nn.Conv1d(d_m6a, d_model//2, kernel_size=3, padding=1)
        self.conv_dna = nn.Conv1d(4, d_model//2, kernel_size=3, padding=1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers=num_layers
        )
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, m6as, dna):
        # m6as: (B, L, D), dna: (B, L, 4)
        x1 = self.conv_m6a(m6as.permute(0, 2, 1))   # (B, d/2, L)
        x2 = self.conv_dna(dna.permute(0, 2, 1))            # (B, d/2, L)
        x = torch.cat([x1, x2], dim=1).permute(2, 0, 1)     # (L, B, d)
        x = self.transformer(x)                             # (L, B, d)
        return self.regressor(x).squeeze(-1).permute(1, 0)  # (B, L)
