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
        # m6as: (B, L, N), dna: (B, L, 4)
        x1 = self.conv_m6a(m6as.permute(0, 2, 1))   # (B, d/2, L)
        x2 = self.conv_dna(dna.permute(0, 2, 1))            # (B, d/2, L)
        x = torch.cat([x1, x2], dim=1).permute(2, 0, 1)     # (L, B, d)
        x = self.transformer(x)                             # (L, B, d)
        return self.regressor(x).squeeze(-1).permute(1, 0)  # (B, L)

class Simple_Add_CNN_Model(nn.Module):
    """
    Collapses N fibers into a pseudo-bulk signal and predicts
    experimental bulk tracks using Convolutions.
    """
    def __init__(self, d_m6a, d_model=64, kernel_size=15):
        super().__init__()

        self.conv_block = nn.Sequential(
            # First layer: Increase channels to capture local motifs
            nn.Conv1d(1, d_model, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),

            # Second layer: Refine features
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),

            # Final layer: Map back to a single bulk track prediction
            nn.Conv1d(d_model, 1, kernel_size=1)
        )

    def forward(self, m6as, dna):
        """
        Args:
            m6as: (B, L, d_m6a) -> B=Batch, N=Number of Fibers, L=Length
            dna:  (B, L, 4)        -> DNA is the same for all fibers in a locus
        """
        # 1. Aggregate Fibers: Sum/Mean across the N dimension
        # Result: (B, L, d_m6a)
        pseudo_bulk_m6a = torch.sum(m6as, dim=-1, keepdim=True)

        # 2. Ignore DNA
        # Shape: (B, L, d_m6a)
        # x = torch.cat(pseudo_bulk_m6a, dim=-1)

        # 3. Reshape for Conv1d: (B, Channels, Length)
        x = pseudo_bulk_m6a.permute(0, 2, 1)

        # 4. Pass through CNN
        out = self.conv_block(x) # (B, 1, L)

        # 5. Return to (B, L) format
        return out.squeeze(1)
