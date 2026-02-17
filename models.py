"""
Main model file

"""

import torch
import torch.nn as nn

#--------------------------------------------------------------------------------------------------
# Various models

class Base_Model(nn.Module):
    """
    Simple conv transformer model

    """
    def __init__(self, d_fibers, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.conv_fibers = nn.Conv1d(d_fibers, d_model//2, kernel_size=3, padding=1)
        self.conv_dna = nn.Conv1d(4, d_model//2, kernel_size=3, padding=1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers=num_layers
        )
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, fibers, dna):
        # fibers: (B, L, N), dna: (B, L, 4)
        x1 = self.conv_fibers(fibers.permute(0, 2, 1))   # (B, d/2, L)
        x2 = self.conv_dna(dna.permute(0, 2, 1))            # (B, d/2, L)
        x = torch.cat([x1, x2], dim=1).permute(2, 0, 1)     # (L, B, d)
        x = self.transformer(x)                             # (L, B, d)
        return self.regressor(x).squeeze(-1).permute(1, 0), fibers  # (B, L)

class Simple_Add_CNN_Model(nn.Module):
    """
    Collapses N fibers into a pseudo-bulk signal and predicts
    experimental bulk tracks using Convolutions.
    """
    def __init__(self, d_fibers, d_model=64, kernel_size=15):
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

    def forward(self, fibers, dna):
        """
        Args:
            fibers: (B, L, d_fibers) -> B=Batch, N=Number of Fibers, L=Length
            dna:  (B, L, 4)        -> DNA is the same for all fibers in a locus
        """
        # 1. Aggregate Fibers: Sum/Mean across the N dimension
        # Result: (B, L, d_fibers)
        pseudo_bulk_fibers = torch.sum(fibers, dim=-1, keepdim=True)

        # 2. Ignore DNA
        # Shape: (B, L, d_fibers)
        # x = torch.cat(pseudo_bulk_fibers, dim=-1)

        # 3. Reshape for Conv1d: (B, Channels, Length)
        x = pseudo_bulk_fibers.permute(0, 2, 1)

        # 4. Pass through CNN
        out = self.conv_block(x) # (B, 1, L)

        # 5. Return to (B, L) format
        return out.squeeze(1)

class Per_Fiber_Conv_Model(nn.Module):
    """
    Simple conv transformer model

    """
    def __init__(self, num_input_features=1, d_model=64, kernel_size=15):
        super().__init__()

        # 1. Input is (B, num_input_features, L, d_fibers)
        # We use a kernel of (K, 1) to process each fiber independently
        self.fiber_conv = nn.Sequential(
            nn.Conv2d(num_input_features, d_model, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0)),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, 1, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        # 2. After processing fibers, we aggregate (Mean/Sum) and refine
        # Now we are back to 1D
        self.bulk_predictor = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(d_model, 1, kernel_size=1),
            nn.Softplus() # Ensures positive bulk signal
        )

    def forward(self, fibers, dna):
        # fibers: (B, L, N), dna: (B, L, 4)

        # Add channel dimension for 2D Conv
        # x = fibers.unsqueeze(1)                             # (B, 1, L, N)

        # Apply fiber-wise convolutions
        processed_fibers = self.fiber_conv(fibers)               # (B, C, L, N)

        # Aggregate across fibers (N dimension)
        # This converts single-molecule features into a summary feature map
        y = torch.sum(processed_fibers, dim=-1)            # (B, 1, L)

        # Final refinement to predict bulk
        out = self.bulk_predictor(y)                        # (B, 1, L)

        return out.squeeze(1), processed_fibers.squeeze(1)  # (B, L), (B, L, N)

class Transformer_Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()



#--------------------------------------------------------------------------------------------------
# model selection based on cmd arg

def model_selector(model_arg, args):

    model_name = model_arg.lower()

    if model_name=="base": return Base_Model(args.fibers_per_entry)
    if model_name=="simple": return Simple_Add_CNN_Model(args.fibers_per_entry)
    if model_name=="fiber_conv": return Per_Fiber_Conv_Model(args.num_input_features, d_model=8)

    raise NotImplementedError(f"Model not implemented: {model_arg}")


#--------------------------------------------------------------------------------------------------
# testing

def tester():
    pass

if __name__=="__main__":

    tester()
