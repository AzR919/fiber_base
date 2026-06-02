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
            nn.GELU(),
            nn.Conv2d(d_model, 2*d_model, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0)),
            nn.BatchNorm2d(2*d_model),
            nn.GELU(),
            nn.Conv2d(2*d_model, 1, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0)),
            nn.GELU()
        )
        # TODO: 2 input branches. conditional autoencoder. Multiple instance learning

        # 2. After processing fibers, we aggregate (Mean/Sum) and refine
        # Now we are back to 1D
        # self.bulk_predictor = nn.Sequential(
        #     nn.Conv1d(2*d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2),
        #     nn.ReLU(),
        #     nn.Conv1d(d_model, 1, kernel_size=1),
        #     nn.Softplus() # Ensures positive bulk signal
        # )
        # TODO: second branch

    def forward(self, fibers, dna):
        # fibers: (B, L, N), dna: (B, L, 4)

        # Add channel dimension for 2D Conv
        # x = fibers.unsqueeze(1)                             # (B, 1, L, N)

        # Apply fiber-wise convolutions
        processed_fibers = self.fiber_conv(fibers)               # (B, C, L, N)

        # Aggregate across fibers (N dimension)
        # This converts single-molecule features into a summary feature map
        y = torch.mean(processed_fibers, dim=-1)            # (B, 1, L)

        # Final refinement to predict bulk
        # out = self.bulk_predictor(y)                        # (B, 1, L)

        # return out.squeeze(1), processed_fibers.squeeze(1)  # (B, L), (B, L, N)
        return y.squeeze(1), processed_fibers.squeeze(1)  # (B, L), (B, L, N)

class FiberConv1dBlock(nn.Module):
    def __init__(self, num_input_features=1, d_model=64, kernel_size=15):
        super().__init__()

        # Input to Conv1d expects (Batch, Channels, Length)
        # We use standard padding to preserve the Length (L) dimension
        self.fiber_conv = nn.Sequential(
            nn.Conv1d(num_input_features, d_model, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, 2*d_model, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(2*d_model),
            nn.GELU(),
            # Last layer maps back to 1 channel per fiber
            nn.Conv1d(2*d_model, 1, kernel_size=kernel_size, padding=kernel_size//2),
            nn.GELU()
        )

    def forward(self, x, dna):
        """
        x: Input tensor of shape (B, C, L, N)
        """
        B, C, L, N = x.shape

        # 1. Permute to get dimensions ready for flattening: (B, N, C, L)
        # 2. Reshape to combine Batch and Fiber count: (B * N, C, L)
        x_flat = x.permute(0, 3, 1, 2).reshape(B * N, C, L)

        # 3. Pass through the 1D Convolutional pipeline
        # Output shape: (B * N, 1, L)
        out_flat = self.fiber_conv(x_flat)

        # 4. Reconstruct original dimensions:
        # Separate B and N again -> (B, N, 1, L)
        # Permute back to the (B, C, L, N) format -> (B, 1, L, N)
        processed_fibers = out_flat.view(B, N, 1, L).permute(0, 2, 3, 1).squeeze(1)

        # y = torch.sum(processed_fibers, dim=-1)            # (B, L)
        y = torch.mean(processed_fibers, dim=-1)            # (B, L)

        return y, processed_fibers

class FiberTransformerVAE(nn.Module):
    def __init__(self, n_channels=5, d_model=256, nhead=8, num_enc_layers=4, num_dec_layers=4, latent_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.latent_seq_len = latent_seq_len

        # --- ENCODER ---
        self.enc_embedding = nn.Linear(n_channels, d_model)
        self.enc_pos_emb = nn.Parameter(torch.randn(1, 2000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)

        # Bottleneck: Reduces L to latent_seq_len via learned pooling or linear projection
        self.to_latent = nn.Linear(2000, latent_seq_len) # Example for fixed L=2000

        # --- DECODER (Autoregressive) ---
        self.dec_embedding = nn.Linear(n_channels, d_model)
        self.dec_pos_emb = nn.Parameter(torch.randn(1, 2000, d_model))

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec_layers)

        self.output_head = nn.Linear(d_model, n_channels)

    def encode(self, x):
        # x: (BN, L, C)
        BN, L, C = x.shape
        x = self.enc_embedding(x) + self.enc_pos_emb[:, :L, :]

        # Full Self-Attention (Bi-directional)
        memory = self.transformer_encoder(x)

        # Compress L dimension to latent_seq_len
        # Memory is (BN, L, d_model) -> Transpose to (BN, d_model, L) for linear compression
        latent = self.to_latent(memory.transpose(1, 2)).transpose(1, 2)
        return latent # (BN, latent_seq_len, d_model)

    def forward(self, x):
        B, C, L, N = x.shape
        # Flatten: (B*N, L, C)
        x_flat = x.permute(0, 3, 2, 1).reshape(B * N, L, C)

        # 1. ENCODE
        latent = self.encode(x_flat)

        # 2. DECODE (Autoregressive)
        # Shift targets for teacher forcing: [0, t1, t2, ...] -> predict [t1, t2, t3, ...]
        dec_input = torch.zeros_like(x_flat)
        dec_input[:, 1:, :] = x_flat[:, :-1, :]

        tgt = self.dec_embedding(dec_input) + self.dec_pos_emb[:, :L, :]

        # Causal mask to prevent looking ahead in the sequence L
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)

        # Decoder attends to 'tgt' causally and uses 'latent' as cross-attention memory
        decoded = self.transformer_decoder(tgt=tgt, memory=latent, tgt_mask=causal_mask)

        logits = self.output_head(decoded)

        # Reshape back to (B, C, L, N)
        return logits.view(B, N, L, C).permute(0, 3, 2, 1)

#--------------------------------------------------------------------------------------------------
# model selection based on cmd arg

def model_selector(model_arg, args):

    model_name = model_arg.lower()

    if model_name=="base": return Base_Model(args.fibers_per_entry)
    if model_name=="simple": return Simple_Add_CNN_Model(args.fibers_per_entry)
    if model_name=="fiber_conv": return Per_Fiber_Conv_Model(args.num_input_features, d_model=args.d_model)
    if model_name=="fiber_conv_1d": return FiberConv1dBlock(args.num_input_features, d_model=args.d_model)

    raise NotImplementedError(f"Model not implemented: {model_arg}")


#--------------------------------------------------------------------------------------------------
# testing

def tester():
    pass

if __name__=="__main__":

    tester()
