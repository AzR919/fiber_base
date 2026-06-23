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
    def __init__(self, num_input_features=1, decoder_type="avg", d_model=64, kernel_size=15, dilation=1):
        super().__init__()

        # Input to Conv1d expects (Batch, Channels, Length)
        # We use standard padding to preserve the Length (L) dimension
        # self.fiber_conv = nn.Sequential(
        #     nn.Conv1d(num_input_features, d_model, kernel_size=kernel_size, padding=kernel_size//2),
        #     nn.BatchNorm1d(d_model),
        #     nn.GELU(),
        #     nn.Conv1d(d_model, 2*d_model, kernel_size=kernel_size, padding=kernel_size//2),
        #     nn.BatchNorm1d(2*d_model),
        #     nn.GELU(),
        #     # Last layer maps back to 1 channel per fiber
        #     nn.Conv1d(2*d_model, 1, kernel_size=kernel_size, padding=kernel_size//2),
        #     nn.GELU()
        # )

        self.fiber_conv = nn.Sequential(
            nn.Conv1d(num_input_features, d_model, kernel_size=kernel_size, padding=kernel_size//2, dilation=dilation),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            # Last layer maps back to 1 channel per fiber
            nn.Conv1d(d_model, 1, kernel_size=kernel_size, padding=kernel_size//2, dilation=dilation),
            nn.GELU()
        )

        self.decoder_type = decoder_type

        self.final_layer = nn.Sequential(
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

        if self.decoder_type == "sum":
            y = torch.sum(processed_fibers, dim=-1)             # (B,L,N) -> (B, L)
        elif self.decoder_type == "avg":
            y = torch.mean(processed_fibers, dim=-1)            # (B,L,N) -> (B, L)
        else:
            raise NotImplementedError(f"decoder_type not implemented: {self.decoder_type}")

        y_final = self.final_layer(y)

        return y_final, processed_fibers

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

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# MODULE 1: INSTANCE ENCODER (Dilated CNN)
# =====================================================================
class InstanceEncoder(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=32, latent_channels=16):
        super(InstanceEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=4, dilation=4)
        self.bottleneck = nn.Conv1d(hidden_channels, latent_channels, kernel_size=1)
        self.layer_norm = nn.LayerNorm(latent_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        h = self.activation(self.conv1(x))
        h = self.activation(self.conv2(h))
        h = self.activation(self.conv3(h))
        latent = self.bottleneck(h)
        latent = latent.permute(0, 2, 1)
        latent = self.layer_norm(latent)
        return latent.permute(0, 2, 1)    # Output Shape: (Batch * N, latent_channels, d)


# =====================================================================
# MODULE 2: BAG AGGREGATOR (Set Transformer Components)
# =====================================================================
class InducedSetAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_inducing_points=16):
        super(InducedSetAttentionBlock, self).__init__()
        self.num_inducing_points = num_inducing_points
        self.inducing_points = nn.Parameter(torch.Tensor(1, num_inducing_points, embed_dim))
        nn.init.xavier_uniform_(self.inducing_points)
        self.mab1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mab2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, X):
        batch_size = X.size(0)
        I = self.inducing_points.repeat(batch_size, 1, 1)
        H, _ = self.mab1(I, X, X)
        Out, _ = self.mab2(X, H, H)
        return Out


class SetTransformerAggregator(nn.Module):
    def __init__(self, latent_channels, num_heads=4, num_inducing_points=16):
        super(SetTransformerAggregator, self).__init__()
        self.isab = InducedSetAttentionBlock(latent_channels, num_heads, num_inducing_points)

    def forward(self, X, n_fibers):
        B, N, C, d = X.shape
        X = X.permute(0, 3, 1, 2).contiguous().view(B * d, N, C)
        X_attn = self.isab(X)
        consensus = X_attn.mean(dim=1)
        consensus = consensus.view(B, d, C).permute(0, 2, 1)
        return consensus


# =====================================================================
# MODULE 3: DECODERS (Bulk Target Decoder & Input Reconstruction Decoder)
# =====================================================================
class DeconvolutionTower(nn.Module):
    def __init__(self, latent_channels, out_channels=1):
        super(DeconvolutionTower, self).__init__()
        """Maps latent space to the unobserved target assay channel."""
        self.conv1 = nn.Conv1d(latent_channels, latent_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(latent_channels, out_channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, h):
        out = self.activation(self.conv1(h))
        out = self.conv2(out)
        return out

class ReconstructionDecoder(nn.Module):
    def __init__(self, latent_channels, original_channels=5):
        super(ReconstructionDecoder, self).__init__()
        """
        NEW MODULE: Maps latent space back to the original input channel shape (5 channels).
        Forces the encoder to retain high-fidelity features of the input fiber.
        """
        self.conv1 = nn.Conv1d(latent_channels, latent_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(latent_channels, original_channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, h):
        out = self.activation(self.conv1(h))
        out = self.conv2(out) # Output shape: (Batch * N, original_channels, d)
        return out


# =====================================================================
# MASTER PIPELINE ENVELOPE (With Joint Objective Support)
# =====================================================================
class FiberMILModel(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=32, latent_channels=16, num_inducing_points=16):
        super(FiberMILModel, self).__init__()

        self.encoder = InstanceEncoder(in_channels, hidden_channels, latent_channels)
        self.aggregator = SetTransformerAggregator(latent_channels, num_heads=4, num_inducing_points=num_inducing_points)
        self.bulk_decoder = DeconvolutionTower(latent_channels, out_channels=1)

        # New Reconstruction Head
        self.recon_decoder = ReconstructionDecoder(latent_channels, original_channels=in_channels)

    def forward(self, X, mode="train"):
        """
        X shape: (Batch, N, in_channels, d)
        """
        B, N, C, d = X.shape

        if mode == "train":
            # 1. Encode instances independently
            X_flat = X.view(B * N, C, d)
            latent_flat = self.encoder(X_flat) # (Batch * N, latent_channels, d)

            # 2. Compute Self-Supervised Reconstruction Matrix
            reconstructed_fibers_flat = self.recon_decoder(latent_flat)
            reconstructed_fibers = reconstructed_fibers_flat.view(B, N, C, d)

            # 3. Pull consensus embedding via Set Transformer for Weak Supervision
            latent_bag = latent_flat.view(B, N, -1, d)
            consensus = self.aggregator(latent_bag, n_fibers=N)

            # 4. Predict population bulk profile
            bulk_prediction = self.bulk_decoder(consensus).squeeze(1) # (Batch, d)

            # Return BOTH predictions so we can compute a combined loss matrix
            return bulk_prediction, reconstructed_fibers

        elif mode == "inference":
            X_flat = X.view(B * N, C, d)
            latent_single = self.encoder(X_flat)
            fiber_predictions = self.bulk_decoder(latent_single)
            return fiber_predictions.view(B, N, d)

#--------------------------------------------------------------------------------------------------
# model selection based on cmd arg

def model_selector(model_arg, args):

    model_name = model_arg.lower()

    if model_name=="base": return Base_Model(args.fibers_per_entry)
    if model_name=="simple": return Simple_Add_CNN_Model(args.fibers_per_entry)
    if model_name=="fiber_conv": return Per_Fiber_Conv_Model(args.num_input_features, d_model=args.d_model)
    if model_name=="fiber_conv_1d": return FiberConv1dBlock(args.num_input_features, d_model=args.d_model,
                                                            decoder_type=args.decoder_type, kernel_size=args.kernel_size,
                                                            dilation=args.dilation)
    if model_name=="mil": return FiberMILModel(in_channels=args.num_input_features, hidden_channels=args.d_model)

    raise NotImplementedError(f"Model not implemented: {model_arg}")


#--------------------------------------------------------------------------------------------------
# testing

def tester_0():

    B, C_in, L, N = 16, 4, 2048, 200
    d_model = 128
    decoder_type = "avg"
    kernel_size = 51
    dilation = 1

    test_model = FiberConv1dBlock(C_in, d_model=d_model,
                                decoder_type=decoder_type, kernel_size=kernel_size,
                                dilation=dilation)

    test_inp = torch.rand((B, C_in, L, N))
    test_out = test_model(test_inp, None)

    pass

def tester_1():

    batch_size = 2
    n_fibers_per_locus = 16
    input_channels = 5
    locus_length = 500

    model = FiberMILModel(in_channels=input_channels)
    model.train()

    # Generate mock inputs
    mock_input_fibers = torch.randn(batch_size, n_fibers_per_locus, input_channels, locus_length)
    mock_true_bulk_target = torch.randn(batch_size, locus_length)

    # Forward Pass
    pred_bulk, pred_recon = model(mock_input_fibers, mode="train")

    # Loss 1: Weakly Supervised Bulk Target Tracking Error
    criterion_bulk = nn.MSELoss()
    loss_bulk = criterion_bulk(pred_bulk, mock_true_bulk_target)

    # Loss 2: Self-Supervised Autoencoder Fiber Reconstruction Error
    criterion_recon = nn.MSELoss()
    loss_recon = criterion_recon(pred_recon, mock_input_fibers)

    # Joint Optimization Strategy: Balance both tasks using a hyperparameter weight (alpha)
    alpha = 0.5  # Adjust based on how heavily you want to regularize the encoder
    total_joint_loss = loss_bulk + (alpha * loss_recon)

    print("--> Multi-Task Execution Successful!")
    print(f"----> Bulk Prediction Loss:       {loss_bulk.item():.4f}")
    print(f"----> Fiber Reconstruction Loss:  {loss_recon.item():.4f}")
    print(f"----> Combined Total Joint Loss:  {total_joint_loss.item():.4f}")

if __name__=="__main__":

    tester_0()
