"""
Main model file
"""
import os
import math
import inspect
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

#--------------------------------------------------------------------------------------------------
# Base Model Class with Reusable Save / Load Logic

class BaseModel(nn.Module):
    """
    Abstract base class providing unified save and load functionality
    for all fiber-seq models.
    """
    def __init__(self, input_flags, dna_type, output_assays=None):
        super().__init__()
        if output_assays is None:
            output_assays = ["atac"]
        self.init_args = {
            "input_flags": input_flags,
            "dna_type": dna_type,
            "output_assays": output_assays,
        }

    def save_model(self, dir_name, epoch, external_config=None):
        """
        Saves both the model configuration parameters and the state dictionary
        together inside a single bundled file.
        """
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        save_path = os.path.join(dir_name, f"Model_epoch_{epoch}.pt")

        if external_config is not None:
            if hasattr(external_config, "__dict__"):
                complete_config = vars(external_config).copy()
            else:
                complete_config = dict(external_config).copy()
        else:
            complete_config = {}

        checkpoint_bundle = {
            "model_config": self.init_args,
            "config": complete_config,
            "state_dict": self.state_dict()
        }

        torch.save(checkpoint_bundle, save_path)
        print(f"Model blueprint and weights successfully bundled into: {save_path}")

    @classmethod
    def load_model(cls, filepath, map_location=None):
        """
        Loads a bundled file, extracts input_flags & configuration parameters to
        instantiate the exact structural layout, and loads the weights.

        Returns:
            model (nn.Module): The reconstituted PyTorch model.
            checkpoint_metadata (dict): Metadata dictionary containing 'input_flags' and full 'config'.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No bundled checkpoint found at path: {filepath}")

        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)

        if "model_config" not in checkpoint or "state_dict" not in checkpoint:
            raise KeyError("The checkpoint file does not match the expected bundle format.")

        init_args = checkpoint["model_config"]
        state_dict = checkpoint["state_dict"]
        config = checkpoint.get("config", {})

        # Sanity check
        if not init_args.get("input_flags"):
            raise ValueError(f"Checkpoint at {filepath} does not contain 'input_flags'.")

        model_name = config.get("model") if isinstance(config, dict) else None
        if not model_name:
            raise ValueError(
                f"Checkpoint at {filepath} has no 'model' key in config — cannot determine architecture."
            )

        model = model_selector(model_name, config)
        model.load_state_dict(state_dict)
        print(f"Model successfully reconstituted from: {filepath}")
        return model, config

#--------------------------------------------------------------------------------------------------
# Model Components

class ResidualBlock1D(nn.Module):
    """
    A 1D Convolutional Residual Block with GroupNorm,
    dynamic dilation/padding, and a 1x1 projection shortcut.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        padding = (dilation * (kernel_size - 1)) // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation
        )
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.act(self.shortcut(x) + self.norm(self.conv(x)))

class PositionalEncoding1D(nn.Module):
    """
    Learned positional embeddings matching dynamic context window sequences up to max_len.
    """
    def __init__(self, d_model, max_len=6000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x shape: [Batch, Length, d_model]
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0) # [1, L]
        return x + self.pos_embedding(positions)

class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def _forward_impl(self, x):
        residual = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)) + residual)
        return x

    def forward(self, x):
        if self.training:
            # Recomputes activations on backward pass instead of storing them all in VRAM
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)

class SinusoidalPositionalEncoding(nn.Module):
    """Standard 1D Sinusoidal Positional Encoding (Vaswani et al.)."""
    def __init__(self, d_model=96, max_len=1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: [1, max_len, d_model] for broadcasting across batch
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [Batch, Seq_Len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

#--------------------------------------------------------------------------------------------------
# Complete Model Implementations

class Base01DebugModel(BaseModel):

    def __init__(self, input_flags, dna_type, kernel_size=15, output_assays=None):
        super().__init__(input_flags, dna_type, output_assays)

        self.init_args["kernel_size"] = kernel_size

        num_input_features = sum(input_flags)
        K = len(self.init_args["output_assays"])

        self.fiber_conv = nn.Sequential(
            ResidualBlock1D(num_input_features, 2, kernel_size, dilation=1)
        )
        self.out_conv = nn.Conv1d(2, K, kernel_size=1)
        self.fiber_act = nn.Softplus()

    def forward(self, x, fiber_coverage=None, *args, **kwargs):
        B, C, L, N = x.shape
        K = self.out_conv.out_channels

        x_flat = x.permute(0, 3, 1, 2).reshape(B * N, C, L)
        feat = self.fiber_conv(x_flat)                                   # [B*N, 2, L]
        out_flat = self.out_conv(feat)                                   # [B*N, K, L]
        raw_fibers = out_flat.view(B, N, K, L).permute(0, 2, 3, 1)       # [B, K, L, N]
        processed_fibers = self.fiber_act(raw_fibers)                    # [B, K, L, N]
        cov = fiber_coverage.float().clamp(min=1).unsqueeze(1)           # [B, 1, L]
        y = processed_fibers.sum(-1) / cov                               # [B, K, L]
        return y, processed_fibers

class UNet03ConvTransformerWithDNA(BaseModel):
    """
    1D Hybrid U-Net Model with Sinusoidal Positional Encoding Transformer Bottleneck.

    Architecture Overview:
    - 3-level 1D Conv Encoder: compresses spatial context down to L / 8.
    - Sinusoidal Positional Encoding + Transformer Bottleneck: global spatial attention on L / 8 tokens.
    - 3-level 1D Conv Decoder: upsamples features back to L.
    - Output projection + Softplus non-negativity + coverage-normalized fiber averaging.
    """

    def __init__(self, input_flags, dna_type="none",
                 kernel_size=15, emb_dims=[8, 32, 64, 128], tf_heads=4,
                 tf_layers=2, max_len=1000, output_assays=None):
        super().__init__(input_flags, dna_type, output_assays)

        assert dna_type in ("none", "ref"), f"dna_type must be 'none' or 'ref', got '{dna_type}'."

        self.init_args["kernel_size"] = kernel_size
        self.init_args["tf_heads"] = tf_heads
        self.init_args["tf_layers"] = tf_layers
        self.init_args["emb_dims"] = emb_dims
        self.init_args["dna_type"] = dna_type

        self.emb_dims = emb_dims
        self.num_input_features = sum(input_flags)
        self.dna_type = dna_type
        self.kernel_size = kernel_size

        # --- Optional Reference DNA Encoder ---
        if self.dna_type == "ref":
            self.dna_encoder = DoubleConv1D(4, self.emb_dims[0], kernel_size=kernel_size)
            total_in_channels = 2 * self.emb_dims[0]
        else:
            self.dna_encoder = None
            total_in_channels = self.emb_dims[0]

        self.fiber_encoder = DoubleConv1D(self.num_input_features, self.emb_dims[0], kernel_size=kernel_size)

        # --- Encoder (Downsampling L -> L/8) ---
        self.enc1 = DoubleConv1D(total_in_channels, self.emb_dims[1], kernel_size=kernel_size)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # L -> L/2

        self.enc2 = DoubleConv1D(self.emb_dims[1], self.emb_dims[2], kernel_size=kernel_size)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # L/2 -> L/4

        self.enc3 = DoubleConv1D(self.emb_dims[2], self.emb_dims[3], kernel_size=kernel_size)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # L/4 -> L/8

        # --- Sinusoidal Positional Encoding & Transformer Bottleneck ---
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=self.emb_dims[3], max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dims[3],
            nhead=tf_heads,
            dim_feedforward=self.emb_dims[3] * 2,
            activation="gelu",
            batch_first=True
        )
        self.transformer_bottleneck = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)

        # --- Decoder (Upsampling L/8 -> L) ---
        self.up3 = nn.ConvTranspose1d(self.emb_dims[3], self.emb_dims[2], kernel_size=2, stride=2)  # Output: 48 channels
        self.dec3 = DoubleConv1D(self.emb_dims[2] + self.emb_dims[3], self.emb_dims[2], kernel_size=kernel_size)   # 48 (up3) + 96 (enc3) = 144 channels

        self.up2 = nn.ConvTranspose1d(self.emb_dims[2], self.emb_dims[1], kernel_size=2, stride=2)  # Output: 24 channels
        self.dec2 = DoubleConv1D(self.emb_dims[1] + self.emb_dims[2], self.emb_dims[1], kernel_size=kernel_size)   # 24 (up2) + 48 (enc2) = 72 channels

        self.up1 = nn.ConvTranspose1d(self.emb_dims[1], self.emb_dims[1], kernel_size=2, stride=2)  # Output: 24 channels
        self.dec1 = DoubleConv1D(self.emb_dims[1] + self.emb_dims[1], self.emb_dims[1], kernel_size=kernel_size)   # 24 (up1) + 24 (enc1) = 48 channels

        # Output projection — K channels, one per assay
        K = len(self.init_args["output_assays"])
        self.out_conv = nn.Conv1d(self.emb_dims[1], K, kernel_size=1)

        # Non-negative activation for single-molecule accessibility
        self.fiber_act = nn.Softplus()

    def forward(self, x, ref_dna=None, fiber_coverage=None, *args, **kwargs):
        """
        Args:
            x: Fiber features tensor of shape [B, C_fiber, L, N]
            ref_dna: Optional reference sequence tensor of shape [B, 4, L]
            fiber_coverage: Per-position fiber count tensor of shape [B, L]
        """
        if fiber_coverage is None:
            raise ValueError("Forward pass requires 'fiber_coverage' tensor when decoder_type is 'avg_n'.")

        B, C, L, N = x.shape

        x = x.permute(0,3,1,2).reshape(B * N, self.num_input_features, L)
        x = self.fiber_encoder(x).reshape(B, N, self.emb_dims[0], L).permute(0,2,3,1)

        # 1. Process Reference DNA sequence if enabled
        if self.dna_type == "ref":
            if ref_dna is None:
                raise ValueError("Model configured with dna_type='ref', but ref_dna=None was provided.")

            dna_feats = self.dna_encoder(ref_dna)  # [B, emb_dim[0], L]
            dna_feats_expanded = dna_feats.unsqueeze(-1).expand(-1, -1, -1, N)
            fused_x = torch.cat([x, dna_feats_expanded], dim=1)  # [B, C + emb_dim[0], L, N]
        else:
            fused_x = x

        # 2. Flatten Batch and Fiber Dimensions: [B * N, C_total, L]
        in_channels = fused_x.shape[1]
        x_flat = fused_x.permute(0, 3, 1, 2).reshape(B * N, in_channels, L)

        # 3. Encoder Pass
        e1 = self.enc1(x_flat)  # [B*N, 24, L]
        p1 = self.pool1(e1)     # [B*N, 24, L/2]

        e2 = self.enc2(p1)      # [B*N, 48, L/2]
        p2 = self.pool2(e2)     # [B*N, 48, L/4]

        e3 = self.enc3(p2)      # [B*N, 96, L/4]
        p3 = self.pool3(e3)     # [B*N, 96, L/8]

        # 4. Transformer Bottleneck Pass
        # Reshape [B*N, 96, L/8] -> [B*N, L/8, 96] for sequence attention
        tf_in = p3.permute(0, 2, 1)

        # Add Sinusoidal Positional Encoding
        tf_in = self.pos_encoder(tf_in)

        # Self-Attention
        tf_out = self.transformer_bottleneck(tf_in)

        # Reshape back to [B*N, 96, L/8] for UNet Decoder
        b = tf_out.permute(0, 2, 1)

        # 5. Decoder Pass
        # Level 3 (L/8 -> L/4)
        u3 = self.up3(b)
        if u3.shape[-1] != e3.shape[-1]:
            u3 = F.pad(u3, (0, e3.shape[-1] - u3.shape[-1]))
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        # Level 2 (L/4 -> L/2)
        u2 = self.up2(d3)
        if u2.shape[-1] != e2.shape[-1]:
            u2 = F.pad(u2, (0, e2.shape[-1] - u2.shape[-1]))
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        # Level 1 (L/2 -> L)
        u1 = self.up1(d2)
        if u1.shape[-1] != e1.shape[-1]:
            u1 = F.pad(u1, (0, e1.shape[-1] - u1.shape[-1]))
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        # 6. Output Projection & Fiber Averaging
        K = self.out_conv.out_channels
        out_flat = self.out_conv(d1)               # [B*N, K, L]
        raw_fibers = out_flat.view(B, N, K, L).permute(0, 2, 3, 1)  # [B, K, L, N]

        processed_fibers = self.fiber_act(raw_fibers)               # [B, K, L, N]
        cov = fiber_coverage.float().clamp(min=1).unsqueeze(1)      # [B, 1, L]
        y = processed_fibers.sum(-1) / cov                          # [B, K, L]

        return y, processed_fibers

#--------------------------------------------------------------------------------------------------
# Model Selection Factory

def model_selector(model_arg, args):
    if isinstance(args, dict):
        args = SimpleNamespace(**args)
    model_name = model_arg.lower()

    if model_name == "base01":
        return Base01DebugModel(
                    input_flags=args.input_flags,
                    dna_type=args.dna_type,
                    kernel_size=args.kernel_size,
                    output_assays=args.output_assays,
                )

    elif model_name == "unet03":
        return UNet03ConvTransformerWithDNA(
                    input_flags=args.input_flags,
                    dna_type=args.dna_type,
                    kernel_size=args.kernel_size,
                    max_len=args.context_length,
                    emb_dims=args.emb_dims,
                    tf_heads=args.tf_heads,
                    tf_layers=args.tf_layers,
                    output_assays=args.output_assays,
                )

    raise NotImplementedError(f"Model not implemented: {model_arg}")


#--------------------------------------------------------------------------------------------------
# Testing

def tester():
    pass

if __name__ == "__main__":
    tester()
