"""
Main model file
"""
import os
import inspect

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
    def __init__(self, input_flags, dna_type):
        super().__init__()
        self.init_args = {
            "input_flags": input_flags,
            "dna_type": dna_type,
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

        # Robustly extract input_flags
        input_flags = init_args.get("input_flags")

        if input_flags is None:
            raise ValueError(f"Checkpoint at {filepath} does not contain 'input_flags'.")

        # DYNAMIC ARGUMENT FILTERING
        # Get the signature of the current class's __init__ method
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self", "args", "kwargs"}

        # Filter config to only include arguments that the constructor actually accepts
        init_kwargs = {k: v for k, v in init_args.items() if k in valid_params}

        # Instantiate the model
        try:
            model = cls(**init_kwargs)
        except TypeError as e:
            raise TypeError(f"Failed to instantiate {cls.__name__}. Missing required args? "
                            f"Expected: {list(valid_params)}, Got: {list(init_kwargs.keys())}. Error: {e}")

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

#--------------------------------------------------------------------------------------------------
# Concrete Model Implementation

class Base01DebugModel(BaseModel):

    def __init__(self, input_flags, dna_type, decoder_type="avg_n", kernel_size=15):
        super().__init__(input_flags, dna_type)

        # Store init args for automatic saving/loading in BaseModel
        self.init_args["kernel_size"] = kernel_size

        self.num_input_features = sum(input_flags)
        self.decoder_type = decoder_type
        self.kernel_size = kernel_size

        channels = [self.num_input_features, 2, 1]
        dilations = [1, 2]

        layers = []
        for i in range(2):
            layers.append(
                ResidualBlock1D(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    kernel_size=kernel_size,
                    dilation=dilations[i]
                )
            )
        self.fiber_conv = nn.Sequential(*layers)

        implemented_decoders = ["avg", "sum", "avg_n"]
        if decoder_type not in implemented_decoders:
            raise NotImplementedError(f"decoder_type not implemented: {decoder_type}")

        self.final_layer = nn.Sequential(nn.GELU())

    def forward(self, x, *args, **kwargs):
        B, C, L, N = x.shape

        # Flatten batch and fiber dimensions: (B * N, C, L)
        x_flat = x.permute(0, 3, 1, 2).reshape(B * N, C, L)

        # Process through the residual bottleneck
        out_flat = self.fiber_conv(x_flat)

        # Reshape back to original dimensions: (B, L, N)
        processed_fibers = out_flat.view(B, N, 1, L).permute(0, 2, 3, 1).squeeze(1)

        if self.decoder_type == "sum":
            y = torch.sum(processed_fibers, dim=-1)
        elif self.decoder_type == "avg":
            y = torch.mean(processed_fibers, dim=-1)
        elif self.decoder_type == "avg_n":
            y = torch.sum(processed_fibers, dim=-1) / kwargs["n_fibers"].unsqueeze(-1)
        else:
            raise NotImplementedError(f"decoder_type not implemented in forward pass: {self.decoder_type}")

        y_final = self.final_layer(y)
        return y_final, processed_fibers

class Deep01ResConv1dBlock(BaseModel):

    def __init__(self, input_flags, dna_type, decoder_type="avg_n", kernel_size=15):
        super().__init__(input_flags, dna_type)

        # Store init args for automatic saving/loading in BaseModel
        self.init_args["kernel_size"] = kernel_size

        self.num_input_features = sum(input_flags)
        self.decoder_type = decoder_type
        self.kernel_size = kernel_size

        channels = [self.num_input_features, 32, 64, 64, 32, 1]
        dilations = [1, 2, 4, 8, 16]

        layers = []
        for i in range(5):
            layers.append(
                ResidualBlock1D(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    kernel_size=kernel_size,
                    dilation=dilations[i]
                )
            )
        self.fiber_conv = nn.Sequential(*layers)

        implemented_decoders = ["avg", "sum", "avg_n"]
        if decoder_type not in implemented_decoders:
            raise NotImplementedError(f"decoder_type not implemented: {decoder_type}")

        self.final_layer = nn.Sequential(nn.GELU())

    def forward(self, x, *args, **kwargs):
        B, C, L, N = x.shape

        # Flatten batch and fiber dimensions: (B * N, C, L)
        x_flat = x.permute(0, 3, 1, 2).reshape(B * N, C, L)

        # Process through the residual bottleneck
        out_flat = self.fiber_conv(x_flat)

        # Reshape back to original dimensions: (B, L, N)
        processed_fibers = out_flat.view(B, N, 1, L).permute(0, 2, 3, 1).squeeze(1)

        if self.decoder_type == "sum":
            y = torch.sum(processed_fibers, dim=-1)
        elif self.decoder_type == "avg":
            y = torch.mean(processed_fibers, dim=-1)
        elif self.decoder_type == "avg_n":
            y = torch.sum(processed_fibers, dim=-1) / kwargs["n_fibers"].unsqueeze(-1)
        else:
            raise NotImplementedError(f"decoder_type not implemented in forward pass: {self.decoder_type}")

        y_final = self.final_layer(y)
        return y_final, processed_fibers

class TransformerFiber1DModel(BaseModel):
    """
    A Transformer Encoder model for high-context window genomic sequence imputation.
    Flattens single-cell tracks and computes dependencies globally across sequence lengths.
    """
    def __init__(self, input_flags, dna_type, decoder_type="avg_n", d_model=64, n_head=4, num_layers=4, dim_feedforward=128, max_len=6000):
        super().__init__(input_flags, dna_type)

        # Save structural parameters for checkpoint serialization blueprinting
        self.init_args.update({
            "d_model": d_model,
            "nhead": n_head,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "max_len": max_len
        })

        self.num_input_features = sum(input_flags)
        self.decoder_type = decoder_type

        # 1. Feature Map Projection Input Layer
        self.input_projection = nn.Linear(self.num_input_features, d_model)
        self.pos_encoder = PositionalEncoding1D(d_model, max_len=max_len)

        # 2. Transformer Encoder Engine
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. Output Projection Layer back to a 1D probability/signal stream
        self.output_projection = nn.Linear(d_model, 1)

        implemented_decoders = ["avg", "sum", "avg_n"]
        if decoder_type not in implemented_decoders:
            raise NotImplementedError(f"decoder_type not implemented: {decoder_type}")

        self.final_layer = nn.Sequential(nn.GELU())

    def forward(self, x, *args, **kwargs):
        B, C, L, N = x.shape

        # 1. Standard structural rearrangement to process context length elementwise
        # [B, C, L, N] -> permute -> [B * N, L, C]
        x_flat = x.permute(0, 3, 2, 1).reshape(B * N, L, C)

        # 2. Project channel configurations into embedding tracks & add positional information
        x_proj = self.input_projection(x_flat)
        x_encoded = self.pos_encoder(x_proj)

        # 3. Evaluate transformer contextual weights sequence-wide
        transformer_out = self.transformer_encoder(x_encoded) # [B * N, L, d_model]

        # 4. Collapse dimension mappings back to single accessibility vectors
        out_flat = self.output_projection(transformer_out).squeeze(-1) # [B * N, L]

        # 5. Map directly back to expected canonical workspace orientation: [B, L, N]
        processed_fibers = out_flat.view(B, N, L).permute(0, 2, 1)

        # 6. Apply backward-compatible resolution decoders
        if self.decoder_type == "sum":
            y = torch.sum(processed_fibers, dim=-1)
        elif self.decoder_type == "avg":
            y = torch.mean(processed_fibers, dim=-1)
        elif self.decoder_type == "avg_n":
            n_fibers = kwargs.get("n_fibers")
            if n_fibers is None:
                raise ValueError("Forward pass requires 'n_fibers' tensor when decoder_type is 'avg_n'.")
            y = torch.sum(processed_fibers, dim=-1) / n_fibers.unsqueeze(-1)
        else:
            raise NotImplementedError(f"decoder_type not implemented in forward pass: {self.decoder_type}")

        y_final = self.final_layer(y)
        return y_final, processed_fibers

class UNet01Conv1d(BaseModel):
    """
    1D U-Net Model for Single-Molecule Genomic Sequence Processing.
    Compresses spatial resolution to capture multi-scale context while
    dramatically reducing memory footprint for long context windows (5000 bp).
    """
    def __init__(self, input_flags, dna_type, decoder_type="avg_n", kernel_size=15):
        super().__init__(input_flags, dna_type)

        assert decoder_type == "avg_n", f"UNet01Conv1d only supports 'avg_n' decoder, got '{decoder_type}'."

        self.init_args["kernel_size"] = kernel_size
        self.num_input_features = sum(input_flags)
        self.kernel_size = kernel_size

        # --- U-Net Architecture Backbone ---
        # Encoder (Downsampling)
        self.enc1 = DoubleConv1D(self.num_input_features, 32, kernel_size=kernel_size)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # L -> L/2

        self.enc2 = DoubleConv1D(32, 64, kernel_size=kernel_size)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # L/2 -> L/4

        # Bottleneck
        self.bottleneck = DoubleConv1D(64, 128, kernel_size=kernel_size)

        # Decoder (Upsampling & Skip Connections)
        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv1D(128, 64, kernel_size=kernel_size)

        self.up1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv1D(64, 32, kernel_size=kernel_size)

        # Output projection back to 1 channel (1D signal)
        self.out_conv = nn.Conv1d(32, 1, kernel_size=1)

        # Non-negative activation applied to individual fiber predictions
        self.fiber_act = nn.Softplus()

    def forward(self, x, n_fibers=None, *args, **kwargs):
        if n_fibers is None:
            raise ValueError("Forward pass requires 'n_fibers' tensor when decoder_type is 'avg_n'.")

        B, C, L, N = x.shape

        # 1. Flatten batch and fiber dimensions: [B * N, C, L]
        x_flat = x.permute(0, 3, 1, 2).reshape(B * N, C, L)

        # 2. Encoder Pass
        e1 = self.enc1(x_flat)      # [B*N, 32, L]
        p1 = self.pool1(e1)         # [B*N, 32, L/2]

        e2 = self.enc2(p1)          # [B*N, 64, L/2]
        p2 = self.pool2(e2)         # [B*N, 64, L/4]

        # 3. Bottleneck
        b = self.bottleneck(p2)     # [B*N, 128, L/4]

        # 4. Decoder Pass with Skip Connections
        u2 = self.up2(b)            # [B*N, 64, L/2]
        if u2.shape[-1] != e2.shape[-1]:
            u2 = F.pad(u2, (0, e2.shape[-1] - u2.shape[-1]))
        d2 = self.dec2(torch.cat([u2, e2], dim=1))  # [B*N, 64, L/2]

        u1 = self.up1(d2)           # [B*N, 32, L]
        if u1.shape[-1] != e1.shape[-1]:
            u1 = F.pad(u1, (0, e1.shape[-1] - u1.shape[-1]))
        d1 = self.dec1(torch.cat([u1, e1], dim=1))  # [B*N, 32, L]

        # 5. Output projection: [B*N, 1, L]
        out_flat = self.out_conv(d1)

        # 6. Reshape back to expected workspace orientation: [B, L, N]
        raw_fibers = out_flat.view(B, N, 1, L).permute(0, 2, 3, 1).squeeze(1)

        # 7. Apply Softplus FIRST to guarantee individual fiber accessibility is strictly >= 0
        processed_fibers = self.fiber_act(raw_fibers)

        # 8. Aggregate fibers to compute non-negative bulk prediction
        y = torch.sum(processed_fibers, dim=-1) / n_fibers.unsqueeze(-1)

        return y, processed_fibers

#--------------------------------------------------------------------------------------------------
# Model Selection Factory

def model_selector(model_arg, args):
    model_name = model_arg.lower()

    if model_name=="base01":
        return Base01DebugModel(
                    input_flags=args.input_flags,
                    dna_type=args.dna_type,
                    decoder_type=args.decoder_type,
                    kernel_size=args.kernel_size
                )
    elif model_name=="deep01":
        return Deep01ResConv1dBlock(
                    input_flags=args.input_flags,
                    dna_type=args.dna_type,
                    decoder_type=args.decoder_type,
                    kernel_size=args.kernel_size
                )

    elif model_name == "trans01":
        return TransformerFiber1DModel(
                    input_flags=args.input_flags,
                    dna_type=args.dna_type,
                    decoder_type=args.decoder_type,
                    d_model=args.d_model,
                    n_head=args.n_head,
                    num_layers=args.num_layers,
                    dim_feedforward=args.dim_feedforward,
                    max_len=args.context_length
                )

    elif model_name=="unet01":
            return UNet01Conv1d(
                        input_flags=args.input_flags,
                        dna_type=args.dna_type,
                        kernel_size=args.kernel_size
                    )

    raise NotImplementedError(f"Model not implemented: {model_arg}")


#--------------------------------------------------------------------------------------------------
# Testing

def tester():
    B, C_in, L, N = 16, 5, 2048, 200
    decoder_type = "sum"
    kernel_size = 15
    input_flags = [1, 1, 1, 1, 1]

    test_model = Deep01ResConv1dBlock(
        input_flags=input_flags,
        dna_type="none",
        decoder_type=decoder_type,
        kernel_size=kernel_size
    )

    test_inp = torch.rand((B, C_in, L, N))
    n_fibers = torch.full((B,), 15)
    test_out, processed = test_model(test_inp, n_fibers=n_fibers)

    print(f"Output shape: {test_out.shape}")
    print(f"Processed fibers shape: {processed.shape}")

    # Test Save & Load functionality
    save_dir = "./test_checkpoints"
    test_model.save_model(save_dir, epoch=1, external_config={"lr": 1e-3})

    ckpt_path = os.path.join(save_dir, "Model_epoch_1.pt")
    loaded_model, cfg = Deep01ResConv1dBlock.load_model(ckpt_path)

    print("Successfully tested model save and load!")

    # Cleanup test output
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    if os.path.exists(save_dir):
        os.rmdir(save_dir)

if __name__ == "__main__":
    tester()
