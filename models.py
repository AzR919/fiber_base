"""
Main model file
"""
import os
import torch
import torch.nn as nn


#--------------------------------------------------------------------------------------------------
# Base Model Class with Reusable Save / Load Logic

class BaseModel(nn.Module):
    """
    Abstract base class providing unified save and load functionality
    for all fiber-seq models.
    """
    def __init__(self):
        super().__init__()
        self.init_args = {}

    def save_model(self, dir_name, epoch, external_config=None):
        """
        Saves both the model configuration parameters and the state dictionary
        together inside a single bundled file.
        """
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        save_path = os.path.join(dir_name, f"Model_epoch_{epoch}.pt")

        # Gather configuration details
        model_config = dict(external_config) if external_config is not None else {}

        # Merge model initialization arguments
        model_config.update(self.init_args)

        checkpoint_bundle = {
            "model_config": model_config,
            "state_dict": self.state_dict()
        }

        torch.save(checkpoint_bundle, save_path)
        print(f"Model blueprint and weights successfully bundled into: {save_path}")

    @classmethod
    def load_model(cls, filepath, map_location=None):
        """
        Loads a bundled file, extracts the configuration parameters to instantiate
        the exact structural layout, and then loads the model weights.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No bundled checkpoint found at path: {filepath}")

        checkpoint = torch.load(filepath, map_location=map_location)

        if "model_config" not in checkpoint or "state_dict" not in checkpoint:
            raise KeyError("The checkpoint file does not match the expected bundle format.")

        config = checkpoint["model_config"]
        state_dict = checkpoint["state_dict"]

        # Build model instance using stored init arguments matching constructor
        init_kwargs = {k: v for k, v in config.items() if k in getattr(cls, "_init_keys", [])}
        model = cls(**init_kwargs)

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


#--------------------------------------------------------------------------------------------------
# Concrete Model Implementation

class Deep01ResConv1dBlock(BaseModel):
    _init_keys = ["num_input_features", "decoder_type", "kernel_size"]

    def __init__(self, num_input_features=5, decoder_type="avg_n", kernel_size=15):
        super().__init__()

        # Store init args for automatic saving/loading in BaseModel
        self.init_args = {
            "num_input_features": num_input_features,
            "decoder_type": decoder_type,
            "kernel_size": kernel_size,
        }

        self.num_input_features = num_input_features
        self.decoder_type = decoder_type
        self.kernel_size = kernel_size

        channels = [num_input_features, 32, 64, 64, 32, 1]
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


#--------------------------------------------------------------------------------------------------
# Model Selection Factory

def model_selector(model_arg, args):
    model_name = model_arg.lower()

    if model_name=="deep01":
        return Deep01ResConv1dBlock(
            num_input_features=args.num_input_features,
            decoder_type=args.decoder_type,
            kernel_size=args.kernel_size
        )

    raise NotImplementedError(f"Model not implemented: {model_arg}")


#--------------------------------------------------------------------------------------------------
# Testing

def tester():
    B, C_in, L, N = 16, 5, 2048, 200
    decoder_type = "sum"
    kernel_size = 15

    test_model = Deep01ResConv1dBlock(
        num_input_features=C_in,
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