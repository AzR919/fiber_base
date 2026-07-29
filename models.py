"""
Main model file
Some older models were deleted to avoid clutter
To find them git search an older push
"""
import os

import torch
import torch.nn as nn

#--------------------------------------------------------------------------------------------------
# Various models

class ResidualBlock1D(nn.Module):
    """
    A 1D Convolutional Residual Block with LayerNorm,
    dynamic dilation/padding, and a 1x1 projection shortcut.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        padding = (dilation * (kernel_size - 1)) // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation
        )
        # LayerNorm expects the normalized shape at the trailing dimensions: (Channels, Length)
        # We specify out_channels; it applies across the channel dimension per locus.
        self.norm = nn.GroupNorm(1, out_channels) # GroupNorm with 1 group is equivalent to LayerNorm for 1D arrays
        self.act = nn.GELU()

        # 1x1 Conv shortcut to project channels if they don't match
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.act(self.shortcut(x) + self.norm(self.conv(x)))

class Deep01ResConv1dBlock(nn.Module):
    def __init__(self, num_input_features=5, decoder_type="avg", kernel_size=15):
        super().__init__()

        self.num_input_features = num_input_features
        self.decoder_type = decoder_type
        self.kernel_size = kernel_size

        # Define the channel architecture you requested
        channels = [num_input_features, 32, 64, 64, 32, 1]
        dilations = [1, 2, 4, 8, 16]

        # Build the sequential deep pipeline using our residual blocks
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
        if decoder_type in implemented_decoders:
            self.decoder_type = decoder_type
        else:
            raise NotImplementedError(f"decoder_type not implemented: {decoder_type}")

        self.final_layer = nn.Sequential(nn.GELU())

    def forward(self, x, *args, **kwargs):
        B, C, L, N = x.shape

        # Flatten batch and fiber dimensions: (B * N, C, L)
        x_flat = x.permute(0, 3, 1, 2).reshape(B * N, C, L)

        # Process through the 5-layer residual bottleneck
        out_flat = self.fiber_conv(x_flat)

        # Reshape back to original dimensions: (B, L, N)
        processed_fibers = out_flat.view(B, N, 1, L).permute(0, 2, 3, 1).squeeze(1)

        if self.decoder_type == "sum":
            y = torch.sum(processed_fibers, dim=-1)
        elif self.decoder_type == "avg":
            y = torch.mean(processed_fibers, dim=-1)
        elif self.decoder_type == "avg_n":
            y = torch.sum(processed_fibers, dim=-1)/kwargs["n_fibers"].unsqueeze(-1)
        else:
            raise NotImplementedError(f"decoder_type not implemented in the forward pass (but passed the check in init): {self.decoder_type}")

        y_final = self.final_layer(y)
        return y_final, processed_fibers

    def save_model(self, dir_name, epoch, external_config=None):
        """
        Saves both the model configuration parameters and the state dictionary
        together inside a single bundled file.

        Args:
            dir_name (str): Target path to save the package (.pt).
            external_config (dict, optional): If your trainer passes a full config dictionary
                                             (containing lr, epochs, etc.), you can pass it here.
                                             If None, it builds a config using the model attributes.
        """
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        save_path = os.path.join(dir_name, f"Model_epoch_{epoch}.pt")

        # 1. Gather configuration details
        if external_config is not None:
            # Copy to prevent changing your trainer state
            model_config = dict(external_config)
        else:
            model_config = {}

        # Ensure critical structural properties are firmly captured
        model_config["num_input_features"] = self.num_input_features
        model_config["decoder_type"] = self.decoder_type
        model_config["kernel_size"] = self.kernel_size

        # 2. Package everything together
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

        Args:
            filepath (str): Path to the bundled .pt checkpoint file.
            map_location (str/torch.device, optional): e.g., 'cpu' or 'cuda' to remap storage.

        Returns:
            model (FiberDeep01ResConv1dBlock): Fully functional model instance.
            config (dict): The configuration dictionary stored inside the checkpoint.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No bundled checkpoint found at path: {filepath}")

        # 1. Unpack the bundle dictionary
        checkpoint = torch.load(filepath, map_location=map_location)

        if "model_config" not in checkpoint or "state_dict" not in checkpoint:
            raise KeyError("The checkpoint file does not match the expected bundle format.")

        config = checkpoint["model_config"]
        state_dict = checkpoint["state_dict"]

        # 2. Build the model structure directly out of the file's configuration metadata
        model = cls(
            num_input_features=config.get("num_input_features", 5),
            decoder_type=config.get("decoder_type", "avg"),
            kernel_size=config.get("kernel_size", 15)
        )

        # 3. Load the parameter matrices
        model.load_state_dict(state_dict)
        print(f"Model successfully reconstituted from: {filepath}")

        return model, config

#--------------------------------------------------------------------------------------------------
# model selection based on cmd arg

def model_selector(model_arg, args):

    model_name = model_arg.lower()

    if model_name=="deep01": return Deep01ResConv1dBlock(args.num_input_features, args.decoder_type, args.kernel_size)

    raise NotImplementedError(f"Model not implemented: {model_arg}")


#--------------------------------------------------------------------------------------------------
# testing

def tester():

    B, C_in, L, N = 16, 4, 2048, 200
    d_model = 128
    decoder_type = "sum"
    kernel_size = 51
    dilation = 1

    test_model = Deep01ResConv1dBlock(C_in, decoder_type, kernel_size)

    test_inp = torch.rand((B, C_in, L, N))
    test_out = test_model(test_inp, dna=None, n_fibers=torch.full((B,), 15))

    pass

if __name__=="__main__":

    tester()
