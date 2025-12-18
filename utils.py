"""
Common utility functions

"""

import torch
import random
import numpy as np


def set_seed(seed):

    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch (CPU and GPU)
    torch.manual_seed(seed)

    # If using CUDA, set seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
