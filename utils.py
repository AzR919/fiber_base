"""
Common utility functions for experiment tracking, seeding, and training.
"""

import os
import json
import yaml
import random
import shutil
import datetime
import numpy as np

import torch
import torch.nn as nn

from torchinfo import summary as torchinfo_summary

#--------------------------------------------------------------------------------------------------
# Reproducibility & Environment Setup

def save_slurm_script(res_dir, slurm_script_path):
    """
    Copies the active Slurm batch submission script into the output result directory.
    """
    os.makedirs(res_dir, exist_ok=True)
    try:
        destination = os.path.join(res_dir, "submitted_sbatch_script.sh")
        shutil.copy(slurm_script_path, destination)
        print(f"[Slurm Tracker] Successfully copied submission script to: {destination}")
    except:
        print(f"[Slurm Tracker] Failed to copy submission script [{slurm_script_path}] to: {destination}")

def set_seed(seed: int = 919):
    """
    Sets seeds for Python, NumPy, and PyTorch across CPU and GPU.
    Enforces deterministic CUDA operations for exact reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """
    Worker init function for PyTorch DataLoader to ensure deterministic multi-processing sampling.
    Pass as: DataLoader(..., worker_init_fn=seed_worker)
    """
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

#--------------------------------------------------------------------------------------------------
# File Utilities & Experiment Tracking

def load_config_file(config_path):
    """Loads JSON or YAML configuration files safely."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        with open(config_path, "r") as f:
            return json.load(f)

def load_metapaths(path: str) -> dict:
    return load_config_file(path)

def resolve_config_with_metapaths(config: dict, metapaths: dict) -> dict:
    assay = config.get("assay")
    if assay not in metapaths["assay_base_paths"]:
        raise ValueError(f"assay '{assay}' not in metapaths.assay_base_paths")

    metadata = {
        "fasta_path":      metapaths["fasta_path"],
        "ccre_path":       metapaths["ccre_path"],
        "fiber_base_path": config.get("fiber_base_path") or metapaths["assay_base_paths"]["fiber_seq"],
        "bulk_base_path":  config.get("bulk_base_path")  or metapaths["assay_base_paths"][assay],
        "train_chrs":      config.get("train_chrs")      or metapaths["train_chrs"],
        "val_chrs":        config.get("val_chrs")        or metapaths["val_chrs"],
        "cell_types":      {},
    }

    mp_cts = metapaths["cell_types"]
    for ct_name, ct_cfg in (config.get("cell_types") or {}).items():
        if ct_name not in mp_cts:
            raise ValueError(f"Cell type '{ct_name}' not in metapaths.cell_types")
        if mp_cts[ct_name].get(assay) is None:
            raise ValueError(f"Assay '{assay}' not available for cell type '{ct_name}' in metapaths")
        entry = {
            "fibers": mp_cts[ct_name]["fiber_seq"],
            "bulk":   mp_cts[ct_name][assay],
        }
        if ct_cfg and "ratio" in ct_cfg:
            entry["ratio"] = ct_cfg["ratio"]
        metadata["cell_types"][ct_name] = entry

    return metadata

def get_config_names_str(args) -> str:
    """
    Extracts filenames (without paths or extensions) from provided config arguments
    and joins them with underscores.
    """
    config_keys = ["data_config", "model_config", "train_config", "eval_config_path"]
    config_names = [args.name_prefix] if args.name_prefix is not None else []

    for key in config_keys:
        cfg_path = getattr(args, key, None)
        if cfg_path:
            # Extract filename without extension (e.g., 'path/to/data_config.yaml' -> 'data_config')
            base_name = os.path.splitext(os.path.basename(cfg_path))[0]
            config_names.append(base_name)

    if args.name_suffix is not None:
        config_names.append(args.name_suffix)

    # Join extracted config names (e.g., "data_config_model_config_train_config")
    return "_".join(config_names)

def create_save_str(args) -> str:
    """Generates a structured, unique run identifier including timestamp and concatenated config names."""
    now = datetime.datetime.now().strftime("%y-%m-%d_T%H-%M-%S")
    configs_str = get_config_names_str(args)

    # Build component list, filtering out empty strings
    components = [now]
    if configs_str:
        components.append(configs_str)

    return "_".join(components)

class AverageMeter:
    """Computes and stores the running average and current value of metrics during training."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count

def unpack_batch(batch, device):
    """Extracts batch dictionary elements and moves tensors to target device."""
    fiber_features = batch["fiber_features"].to(device)
    target = batch["target_bulk"].to(device)
    n_fibers = batch["n_fibers"].to(device)

    kwargs = {"n_fibers": n_fibers}

    if "fiber_coverage" in batch:
        kwargs["fiber_coverage"] = batch["fiber_coverage"].to(device)
    if "ref_dna" in batch:
        kwargs["ref_dna"] = batch["ref_dna"].to(device)
    if "fiber_dna_tensor" in batch:
        kwargs["fiber_dna_tensor"] = batch["fiber_dna_tensor"].to(device)

    return fiber_features, target, kwargs

def count_parameters(model: nn.Module) -> int:
    """Returns total count of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model: nn.Module, input_size: tuple = (16, 5, 2048, 200)):
    """
    Pretty-prints model architecture summary using torchinfo if available,
    otherwise falls back to parameter count and string representation.
    """
    print("\n" + "=" * 60)
    print(f" MODEL SUMMARY: {model.__class__.__name__}")
    print("=" * 60)
    print(f" Total Trainable Parameters: {count_parameters(model):,}\n")

    try:
        summary_str = torchinfo_summary(
            model,
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params", "kernel_size"],
            row_settings=["var_names"],
            verbose=0,
            n_fibers=torch.ones(input_size[0]),
            fiber_coverage=torch.ones(input_size[0], input_size[2]),
            ref_dna=torch.ones((input_size[0],4,input_size[-2]))
        )
        print(summary_str)
    except Exception as e:
        print(f"Notice: torchinfo summary could not run on sample input shape {input_size}. ({e})")
        print(model)
    print("=" * 60 + "\n")

def print_gpu_memory(stage=""):
    if torch.cuda.is_available():
        # Convert bytes to Gigabytes for readability
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)

        print(f"[{stage}] Allocated: {allocated:.2f} GB | Peak Allocated: {max_allocated:.2f} GB | Reserved: {reserved:.2f} GB")


#--------------------------------------------------------------------------------------------------
# Testing

def tester():
    set_seed(919)
    print("Testing utils module...")

    # Test Meter
    meter = AverageMeter()
    meter.update(2.5, n=2)
    meter.update(5.0, n=1)
    print(f"AverageMeter average: {meter.avg:.2f} (Expected: 3.33)")

    # Test Dummy Model for Pretty Print Summary
    dummy_model = nn.Sequential(
        nn.Conv1d(5, 32, kernel_size=15, padding=7),
        nn.GELU(),
        nn.Conv1d(32, 1, kernel_size=1)
    )
    print_model_summary(dummy_model, input_size=(16, 5, 2048, 200))

if __name__ == "__main__":
    tester()
