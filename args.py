"""
File for argparsing and configuration management.
Priority Order: Command Line Arguments > Config JSON/YAML Files > Parser Defaults
"""

import os
import sys
import argparse

from utils import *

#--------------------------------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Fiber-seq and Bulk Chromatin Accessibility Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config Files (Metadata)
    config_group = parser.add_argument_group("Metadata Config Files")
    config_group.add_argument("--data_config", type=str, default=None, required=True,
                             help="Path to JSON or YAML file containing dataset metadata and configuration")
    config_group.add_argument("--model_config", type=str, default=None,
                             help="Path to JSON or YAML file containing model configuration")
    config_group.add_argument("--train_config", type=str, default=None,
                             help="Path to JSON or YAML file containing trainer configuration")
    config_group.add_argument("--eval_config_path", type=str, default=None,
                             help="Path to JSON or YAML file containing evaluation configuration")

    # Data - Paths & Metadata Defaults
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--fasta_path", type=str,
                            default="/home/azr/projects/def-maxwl/azr/data/misc/hg38.fa")
    data_group.add_argument("--ccre_path", type=str,
                            default="/home/azr/projects/def-maxwl/azr/data/misc/grch38_ccres.bed")
    data_group.add_argument("--train_chrs", type=str, nargs="+", default=["chr20"])
    data_group.add_argument("--val_chrs", type=str, nargs="+", default=["chr21"])
    data_group.add_argument("--fiber_base_path", type=str, default="")
    data_group.add_argument("--bulk_base_path", type=str, default="")
    data_group.add_argument("--bulk_name", type=str, default="Not_set")

    # Dataset Parameters
    data_group.add_argument("--context_length", type=int, default=4096)
    data_group.add_argument("--fibers_per_entry", type=int, default=200)


    # Model
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--model", type=str, default="base")
    model_group.add_argument("--d_model", type=int, default=32)
    model_group.add_argument("--decoder_type", type=str, default="avg_n")
    model_group.add_argument("--kernel_size", type=int, default=15)
    model_group.add_argument("--dilation", type=int, default=1)
    model_group.add_argument("--input_flags", type=int, nargs="+", default=[1, 1, 1, 1, 1],
                                help="Binary indicators list [m6a, cpg, msp, nuc, fire_msp]")
    model_group.add_argument("--dna_type", type=str, default="none", choices=["none", "ref", "fiber", "both"],
                            help="dna_type to use as input")

    # Individual input feature flags for hyperparameter sweeps
    model_group.add_argument("--use_individual_input_flags", action="store_true",
                            help="If set, overrides --input_flags with values from --m6a, --cpg, etc.")
    model_group.add_argument("--m6a", type=int, default=1, choices=[0, 1])
    model_group.add_argument("--cpg", type=int, default=1, choices=[0, 1])
    model_group.add_argument("--msp", type=int, default=1, choices=[0, 1])
    model_group.add_argument("--nuc", type=int, default=1, choices=[0, 1])
    model_group.add_argument("--fire_msp", type=int, default=1, choices=[0, 1])
    model_group.add_argument("--num_input_features", type=int, default=0,
                            help="Computed at runtime based on active flags")

    # Train
    trainer_group = parser.add_argument_group("Training Configuration")
    trainer_group.add_argument("--epochs", type=int, default=2)
    trainer_group.add_argument("--iters_per_epoch", type=int, default=1000)
    trainer_group.add_argument("--batch_size", type=int, default=16)
    trainer_group.add_argument("--lr", type=float, default=1e-4)

    # I/O
    io_group = parser.add_argument_group("Model I/O")
    io_group.add_argument("--res_dir", type=str, default="./results",
                          help="Directory to save trained models and final results")
    io_group.add_argument("--name_prefix", type=str, default=None,
                              help="prefix to append to auto-generated model name")
    io_group.add_argument("--name_suffix", type=str, default=None,
                          help="Suffix to append to auto-generated model name")

    # Misc
    misc_group = parser.add_argument_group("Miscellaneous Arguments")
    misc_group.add_argument("--debug", "-D", action='store_true',
                            help="Enable debug mode with extra logging")
    misc_group.add_argument("--seed", type=int, default=919)
    misc_group.add_argument("--script_path", type=str, default="./slurm_batch_command.sh",
                              help="slurm batch script to save with the model")

    # Step 1: Parse arguments passed via CLI
    parsed_args = parser.parse_args()

    # Step 2: Track explicit CLI flags so config files don't overwrite user CLI commands
    cli_args_set = set()
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            # Strip leading '-' or '--', then take whatever comes before '=' (if any)
            clean_arg = arg.lstrip("-").split("=")[0]
            if clean_arg:
                cli_args_set.add(clean_arg)

    # Default metadata structure
    parsed_args.metadata = {
        "fasta_path": parsed_args.fasta_path,
        "ccre_path": parsed_args.ccre_path,
        "train_chrs": parsed_args.train_chrs,
        "val_chrs": parsed_args.val_chrs,
        "fiber_base_path": parsed_args.fiber_base_path,
        "bulk_base_path": parsed_args.bulk_base_path,
        "cell_types": {}
    }

    # Step 3: Load data config JSON/YAML file if provided
    if parsed_args.data_config:
        data_cfg = load_config_file(parsed_args.data_config)

        # Merge metadata section from config file
        if "metadata" in data_cfg:
            meta_cfg = data_cfg["metadata"]
            for key in ["fasta_path", "ccre_path", "train_chrs", "val_chrs", "fiber_base_path", "bulk_base_path", "cell_types"]:
                if key in meta_cfg and key not in cli_args_set:
                    parsed_args.metadata[key] = meta_cfg[key]
                    if hasattr(parsed_args, key):
                        setattr(parsed_args, key, meta_cfg[key])

        # Merge non-metadata data configuration parameters
        for key, val in data_cfg.items():
            if key != "metadata" and hasattr(parsed_args, key) and key not in cli_args_set:
                setattr(parsed_args, key, val)

    # Load model and train config JSON/YAML files if provided
    for cfg_path in [parsed_args.model_config, parsed_args.train_config]:
        if cfg_path:
            cfg_dict = load_config_file(cfg_path)
            for key, val in cfg_dict.items():
                if hasattr(parsed_args, key) and key not in cli_args_set:
                    setattr(parsed_args, key, val)
                elif not hasattr(parsed_args,key):
                    setattr(parsed_args, key, val)

    # Step 4: Reconcile sweep feature flags vs input_flags list
    indiv_flags = ["m6a", "cpg", "msp", "nuc", "fire_msp"]
    any_indiv_in_cli = any(flag in cli_args_set for flag in indiv_flags)

    if parsed_args.use_individual_input_flags or any_indiv_in_cli:
        parsed_args.input_flags = [
            parsed_args.m6a,
            parsed_args.cpg,
            parsed_args.msp,
            parsed_args.nuc,
            parsed_args.fire_msp
        ]

    parsed_args.num_input_features = sum(parsed_args.input_flags)

    return parsed_args

#--------------------------------------------------------------------------------------------------
# Testing

def tester():
    args = get_args()
    print("--- Parsed Arguments & Metadata ---")
    print(f"Context Length          : {args.context_length}")
    print(f"Fibers per Entry        : {args.fibers_per_entry}")
    print(f"Return DNA Tensors      : {args.dna_type}")
    print(f"Active Input Flags      : {args.input_flags}")
    print(f"Num Input Features      : {args.num_input_features}")
    print("\n--- Metadata Dict ---")
    for k, v in args.metadata.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    tester()
