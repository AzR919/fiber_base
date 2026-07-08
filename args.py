"""
File for argparsing
"""

import os
import sys
import argparse

def get_args():

    """Argument parser"""
    parser = argparse.ArgumentParser(
        description="Fiber seq test runs",
    )

    # Data
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--fiber_data_path", type=str,
                           default="/home/azr")
    data_group.add_argument("--other_data_path", type=str,
                           default="/home/azr")
    data_group.add_argument("--fiber_data_path_b", type=str, default=None,
                           help="Second Fiber-seq CRAM for mixed-cell experiments")
    data_group.add_argument("--other_data_path_b", type=str, default=None,
                           help="Second target BigWig for mixed-cell experiments")
    data_group.add_argument("--mix_fraction", type=float, default=1.0,
                           help="Weight for cell A in [0, 1]; target and fiber "
                                "subsample use alpha*A + (1-alpha)*B")
    data_group.add_argument("--context_length", type=int,
                           default=2048)
    data_group.add_argument("--fibers_per_entry", type=int,
                             default=32)
    data_group.add_argument("--input_flags", type=int, nargs="+", default=[1,1,1,1,1],
                            help="binary indicators of what to use as input." \
                            "0: m6a, 1: cpg, 2: msp, 3: nuc, 4: fire_msp")
    data_group.add_argument("--m6a", type=int, default=1, choices=[0, 1])
    data_group.add_argument("--cpg", type=int, default=1, choices=[0, 1])
    data_group.add_argument("--msp", type=int, default=1, choices=[0, 1])
    data_group.add_argument("--nuc", type=int, default=1, choices=[0, 1])
    data_group.add_argument("--fire_msp", type=int, default=1, choices=[0, 1])
    data_group.add_argument("--num_input_features", type=int, default=0,
                            help="computed at runtime")

    # Model
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--model", type=str,
                             default="base")
    model_group.add_argument("--d_model", type=int,
                             default=32)
    model_group.add_argument("--decoder_type", type=str,
                             default="avg")
    model_group.add_argument("--kernel_size", type=int,
                             default=15)
    model_group.add_argument("--dilation", type=int,
                             default=1)

    # Train
    trainer_group = parser.add_argument_group("Model Architecture")
    trainer_group.add_argument("--epochs", type=int,
                             default=2)
    trainer_group.add_argument("--iters_per_epoch", type=int,
                             default=1024)
    trainer_group.add_argument("--batch_size", type=int,
                             default=32)


    # I/O
    io_group = parser.add_argument_group("Model I/O")
    io_group.add_argument("--res_dir", type=str,
                          default="./results",
                          help="Directory to save trained models")
    io_group.add_argument("--name_suffix", type=str, default="",
                         help="Suffix to append to auto-generated model name")

    # misc
    misc_group = parser.add_argument_group("miscellaneous arguments")
    misc_group.add_argument("--debug", "-D", action='store_true',
                            help='Enable debug mode with extra logging')
    misc_group.add_argument('--seed', type=int, default=919)

    parsed_args = parser.parse_args()
    # -------------------------------------------------------------------------
    # Recovery Code: Unpack the string back into your 5-element list of ints
    # -------------------------------------------------------------------------
    if "sweep" in parsed_args.name_suffix.lower():
        parsed_args.input_flags = [parsed_args.m6a, parsed_args.cpg, parsed_args.msp, parsed_args.nuc, parsed_args.fire_msp]
    parsed_args.num_input_features = sum(parsed_args.input_flags)

    if parsed_args.mix_fraction < 0.0 or parsed_args.mix_fraction > 1.0:
        parser.error("--mix_fraction must be between 0 and 1")

    mixed_paths = [parsed_args.fiber_data_path_b, parsed_args.other_data_path_b]
    if any(p is not None for p in mixed_paths) and not all(p is not None for p in mixed_paths):
        parser.error(
            "Mixed-cell mode requires both --fiber_data_path_b and --other_data_path_b"
        )

    return parsed_args


#--------------------------------------------------------------------------------------------------
# testing

def tester():
    pass

if __name__=="__main__":

    tester()
