"""
Main file for argparsing

"""

import argparse

def get_arg():

    """Argument parser"""
    parser = argparse.ArgumentParser(
        description="Fiber seq test runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--data_path", type=str,
                           default="/home/azr")
    data_group.add_argument("--input_window", type=int,
                           default=2048)


    # Model
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--model", type=str,
                             default="base")

    # I/O
    io_group = parser.add_argument_group("Model I/O")
    io_group.add_argument("--res_dir", type=str,
                          default="../results",
                          help="Directory to save trained models")
    io_group.add_argument("--name_suffix", type=str, default="",
                         help="Suffix to append to auto-generated model name")

    # misc
    misc_group = parser.add_argument_group("miscellaneous arguments")
    misc_group.add_argument("--debug", "-D", action='store_true',
                            help='Enable debug mode with extra logging')
    misc_group.add_argument('--seed', type=int, default=919)

    return parser.parse_args()
