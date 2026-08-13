"""
Main file
"""

import os
import sys

import yaml

from args import get_args
from data_utils import fiber_data_iterator
from eval_dataset import MixedCellFiberDataset
from trainer import Trainer
from models import model_selector
from utils import *

#--------------------------------------------------------------------------------------------------
# Main

def main():

    args = get_args()
    set_seed(args.seed)

    save_str = create_save_str(args)
    res_dir = os.path.join(args.res_dir, save_str)
    os.makedirs(res_dir, exist_ok=True)
    save_slurm_script(res_dir, args.script_path)

    kwargs = {
        "metadata": args.metadata,
        "fibers_per_entry": args.fibers_per_entry,
        "context_length": args.context_length,
        "iters_per_epoch": args.iters_per_epoch,
        "input_flags": args.input_flags,
        "seed": args.seed,
        "return_dna": args.dna_type != "none",
        "bulk_name": args.bulk_name
    }

    train_data_iterator = fiber_data_iterator(mode="train", **kwargs)
    val_data_iterator = fiber_data_iterator(mode="val", **kwargs)

    model = model_selector(args.model, args)
    input_size = (args.batch_size, sum(args.input_flags), args.context_length, args.fibers_per_entry)
    print_model_summary(model, input_size)

    trainer = Trainer(model, train_data_iterator, val_data_iterator, args.eval_config_path,
                      epochs=args.epochs, batch_size=args.batch_size,
                      run_name=get_config_names_str(args), config=args)

    trainer.train(save_dir=res_dir)

if __name__=="__main__":
    main()
