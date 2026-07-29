"""
Main file
"""

import os
import sys

import yaml

from args import get_args
from data_utils import fiber_data_iterator
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

    with open(args.data_config, "r") as f:
        data_config = yaml.safe_load(f)

    kwargs = {
        "metadata": data_config["metadata"],
        "fibers_per_entry": args.fibers_per_entry,
        "context_length": args.context_length,
        "iters_per_epoch": args.iters_per_epoch,
        "input_flags": args.input_flags,
        "seed": args.seed,
        "return_dna": args.return_dna
    }

    train_data_iterator = fiber_data_iterator(mode="train", **kwargs)
    val_data_iterator = fiber_data_iterator(mode="val", **kwargs)

    model = model_selector(args.model, args)

    trainer = Trainer(model, train_data_iterator, val_data_iterator,
                      epochs=args.epochs, batch_size=args.batch_size,
                      run_name=get_config_names_str(args), config=args)

    trainer.train(save_dir=res_dir)

if __name__=="__main__":
    main()
