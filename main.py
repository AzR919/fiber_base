"""
Main file
"""

import os
import wandb

from args import get_args
from data_utils import make_fiber_dataset
from evaluator import run_final_eval
from trainer import Trainer
from models import model_selector
from utils import *


def main():
    args = get_args()
    set_seed(args.seed)

    save_str = create_save_str(args)
    res_dir = os.path.join(args.res_dir, save_str)

    dataset_kwargs = {
        "metadata": args.metadata,
        "fibers_per_entry": args.fibers_per_entry,
        "context_length": args.context_length,
        "iters_per_epoch": args.iters_per_epoch,
        "num_sample_ccres": args.iters_per_epoch,
        "input_flags": args.input_flags,
        "seed": args.seed,
        "dna_type": args.dna_type,
        "bulk_name": args.bulk_name,
    }

    train_dataset = make_fiber_dataset(args.dataset_type, mode="train", **dataset_kwargs)
    val_dataset = make_fiber_dataset(args.dataset_type, mode="eval", **dataset_kwargs)

    model = model_selector(args.model, args)

    wandb_run = setup_wandb_run(args, model, with_test_loss=args.eval_config_path is not None)
    if wandb_run is None: return

    input_size = (args.batch_size, sum(args.input_flags), args.context_length, args.fibers_per_entry)
    print_model_summary(model, input_size)

    trainer = Trainer(model, train_dataset, val_dataset,
                          wandb_run=wandb_run, epochs=args.epochs,
                          batch_size=args.batch_size, config=args)

    trainer.train(save_dir=res_dir)

    if args.eval_config_path is not None:
        run_final_eval(model, args.eval_config_path, args, wandb_run, trainer.device)


if __name__ == "__main__":
    main()
