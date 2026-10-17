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
        "input_flags": args.input_flags,
        "seed": args.seed,
        "dna_type": args.dna_type,
        "output_assays": args.output_assays,
    }

    # Training regions are drawn from the pools at runtime; validation iterates a fixed window set
    train_dataset = make_fiber_dataset(args.dataset_type, mode="train", samples_per_epoch=args.samples_per_epoch,
                                       train_windows=args.train_windows, **dataset_kwargs)
    val_dataset = make_fiber_dataset(args.dataset_type, mode="eval", windows=args.val_windows, **dataset_kwargs)
    print(f"Training regions: {args.train_windows} on {args.metadata['train_chrs']} | "
          f"validation: {args.val_windows} ({val_dataset.expected_samples} samples per epoch)", flush=True)

    model = model_selector(args.model, args)

    wandb_run = setup_wandb_run(args, model, with_test_loss=args.eval_configs is not None)
    if wandb_run is None: return

    input_size = (args.batch_size, sum(args.input_flags), args.context_length, args.fibers_per_entry)
    print_model_summary(model, input_size)

    trainer = Trainer(model, train_dataset, val_dataset,
                          wandb_run=wandb_run, epochs=args.epochs,
                          batch_size=args.batch_size, config=args)

    trainer.train(save_dir=res_dir)

    if args.eval_configs is not None:
        run_final_eval(model, args.eval_configs, args, wandb_run, trainer.device)


if __name__ == "__main__":
    main()
