"""
Main file
"""

import os
import torch
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

    # Continue an unfinished run (its folder, wandb run and optimizer state), or start a new one
    resume_dir = None
    if args.resume:
        resume_dir = find_resumable_run(args.res_dir, args) if args.resume == "latest" else args.resume
        if resume_dir is None:
            print(f"[resume] No unfinished run of {get_config_names_str(args)} under {args.res_dir}: nothing to do.")
            return
    res_dir = resume_dir or os.path.join(args.res_dir, create_save_str(args))

    dataset_kwargs = {
        "metadata": args.metadata,
        "fibers_per_entry": args.fibers_per_entry,
        "context_length": args.context_length,
        "input_flags": args.input_flags,
        "seed": args.seed,
        "dna_type": args.dna_type,
        "output_assays": args.output_assays,
    }
    if args.dataset_type == "mixed":
        # Validation mixes too (proportions seeded per region): model selection never sees a cell type's own track
        dataset_kwargs["mix_range"] = args.mix_range

    # Training regions are drawn from the pools at runtime; validation iterates a fixed window set
    train_dataset = make_fiber_dataset(args.dataset_type, mode="train", samples_per_epoch=args.samples_per_epoch,
                                       train_windows=args.train_windows, **dataset_kwargs)
    val_dataset = make_fiber_dataset(args.dataset_type, mode="eval", windows=args.val_windows, **dataset_kwargs)
    print(f"Training regions: {args.train_windows} on {args.metadata['train_chrs']} | "
          f"validation: {args.val_windows} ({val_dataset.expected_samples} samples per epoch)", flush=True)

    model = model_selector(args.model, args)
    ckpt = None
    if resume_dir:
        ckpt = torch.load(os.path.join(resume_dir, LAST_CHECKPOINT), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[resume] {resume_dir}: continuing after epoch {ckpt['epoch']} of {args.epochs}", flush=True)

    wandb_run = setup_wandb_run(args, model, with_test_loss=args.eval_configs is not None,
                                run_id=ckpt.get("wandb_id") if ckpt else None)
    if wandb_run is None: return

    input_size = (args.batch_size, sum(args.input_flags), args.context_length, args.fibers_per_entry)
    print_model_summary(model, input_size)

    trainer = Trainer(model, train_dataset, val_dataset,
                          wandb_run=wandb_run, epochs=args.epochs,
                          batch_size=args.batch_size, lr=args.lr, patience=args.patience, config=args,
                          deadline=training_deadline(args.max_hours))
    if ckpt:
        trainer.load_checkpoint(ckpt)

    if not trainer.train(save_dir=res_dir):
        print(f"[checkpoint] Stopped early to fit the time limit; checkpoint in {res_dir}/{LAST_CHECKPOINT}. "
              f"Continue with: --resume latest (or --resume {res_dir})", flush=True)
        wandb_run.finish()
        return

    if args.eval_configs is not None:
        run_final_eval(model, args.eval_configs, args, wandb_run, trainer.device, output_dir=res_dir)


if __name__ == "__main__":
    main()
