"""
Standalone CLI for evaluating saved fiber-seq models against one or more eval configs.

Usage:
    python eval.py --checkpoint results/.../Model_epoch_N.pt \
                   --eval_configs configs/evals/eval12_gm_atac.yaml [more...] \
                   [--metapaths configs/metapaths.yaml] \
                   [--device cpu|cuda] \
                   [--output_dir ./eval_results] \
                   [--save_plots] \
                   [--wandb]
"""

import os
import json
import argparse
from types import SimpleNamespace

import torch
import wandb

from models import BaseModel
from evaluator import run_final_eval
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pre-trained Fiber-seq models.")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved model checkpoint (.pt).")
    parser.add_argument("--eval_configs", type=str, nargs="+", required=True,
                        help="One or more eval YAML config paths, one per cell type.")
    parser.add_argument("--metapaths", type=str, default="configs/metapaths.yaml",
                        help="Path to metapaths YAML for path resolution.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_plots", action="store_true",
                        help="Save per-locus dashboard plots to output_dir/{cell_type}/.")
    parser.add_argument("--wandb", action="store_true",
                        help="Log results to a new Weights & Biases run.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print(" STANDALONE MODEL EVALUATION ")
    print("=" * 60)
    print(f" Checkpoint   : {args.checkpoint}")
    print(f" Eval configs : {args.eval_configs}")
    print(f" Device       : {args.device}")
    print(f" WandB        : {args.wandb}")
    print("=" * 60 + "\n")

    print("--> Loading model checkpoint...")
    model, _ = BaseModel.load_model(args.checkpoint, map_location=args.device)
    model.to(args.device)
    model.eval()

    run_name = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint))) + "_eval"

    # Minimal namespace so run_final_eval has fallback values for context_length / fibers_per_entry
    train_args_ns = SimpleNamespace(
        metapaths=args.metapaths,
        fibers_per_entry=200,
        context_length=4096,
    )

    wandb_run = None
    if args.wandb:
        wandb_run = wandb.init(
            project="fiber",
            entity="liblab",
            job_type="eval",
            name=run_name,
            config={"checkpoint": args.checkpoint, "eval_configs": args.eval_configs},
        )

    output_dir = args.output_dir if args.output_dir is not None else run_name
    plots_dir = output_dir if args.save_plots else None
    summary = run_final_eval(
        model, args.eval_configs, train_args_ns, wandb_run, args.device,
        plots_output_dir=plots_dir,
    )

    if wandb_run is not None:
        wandb_run.finish()
        print("WandB run finished.")

    # Save JSON summary
    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "eval_summary.json")
    with open(out_json, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "results": summary}, f, indent=4)
    print(f"Summary saved to: {out_json}")


if __name__ == "__main__":
    main()
