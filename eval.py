"""
Standalone CLI for evaluating saved fiber-seq models against one or more eval configs.

Usage:
    python eval.py --checkpoint results/.../Model_epoch_N.pt \
                   --eval_configs configs/evals/eval21_GM12878.yaml [more...] \
                   [--baseline] \
                   [--metapaths configs/metapaths.yaml] \
                   [--device cpu|cuda] \
                   [--output_dir ./results] \
                   [--save_plots] \
                   [--wandb]

    # Cross-cell-average baseline only (no checkpoint, no GPU needed): mean of the training cells' tracks
    python eval.py --baseline_only --train_data_config configs/data/data20_dev5_200U.yaml \
                   --eval_configs configs/evals/eval21_GM12878.yaml [more...]

--baseline adds the baseline (training cells read from the checkpoint) next to the model, plus the cell-specific
correlation (partial correlation of prediction and target given the baseline). With --mixture_reference the
baseline includes the evaluated cell's own line: for a model trained on mixtures of its training cells (T3) it is
the mixture track, i.e. what predicting the mixture bulk on every molecule would score. The baseline reads every training cell's tracks for
each sample, so it is slow; it depends only on the data, so run it once per training set with --baseline_only.
"""

import os
import re
import json
import argparse
import datetime
from types import SimpleNamespace

import torch
import wandb
import yaml

from models import BaseModel
from evaluator import run_final_eval
from utils import set_seed, load_config_file

ENCODE_SPEC = "data_prep/encode_spec.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pre-trained Fiber-seq models.")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to saved model checkpoint (.pt). Not used with --baseline_only.")
    parser.add_argument("--eval_configs", type=str, nargs="+", required=True,
                        help="One or more eval YAML config paths, one per cell type.")
    parser.add_argument("--baseline", action="store_true",
                        help="Also score the cross-cell-average baseline of the checkpoint's training cells.")
    parser.add_argument("--mixture_reference", action="store_true",
                        help="Baseline = mean of all training cells including the evaluated cell's own line "
                             "(the mixture reference of T3) instead of the cross-cell average without it.")
    parser.add_argument("--baseline_only", action="store_true",
                        help="Score only the cross-cell-average baseline (no checkpoint); needs --train_data_config.")
    parser.add_argument("--train_data_config", type=str, default=None,
                        help="With --baseline_only: data config whose cell_types are the training cells to average.")
    parser.add_argument("--output_assays", type=str, nargs="+", default=None,
                        help=f"With --baseline_only: assays to score (default: all assays in {ENCODE_SPEC}).")
    parser.add_argument("--metapaths", type=str, default="configs/metapaths.yaml",
                        help="Path to metapaths YAML for path resolution.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--save_plots", action="store_true",
                        help="Save per-locus dashboard plots to output_dir/{runname}/{cell_type}/.")
    parser.add_argument("--wandb", action="store_true",
                        help="Log results to a new Weights & Biases run.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers for loading eval regions (0 = load in the main process).")
    parser.add_argument("--min_coverage", type=int, default=5,
                        help="Metrics only use positions covered by at least this many fibers.")
    args = parser.parse_args()
    if args.baseline_only and not args.train_data_config:
        parser.error("--baseline_only needs --train_data_config (whose cell_types are the training cells).")
    if not args.baseline_only and not args.checkpoint:
        parser.error("--checkpoint is required (unless --baseline_only).")
    return args


def main():
    args = parse_args()
    now = datetime.datetime.now().strftime("%y-%m-%d_T%H-%M-%S")

    print("\n" + "=" * 60)
    print(" STANDALONE " + ("BASELINE" if args.baseline_only else "MODEL") + " EVALUATION ")
    print("=" * 60)
    print(f" Checkpoint   : {args.checkpoint if not args.baseline_only else '- (cross-cell-average baseline only)'}")
    print(f" Eval configs : {args.eval_configs}")
    print(f" Device       : {args.device}")
    print(f" WandB        : {args.wandb}")
    print("=" * 60 + "\n")

    model, output_assays, baseline_cells = None, None, None
    if args.baseline_only:
        baseline_cells = list(load_config_file(args.train_data_config)["cell_types"])
        output_assays = args.output_assays or list(yaml.safe_load(open(ENCODE_SPEC))["assays"])
        base_name = f"BASELINE_{os.path.splitext(os.path.basename(args.train_data_config))[0]}"
        print(f"--> Baseline: training cells {baseline_cells} from {args.train_data_config}, assays {output_assays}")
        run_name, wandb_run_name = f"{now}_{base_name}", base_name
    else:
        print("--> Loading model checkpoint...")
        model, ckpt_config = BaseModel.load_model(args.checkpoint, map_location=args.device)
        model.to(args.device)
        model.eval()
        if args.baseline:
            try:
                baseline_cells = list(ckpt_config["metadata"]["cell_types"])
            except (KeyError, TypeError):
                raise ValueError(f"{args.checkpoint} does not record its training cells (config.metadata.cell_types); "
                                 "use --baseline_only --train_data_config <its data config> instead.")
            print(f"--> Baseline: training cells {baseline_cells} (from the checkpoint)")
        run_dir = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
        base_name = re.sub(r'^\d{2}-\d{2}-\d{2}_T\d{2}-\d{2}-\d{2}_?', '', run_dir)
        run_name, wandb_run_name = f"{now}_{base_name}_eval", f"{base_name}_eval"

    # Minimal namespace so run_final_eval has fallback values for context_length / fibers_per_entry
    train_args_ns = SimpleNamespace(
        metapaths=args.metapaths,
        fibers_per_entry=200,
        context_length=4096,
        num_workers=args.num_workers,
        min_coverage=args.min_coverage,
    )

    wandb_run = None
    if args.wandb:
        wandb_run = wandb.init(
            project="fiber",
            entity="liblab",
            job_type="eval",
            name=wandb_run_name,
            config={"checkpoint": args.checkpoint, "eval_configs": args.eval_configs, "baseline_cells": baseline_cells},
        )

    output_dir = os.path.join(args.output_dir, run_name)
    plots_dir = output_dir if args.save_plots else None
    summary = run_final_eval(
        model, args.eval_configs, train_args_ns, wandb_run, args.device,
        plots_output_dir=plots_dir, baseline_cells=baseline_cells, output_assays=output_assays,
        output_dir=output_dir, baseline_include_own=args.mixture_reference,
    )

    if wandb_run is not None:
        wandb_run.finish()
        print("WandB run finished.")

    # Save JSON summary
    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "eval_summary.json")
    with open(out_json, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "baseline_train_cells": baseline_cells,
                   "baseline_include_own": args.mixture_reference,
                   "train_data_config": args.train_data_config, "results": summary}, f, indent=4)
    print(f"Summary saved to: {out_json}")


if __name__ == "__main__":
    main()
