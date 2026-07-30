"""
Standalone CLI entrypoint for evaluating saved models on mixed cell-type fiber data.
"""

import os
import json
import argparse
import yaml
import torch

from models import BaseModel, model_selector
from eval_dataset import MixedCellFiberDataset
from evaluator import Evaluator
from utils import set_seed, print_model_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pre-trained Fiber-seq models on mixed cell-type datasets.")

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the saved PyTorch model checkpoint bundle (.pt)."
    )
    parser.add_argument(
        "--eval_config", type=str, required=True,
        help="Path to the evaluation YAML config file (e.g., configs/eval_mixed_config.yaml)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on ('cuda' or 'cpu')."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./eval_results",
        help="Directory where evaluation metrics and logs will be saved."
    )
    parser.add_argument(
        "--seed", type=int, default=919,
        help="Random seed for evaluation reproducible sampling."
    )
    return parser.parse_args()


def load_yaml_config(config_path):
    """Loads and returns a dictionary from a YAML config file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Evaluation YAML config file not found at: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    set_seed(args.seed)

    print("\n" + "=" * 60)
    print(" STANDALONE MODEL EVALUATION ")
    print("=" * 60)
    print(f" Checkpoint Path : {args.checkpoint}")
    print(f" Eval Config     : {args.eval_config}")
    print(f" Target Device   : {args.device}")
    print("=" * 60 + "\n")

    # 1. Load Model Checkpoint Bundle
    print("--> Loading model checkpoint...")
    try:
        model, checkpoint_config = BaseModel.load_model(args.checkpoint, map_location=args.device)
    except Exception as e:
        print(f"Error loading model using BaseModel.load_model: {e}")
        print("Falling back to raw state dict loading via model_selector...")

        # Fallback to config reading if not using bundled BaseModel architecture
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        config = checkpoint.get("model_config", {})

        # Build dummy args object
        class DummyArgs:
            pass
        d_args = DummyArgs()
        d_args.num_input_features = config.get("num_input_features", 5)
        d_args.decoder_type = config.get("decoder_type", "avg_n")
        d_args.kernel_size = config.get("kernel_size", 15)

        model_type = config.get("model_type", "deep01")
        model = model_selector(model_type, d_args)
        model.load_state_dict(state_dict)

    model.to(args.device)
    model.eval()

    # 2. Load Evaluation Configuration
    print("--> Parsing evaluation YAML config...")
    eval_cfg = load_yaml_config(args.eval_config)

    # 3. Instantiate MixedCellFiberDataset & DataLoader
    print("--> Building MixedCellFiberDataset...")
    dataset_kwargs = {
        "metadata": eval_cfg["metadata"],
        "fibers_per_entry": eval_cfg.get("fibers_per_entry", 200),
        "context_length": eval_cfg.get("context_length", 4096),
        "iters_per_epoch": eval_cfg.get("iters_per_epoch", 50),
        "input_flags": eval_cfg.get("input_flags", [1, 1, 1, 1, 1]),
        "cell_ratios": eval_cfg.get("cell_ratios", None),
        "mode": eval_cfg.get("mode", "val"),
        "seed": args.seed,
        "return_dna": eval_cfg.get("return_dna", False)
    }

    eval_dataset = MixedCellFiberDataset(**dataset_kwargs)

    # DataLoader wrapper around IterableDataset
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=None,  # Handled inside iterator return
        num_workers=0     # Set to 0 for linear execution/debugging
    )

    # 4. Initialize Evaluator & Execute Loop
    print("\n--> Running evaluation loop...")
    evaluator = Evaluator(model, device=args.device)
    results = evaluator.evaluate(eval_loader)

    # 5. Display Console Metrics
    print("\n" + "=" * 60)
    print(" EVALUATION RESULTS SUMMARY ")
    print("=" * 60)
    print(f" Composite MSE Loss  : {results['composite']['loss']:.6f}")
    print(f" Composite Pearson R : {results['composite']['pearson_r']:.4f}")
    print("-" * 60)
    print(" Per-Cell-Type Deconvolution Breakdown:")
    for ct, metrics in results["per_cell_type"].items():
        print(f"   * [{ct:10s}] MSE Loss: {metrics['loss']:.6f} | Pearson R: {metrics['pearson_r']:.4f}")
    print("=" * 60 + "\n")

    # 6. Save JSON Results Summary
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_filename = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_json_path = os.path.join(args.output_dir, f"{ckpt_filename}_eval_summary.json")

    # Prepare JSON serializable summary
    json_summary = {
        "checkpoint": args.checkpoint,
        "eval_config": args.eval_config,
        "composite_metrics": results["composite"],
        "per_cell_type_metrics": results["per_cell_type"],
        "num_loci_evaluated": len(results["locus_records"])
    }

    with open(out_json_path, "w") as f:
        json.dump(json_summary, f, indent=4)

    print(f"Full evaluation summary saved to: {out_json_path}")


if __name__ == "__main__":
    main()
