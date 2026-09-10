"""
Standalone CLI for evaluating saved fiber-seq models.

Usage:
    python eval.py --checkpoint results/.../Model_epoch_N.pt \
                   --eval_config configs/evals/eval00.yaml \
                   [--metapaths configs/metapaths.yaml] \
                   [--device cpu|cuda] \
                   [--output_dir ./eval_results] \
                   [--save_plots]
"""

import os
import json
import argparse
import torch

from models import BaseModel
from data_utils import make_fiber_dataset
from evaluator import Evaluator
from utils import set_seed, load_config_file, load_metapaths, resolve_config_with_metapaths


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pre-trained Fiber-seq models.")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved model checkpoint (.pt).")
    parser.add_argument("--eval_config", type=str, required=True,
                        help="Path to evaluation YAML config.")
    parser.add_argument("--metapaths", type=str, default="configs/metapaths.yaml",
                        help="Path to metapaths YAML for path resolution.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--save_plots", action="store_true",
                        help="Save per-locus dashboard plots to output_dir.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print(" STANDALONE MODEL EVALUATION ")
    print("=" * 60)
    print(f" Checkpoint  : {args.checkpoint}")
    print(f" Eval Config : {args.eval_config}")
    print(f" Device      : {args.device}")
    print("=" * 60 + "\n")

    # Load model; input_flags and dna_type come from model checkpoint, not config
    print("--> Loading model checkpoint...")
    model, _ = BaseModel.load_model(args.checkpoint, map_location=args.device)
    model.to(args.device)
    model.eval()

    input_flags = model.init_args["input_flags"]
    dna_type = model.init_args["dna_type"]

    # Load and resolve eval config
    print("--> Loading eval config...")
    eval_cfg = load_config_file(args.eval_config)
    metapaths = load_metapaths(args.metapaths) if os.path.exists(args.metapaths) else None

    if metapaths and "assay" in eval_cfg:
        metadata = resolve_config_with_metapaths(eval_cfg, metapaths)
    else:
        metadata = eval_cfg.get("metadata", {})

    dataset_type = eval_cfg.get("dataset_type", "mixed")
    seed = eval_cfg.get("seed", 919)
    num_sample_ccres = eval_cfg.get("num_sample_ccres", 100)
    set_seed(seed)

    # Build dataset in eval mode (fully deterministic)
    print("--> Building dataset...")
    dataset = make_fiber_dataset(
        dataset_type,
        mode="eval",
        metadata=metadata,
        fibers_per_entry=eval_cfg.get("fibers_per_entry", 200),
        context_length=eval_cfg.get("context_length", 4096),
        num_sample_ccres=num_sample_ccres,
        iters_per_epoch=num_sample_ccres,
        input_flags=input_flags,
        dna_type=dna_type,
        bulk_name=eval_cfg.get("bulk_name", "N/A"),
        seed=seed,
    )

    # Run evaluation
    print("--> Running evaluation...")
    save_path = None
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = args.output_dir + "/"

    evaluator = Evaluator(model, dataset, batch_size=1, num_plots_to_log=5,
                          device=args.device, seed=seed)
    results = evaluator.evaluate(save_path=save_path)

    # Print results
    print("\n" + "=" * 60)
    print(" EVALUATION RESULTS ")
    print("=" * 60)
    print(f" Composite MSE Loss : {results['composite']['loss']:.6f}")
    if results["per_cell_type"]:
        print("-" * 60)
        print(" Per-Cell-Type Breakdown:")
        for ct, metrics in results["per_cell_type"].items():
            print(f"   * [{ct:10s}] MSE Loss: {metrics['loss']:.6f}")
    print(f" Loci evaluated     : {results['num_locus']}")
    print("=" * 60 + "\n")

    # Save JSON summary
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_json = os.path.join(args.output_dir, f"{ckpt_name}_eval_summary.json")

    json_summary = {
        "checkpoint": args.checkpoint,
        "eval_config": args.eval_config,
        "composite_metrics": results["composite"],
        "per_cell_type_metrics": results["per_cell_type"],
        "num_loci_evaluated": results["num_locus"],
    }

    with open(out_json, "w") as f:
        json.dump(json_summary, f, indent=4)

    print(f"Results saved to: {out_json}")


if __name__ == "__main__":
    main()
