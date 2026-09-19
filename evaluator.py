"""
Evaluator module for testing fiber data models.
Takes a model and a dataset, runs inference, and computes MSE metrics.
Supports both single-cell and mixed-cell batch types (auto-detected from batch keys).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from data_utils import make_fiber_dataset
from metrics import mse_loss
from utils import *
from vis_utils import plot_evaluator_record_t


class Evaluator:

    def __init__(self, model, dataset, batch_size=1, num_plots_to_log=5, device="cuda", seed=919):
        self.model = model
        self.device = torch.device(device)
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()

        self.rng = random.Random(seed)
        self.num_to_save = num_plots_to_log

    def evaluate(self, save_path=None):
        self.model.eval()

        test_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            worker_init_fn=seed_worker,
            drop_last=False,
        )

        composite_loss_meter = AverageMeter()
        output_assays = self.model.init_args.get("output_assays", ["atac"])
        assay_loss_meters = {a: AverageMeter() for a in output_assays}
        locus_records = []
        valid_locus_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                fiber_features, target_bulk, forward_kwargs = unpack_batch(batch, self.device)

                with torch.amp.autocast(self.device_type, dtype=torch.float16):
                    pred_composite_bulk, processed_fibers = self.model(fiber_features, **forward_kwargs)

                comp_loss = self.criterion(pred_composite_bulk, target_bulk).item()
                composite_loss_meter.update(comp_loss)
                for k, assay in enumerate(output_assays):
                    assay_loss_meters[assay].update(
                        self.criterion(pred_composite_bulk[:, k], target_bulk[:, k]).item()
                    )

                ct = batch.get("cell_type", ["Unknown"])
                cell_type_name = ct[0] if isinstance(ct, (list, tuple)) else ct

                locus_record = {
                    "locus": batch["locus"],
                    "cell_type": cell_type_name,
                    "fiber_features": fiber_features.cpu(),
                    "processed_fibers": processed_fibers.cpu(),
                    "pred_bulk": pred_composite_bulk.cpu(),
                    "target_bulk": target_bulk.cpu(),
                    "loss": comp_loss,
                }
                valid_locus_count += 1

                if len(locus_records) < self.num_to_save:
                    locus_records.append(locus_record)
                else:
                    j = self.rng.randint(0, valid_locus_count - 1)
                    if j < self.num_to_save:
                        locus_records[j] = locus_record

                if save_path is not None:
                    print(f"saving idx{batch_idx}")
                    fig = plot_evaluator_record_t(
                        record_t=locus_record,
                        input_flags=self.model.init_args["input_flags"],
                        assay_avg_losses=None,
                        output_assays=output_assays,
                        mode="Test"
                    )
                    plt.savefig(f"{save_path}test_e_{batch_idx}.png")
                    plt.close()

        return {
            "composite": {"loss": composite_loss_meter.avg},
            "per_assay": {a: {"loss": assay_loss_meters[a].avg} for a in assay_loss_meters},
            "locus_records": locus_records,
            "num_locus": valid_locus_count * self.batch_size,
        }


def run_final_eval(model, eval_configs, train_args, wandb_run, device, plots_output_dir=None):
    """
    Run evaluation over a list of eval config paths.
    Logs to wandb_run immediately after each config (if wandb_run is not None).
    Returns a summary dict keyed by cell type.
    """
    print("\n" + "=" * 60)
    print(f" Running Final Evaluation ({len(eval_configs)} config(s))...")
    print("=" * 60)

    metapaths = None
    if hasattr(train_args, "metapaths") and os.path.exists(train_args.metapaths):
        metapaths = load_metapaths(train_args.metapaths)

    output_assays = model.init_args.get("output_assays", ["atac"])
    summary = {}

    for eval_config_path in eval_configs:
        eval_cfg = load_config_file(eval_config_path)

        cell_types_dict = eval_cfg.get("cell_types", {})
        cell_type = next(iter(cell_types_dict)) if cell_types_dict else "unknown"

        print(f"\n[eval] Cell type: {cell_type} | Config: {eval_config_path}")

        if metapaths is not None:
            eval_metadata = resolve_config_with_metapaths(eval_cfg, metapaths, output_assays)
        else:
            eval_metadata = eval_cfg.get("metadata", {})

        eval_seed = eval_cfg.get("seed", 919)
        num_sample_ccres = eval_cfg.get("num_sample_ccres", 100)
        test_set = make_fiber_dataset(
            eval_cfg.get("dataset_type", "single"),
            mode="eval",
            metadata=eval_metadata,
            fibers_per_entry=eval_cfg.get("fibers_per_entry", getattr(train_args, "fibers_per_entry", 200)),
            context_length=eval_cfg.get("context_length", getattr(train_args, "context_length", 4096)),
            iters_per_epoch=num_sample_ccres,
            num_sample_ccres=num_sample_ccres,
            input_flags=model.init_args["input_flags"],
            dna_type=model.init_args["dna_type"],
            output_assays=output_assays,
            seed=eval_seed,
        )

        save_path = None
        if plots_output_dir is not None:
            save_path = os.path.join(plots_output_dir, cell_type) + "/"
            os.makedirs(save_path, exist_ok=True)

        evaluator = Evaluator(model, test_set, batch_size=1, num_plots_to_log=5, device=device, seed=eval_seed)
        eval_results = evaluator.evaluate(save_path=save_path)

        composite_loss = eval_results["composite"]["loss"]
        print(f"[eval] {cell_type} composite loss: {composite_loss:.6f}")

        summary[cell_type] = {
            "eval_config": eval_config_path,
            "composite_loss": composite_loss,
            "per_assay_loss": {a: eval_results["per_assay"][a]["loss"] for a in output_assays},
            "num_loci": eval_results["num_locus"],
        }

        if wandb_run is not None:
            prefix = f"test/{cell_type}"
            ct_log_dict = {f"{prefix}/loss": composite_loss}
            for assay, data in eval_results.get("per_assay", {}).items():
                ct_log_dict[f"{prefix}/loss_{assay}"] = data["loss"]

            assay_avg = {a: eval_results["per_assay"][a]["loss"] for a in output_assays}
            wandb_image_list = []
            for idx, record in enumerate(eval_results.get("locus_records", [])):
                fig = plot_evaluator_record_t(
                    record_t=record,
                    input_flags=test_set.input_flags,
                    assay_avg_losses=assay_avg,
                    output_assays=output_assays,
                    mode="Test"
                )
                chr_name = record["locus"][0][0]
                start = record["locus"][1][0]
                end = record["locus"][2][0]
                caption = f"[{cell_type}] Locus {idx}/{eval_results['num_locus']}: {chr_name}:{start}-{end}"
                wandb_image_list.append(wandb.Image(fig, caption=caption))
                plt.close(fig)

            ct_log_dict[f"Evaluation/{cell_type}_Dashboards"] = wandb_image_list
            wandb_run.log(ct_log_dict)
            print(f"[eval] Logged {len(wandb_image_list)} dashboards for {cell_type} to WandB.")

    print(f"\n Evaluation complete for {len(eval_configs)} cell type(s).")
    return summary
