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

from data_utils import make_fiber_dataset, loader_context
from metrics import masked_mse, per_assay_mse, PearsonMeter, score_weights, MIN_COVERAGE
from regions import STRATA
from utils import *
from vis_utils import plot_evaluator_record_t

# Region types metrics are reported for, per base pair (see configs/README.md, Regions):
# all scored bases, bases in any cCRE, in each cCRE class, and outside cCREs
REPORT_STRATA = ["all", "ccre"] + STRATA + ["non_ccre"]


def _nan_to_none(x):
    return None if x is None or np.isnan(x) else float(x)


def _meter_summary(meter, output_assays):
    """Per-assay loss (MSE) and Pearson r from a PearsonMeter, plus their means over assays with data."""
    r, mse = meter.value.tolist(), meter.mse.tolist()
    per_assay = {a: {"loss": _nan_to_none(mse[k]), "pearson": _nan_to_none(r[k])} for k, a in enumerate(output_assays)}
    rs = [v["pearson"] for v in per_assay.values() if v["pearson"] is not None]
    ls = [v["loss"] for v in per_assay.values() if v["loss"] is not None]
    return {"n_bp": int(meter.n.max().item()),
            "pearson": float(np.mean(rs)) if rs else None,
            "loss": float(np.mean(ls)) if ls else None,
            "per_assay": per_assay}


def _print_strata_table(cell_type, strata, output_assays):
    print(f"[eval] {cell_type}: Pearson r per base pair, by region type")
    print(f"  {'region':9s} {'bases':>13s} " + " ".join(f"{a[:8]:>8s}" for a in output_assays) + f" {'mean':>7s}")
    for name, s in strata.items():
        vals = " ".join(f"{'-':>8s}" if s["per_assay"][a]["pearson"] is None else f"{s['per_assay'][a]['pearson']:8.3f}"
                        for a in output_assays)
        mean = f"{'-':>7s}" if s["pearson"] is None else f"{s['pearson']:7.3f}"
        print(f"  {name:9s} {s['n_bp']:>13,} {vals} {mean}")


class Evaluator:

    def __init__(self, model, dataset, batch_size=1, num_plots_to_log=5, device="cuda", seed=919, num_workers=0,
                 min_coverage=MIN_COVERAGE):
        self.model = model
        self.device = torch.device(device)
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_coverage = min_coverage  # positions covered by fewer fibers are not scored

        self.rng = random.Random(seed)
        self.num_to_save = num_plots_to_log

    def evaluate(self, save_path=None):
        self.model.eval()

        test_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            worker_init_fn=seed_worker,
            drop_last=False,
            num_workers=self.num_workers,  # the dataset strides its regions across workers
            multiprocessing_context=loader_context(self.num_workers),
        )

        composite_loss_meter = AverageMeter()
        output_assays = self.model.init_args.get("output_assays", ["atac"])
        assay_loss_meters = {a: AverageMeter() for a in output_assays}
        # Pooled bp-level r (and MSE) per region type; "all" covers the same positions as the MSE above
        strata_meters = {s: PearsonMeter(len(output_assays)) for s in REPORT_STRATA}
        locus_records = []
        valid_locus_count = 0
        num_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                fiber_features, target_bulk, forward_kwargs = unpack_batch(batch, self.device)
                assay_mask = batch["assay_mask"].to(self.device)
                coverage = batch["fiber_coverage"].to(self.device)

                with torch.amp.autocast(self.device_type, dtype=torch.float16):
                    pred_composite_bulk, processed_fibers = self.model(fiber_features, **forward_kwargs)

                # Scored entries only: assays the cell type has, at positions covered by >= min_coverage fibers
                weights = score_weights(assay_mask, coverage, self.min_coverage)    # (B, K, L)
                comp_loss = masked_mse(pred_composite_bulk, target_bulk, weights).item()
                composite_loss_meter.update(comp_loss)
                losses, counts = per_assay_mse(pred_composite_bulk, target_bulk, weights)
                for k, assay in enumerate(output_assays):
                    if counts[k] > 0:
                        assay_loss_meters[assay].update(losses[k].item(), n=int(counts[k].item()))
                num_samples += fiber_features.shape[0]

                strata_meters["all"].update(pred_composite_bulk, target_bulk, weights)
                if "strata" in batch:
                    st = batch["strata"].to(self.device)        # (B, len(STRATA), L) bool
                    in_ccre = st.any(dim=1)
                    selections = {"ccre": in_ccre, "non_ccre": ~in_ccre, **{c: st[:, i] for i, c in enumerate(STRATA)}}
                    for name, sel in selections.items():
                        strata_meters[name].update(pred_composite_bulk, target_bulk, weights * sel[:, None, :].float())

                ct = batch.get("cell_type", ["Unknown"])
                cell_type_name = ct[0] if isinstance(ct, (list, tuple)) else ct

                locus_record = {
                    "locus": batch["locus"],
                    "cell_type": cell_type_name,
                    "fiber_features": fiber_features.cpu(),
                    "processed_fibers": processed_fibers.cpu(),
                    "pred_bulk": pred_composite_bulk.cpu(),
                    "target_bulk": target_bulk.cpu(),
                    "assay_mask": assay_mask.cpu(),
                    "score_mask": (coverage >= self.min_coverage).cpu(),
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

        # None = the evaluated cell type has no bulk data for that assay (or r undefined: flat signal)
        strata = {s: _meter_summary(m, output_assays) for s, m in strata_meters.items() if m.n.max() > 0}
        all_bp = strata.get("all", {"pearson": None, "per_assay": {a: {"pearson": None} for a in output_assays}})
        per_assay = {a: {"loss": assay_loss_meters[a].avg if assay_loss_meters[a].count > 0 else None,
                         "pearson": all_bp["per_assay"][a]["pearson"]} for a in output_assays}
        return {
            # composite pearson = mean of the per-assay r over assays with data
            "composite": {"loss": composite_loss_meter.avg, "pearson": all_bp["pearson"]},
            "per_assay": per_assay,
            "strata": strata,
            "locus_records": locus_records,
            "num_locus": num_samples,
            "expected_samples": getattr(self.dataset, "expected_samples", None),
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

        windows = eval_cfg.get("windows")
        if not windows:
            raise ValueError(f"{eval_config_path}: eval configs need 'windows: <window set name>' "
                             "(e.g. test_chr21_mini2000); see configs/README.md, Regions.")
        eval_seed = eval_cfg.get("seed", 919)
        test_set = make_fiber_dataset(
            eval_cfg.get("dataset_type", "single"),
            mode="eval",
            metadata=eval_metadata,
            fibers_per_entry=eval_cfg.get("fibers_per_entry", getattr(train_args, "fibers_per_entry", 200)),
            context_length=eval_cfg.get("context_length", getattr(train_args, "context_length", 4096)),
            windows=windows,
            max_regions=eval_cfg.get("max_regions"),
            input_flags=model.init_args["input_flags"],
            dna_type=model.init_args["dna_type"],
            output_assays=output_assays,
            seed=eval_seed,
        )

        save_path = None
        if plots_output_dir is not None:
            save_path = os.path.join(plots_output_dir, cell_type) + "/"
            os.makedirs(save_path, exist_ok=True)

        evaluator = Evaluator(model, test_set, batch_size=1, num_plots_to_log=5, device=device, seed=eval_seed,
                              num_workers=getattr(train_args, "num_workers", 0),
                              min_coverage=getattr(train_args, "min_coverage", MIN_COVERAGE))
        eval_results = evaluator.evaluate(save_path=save_path)

        composite_loss = eval_results["composite"]["loss"]
        composite_r = eval_results["composite"]["pearson"]
        r_str = f"{composite_r:.4f}" if composite_r is not None else "n/a"
        n_done, n_expected = eval_results["num_locus"], eval_results["expected_samples"]
        print(f"[eval] {cell_type} on {windows}: {n_done} of {n_expected} samples evaluated "
              f"({n_expected - n_done} skipped: no fibers) | composite loss: {composite_loss:.6f} | pearson: {r_str}")
        _print_strata_table(cell_type, eval_results["strata"], output_assays)

        assay_avg = {a: eval_results["per_assay"][a]["loss"] for a in output_assays}
        assay_avg_r = {a: eval_results["per_assay"][a]["pearson"] for a in output_assays}
        summary[cell_type] = {
            "eval_config": eval_config_path,
            "windows": windows,
            "composite_loss": composite_loss,
            "pearson": composite_r,
            "per_assay_loss": assay_avg,
            "per_assay_pearson": assay_avg_r,
            "num_loci": n_done,
            "expected_samples": n_expected,
            "strata": eval_results["strata"],   # bp-level loss / r per region type, per assay
        }

        if wandb_run is not None:
            prefix = f"test/{cell_type}"
            ct_log_dict = {f"{prefix}/loss": composite_loss}
            if composite_r is not None:
                ct_log_dict[f"{prefix}/pearson"] = composite_r
            for assay, data in eval_results.get("per_assay", {}).items():
                if data["loss"] is not None:
                    ct_log_dict[f"{prefix}/loss_{assay}"] = data["loss"]
                if data["pearson"] is not None:
                    ct_log_dict[f"{prefix}/pearson_{assay}"] = data["pearson"]
            # Per region type: means over assays (per-assay detail is in the summary JSON)
            for name, s in eval_results["strata"].items():
                if name == "all":
                    continue
                if s["pearson"] is not None:
                    ct_log_dict[f"{prefix}/pearson@{name}"] = s["pearson"]
                if s["loss"] is not None:
                    ct_log_dict[f"{prefix}/loss@{name}"] = s["loss"]

            wandb_image_list = []
            for idx, record in enumerate(eval_results.get("locus_records", [])):
                fig = plot_evaluator_record_t(
                    record_t=record,
                    input_flags=test_set.input_flags,
                    assay_avg_losses=assay_avg,
                    output_assays=output_assays,
                    mode="Test",
                    assay_avg_pearson=assay_avg_r,
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
