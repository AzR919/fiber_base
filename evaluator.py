"""
Evaluator module for testing and deconvoluting mixed cell-type fiber data.
Evaluates model performance across mixed composite signals as well as
individual cell-type reconstructions.
"""

import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eval_dataset import MixedCellFiberDataset
from utils import *

#--------------------------------------------------------------------------------------------------
# data setup

def test_dataset_from_path_and_extra_args(eval_config_path, overwrite_args):

    eval_config = load_config_file(eval_config_path)
    kwargs = {
            "metadata": eval_config["metadata"],
            "context_length": eval_config["context_length"],
            "fibers_per_entry": eval_config["fibers_per_entry"],
            "num_sample_ccres": eval_config["num_sample_ccres"],
            "bulk_name": eval_config["bulk_name"],
            "seed": eval_config["seed"],
            "input_flags": [1, 1, 1, 1, 1],
            "return_dna": False
        }

    kwargs.update(overwrite_args)

    return MixedCellFiberDataset(**kwargs)


class Evaluator:
    """
    Evaluation runner that executes forward passes on mixed-cell datasets,
    deconvolutes single-cell fiber predictions per cell type, and computes
    MSE loss and Pearson correlation metrics.
    """

    def __init__(self, model, test_set, batch_size, device="cuda", criterion=None):
        """
        Args:
            model (nn.Module): Pre-trained PyTorch model instance.
            device (str or torch.device): Hardware device ('cuda' or 'cpu').
            criterion (nn.Module, optional): Loss function (defaults to MSELoss).
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.test_set = test_set
        self.batch_size = batch_size
        self.criterion = criterion if criterion is not None else nn.MSELoss()

    @staticmethod
    def _compute_pearson_r(pred, target):
        """
        Computes Pearson correlation coefficient between two 1D or 2D tensors.

        Args:
            pred (torch.Tensor or np.ndarray): Predicted signal profile.
            target (torch.Tensor or np.ndarray): Ground truth target profile.

        Returns:
            float: Pearson correlation coefficient.
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        pred_flat = pred.flatten()
        target_flat = target.flatten()

        pred_std = np.std(pred_flat)
        target_std = np.std(target_flat)

        if pred_std == 0 or target_std == 0:
            return 0.0

        cov = np.cov(pred_flat, target_flat)[0, 1]
        return float(cov / (pred_std * target_std))

    def _deconvolve_cell_type_bulk(self, processed_fibers, ct_mask, decoder_type):
        """
        Applies decoder aggregation over a masked subset of single-cell fibers.

        Args:
            processed_fibers (torch.Tensor): Processed single-cell fibers [B, L, N].
            ct_mask (torch.Tensor): Boolean mask tensor [N] indicating fibers for this cell type.
            decoder_type (str): Aggregation method ('avg', 'avg_n', 'sum').

        Returns:
            torch.Tensor: Reconstructed cell-type bulk profile [B, L].
        """
        # Slice fibers belonging to this cell type: [B, L, N_ct]
        ct_fibers = processed_fibers[:, :, ct_mask]
        n_ct_fibers = ct_fibers.shape[-1]

        if n_ct_fibers == 0:
            return torch.zeros((processed_fibers.shape[0], processed_fibers.shape[1]), device=self.device)

        if decoder_type == "sum":
            return torch.sum(ct_fibers, dim=-1)
        elif decoder_type in ["avg", "avg_n"]:
            return torch.mean(ct_fibers, dim=-1)
        else:
            raise NotImplementedError(f"Decoder type '{decoder_type}' not supported for deconvolution.")

    def evaluate(self):
        """
        Runs evaluation loop over the provided mixed-cell dataloader.

        Returns:
            dict: Comprehensive evaluation results including composite loss/pearson,
                  per-cell-type losses/pearsons, and plotting data payload.
        """
        self.model.eval()

        test_loader = DataLoader(
                                self.test_set,
                                batch_size=self.batch_size,
                                worker_init_fn=seed_worker,
                            )

        # Metrics trackers
        composite_loss_meter = AverageMeter()
        cell_type_loss_meters = {}

        # Store locus-level outputs for plotting/inspection
        locus_records = []

        decoder_type = getattr(self.model, "decoder_type", "avg_n")

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Prepare model inputs
                inputs = batch["fiber_features"].to(self.device)  # [B, C, L, N]
                target_bulk = batch["target_bulk"].to(self.device)  # [B, L]
                n_fibers = batch["n_fibers"].to(self.device)  # [B]

                # Model Forward Pass
                pred_composite_bulk, processed_fibers = self.model(inputs, n_fibers=n_fibers)

                # Composite evaluation
                comp_loss = self.criterion(pred_composite_bulk, target_bulk).item()
                composite_loss_meter.update(comp_loss)

                # Cell-type specific deconvolution and evaluation
                ct_targets = batch["cell_type_targets"]
                ct_masks = batch["cell_type_masks"]

                cell_type_preds = {}
                cell_type_losses = {}

                for ct_name, mask_tensor in ct_masks.items():
                    # Initialize meters if seeing cell type for the first time
                    if ct_name not in cell_type_loss_meters:
                        cell_type_loss_meters[ct_name] = AverageMeter()

                    # Squeeze batch dimension for mask
                    ct_mask = mask_tensor[0] if mask_tensor.dim() > 1 else mask_tensor
                    ct_mask = ct_mask.to(self.device)

                    # Reconstruct single cell-type bulk profile from masked fibers
                    pred_ct_bulk = self._deconvolve_cell_type_bulk(
                        processed_fibers, ct_mask, decoder_type
                    )
                    target_ct_bulk = ct_targets[ct_name].to(self.device)

                    # Calculate cell-type specific metrics
                    ct_loss = self.criterion(pred_ct_bulk, target_ct_bulk).item()
                    cell_type_loss_meters[ct_name].update(ct_loss)

                    cell_type_preds[ct_name] = pred_ct_bulk.cpu()
                    cell_type_losses[ct_name] = ct_loss

                # Package payload for downstream plotting modules
                locus_records.append({
                    "locus": batch["locus"],
                    "fiber_features": inputs.cpu(),
                    "processed_fibers": processed_fibers.cpu(),
                    "pred_bulk": pred_composite_bulk.cpu(),
                    "target_bulk": target_bulk.cpu(),
                    "pred_cell_type_bulks": cell_type_preds,
                    "target_cell_type_bulks": {k: v.cpu() for k, v in ct_targets.items()},
                    "cell_type_masks": ct_masks,
                    "loss": comp_loss,
                    "cell_type_losses": cell_type_losses
                })

        # Compile final metric dictionary
        metrics_summary = {
            "composite": {
                "loss": composite_loss_meter.avg,
            },
            "per_cell_type": {
                ct: {
                    "loss": cell_type_loss_meters[ct].avg,
                }
                for ct in cell_type_loss_meters
            },
            "locus_records": locus_records
        }

        return metrics_summary


#--------------------------------------------------------------------------------------------------
# Verification Test

def tester():
    from models import Deep01ResConv1dBlock

    print("Initializing Evaluator test with dummy inputs...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Deep01ResConv1dBlock(num_input_features=5, decoder_type="avg_n", kernel_size=15)

    evaluator = Evaluator(model, device=device)

    # Synthesize dummy batch matching MixedCellFiberDataset payload
    B, C, L, N = 1, 5, 2048, 200
    dummy_inputs = torch.rand((B, C, L, N))
    dummy_composite_target = torch.rand((B, L))

    # Mask: 100 fibers for GM12878, 100 fibers for K562
    gm_mask = torch.zeros(N, dtype=torch.bool)
    gm_mask[:100] = True
    k562_mask = torch.zeros(N, dtype=torch.bool)
    k562_mask[100:] = True

    dummy_batch = {
        "fiber_features": dummy_inputs,
        "target_bulk": dummy_composite_target,
        "n_fibers": N,
        "locus": [("chr21",), (10000000,), (10002048,)],
        "cell_type_targets": {
            "GM12878": torch.rand((B, L)),
            "K562": torch.rand((B, L))
        },
        "cell_type_masks": {
            "GM12878": gm_mask,
            "K562": k562_mask
        }
    }

    dummy_loader = [dummy_batch]

    results = evaluator.evaluate(dummy_loader)

    print("\n--- Evaluation Summary ---")
    print(f"Composite Loss: {results['composite']['loss']:.6f}")
    print(f"Composite Pearson R: {results['composite']['pearson_r']:.4f}")

    print("\nPer-Cell-Type Breakdown:")
    for ct, metrics in results["per_cell_type"].items():
        print(f"  [{ct}] MSE Loss: {metrics['loss']:.6f} | Pearson R: {metrics['pearson_r']:.4f}")

    print("\nEvaluator test successfully completed!")


if __name__ == "__main__":
    tester()
