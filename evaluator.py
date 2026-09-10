"""
Evaluator module for testing fiber data models.
Takes a model and a dataset, runs inference, and computes MSE metrics.
Supports both single-cell and mixed-cell batch types (auto-detected from batch keys).
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

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

    def _deconvolve_cell_type_bulk(self, processed_fibers, ct_mask, decoder_type):
        if decoder_type != "avg_n":
            raise ValueError(f"Deconvolution only supported for 'avg_n' decoder, got {decoder_type!r}")
        ct_fibers = processed_fibers[:, :, ct_mask]
        if ct_fibers.shape[-1] == 0:
            return torch.zeros((processed_fibers.shape[0], processed_fibers.shape[1]), device=self.device)
        return torch.mean(ct_fibers, dim=-1)

    def evaluate(self, save_path=None):
        self.model.eval()

        test_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            worker_init_fn=seed_worker,
            drop_last=False,
        )

        composite_loss_meter = AverageMeter()
        cell_type_loss_meters = {}
        locus_records = []
        valid_locus_count = 0
        decoder_type = getattr(self.model, "decoder_type", "avg_n")

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                fiber_features, target_bulk, forward_kwargs = unpack_batch(batch, self.device)

                with torch.amp.autocast(self.device_type, dtype=torch.float16):
                    pred_composite_bulk, processed_fibers = self.model(fiber_features, **forward_kwargs)

                comp_loss = self.criterion(pred_composite_bulk, target_bulk).item()
                composite_loss_meter.update(comp_loss)

                is_mixed = "cell_type_masks" in batch
                cell_type_preds = {}
                cell_type_losses = {}

                if is_mixed:
                    ct_targets = batch["cell_type_targets"]
                    ct_masks = batch["cell_type_masks"]

                    for ct_name, mask_tensor in ct_masks.items():
                        if ct_name not in cell_type_loss_meters:
                            cell_type_loss_meters[ct_name] = AverageMeter()

                        ct_mask = mask_tensor[0] if mask_tensor.dim() > 1 else mask_tensor
                        ct_mask = ct_mask.to(self.device)

                        pred_ct_bulk = self._deconvolve_cell_type_bulk(processed_fibers, ct_mask, decoder_type)
                        target_ct_bulk = ct_targets[ct_name].to(self.device)

                        ct_loss = self.criterion(pred_ct_bulk, target_ct_bulk).item()
                        cell_type_loss_meters[ct_name].update(ct_loss)

                        cell_type_preds[ct_name] = pred_ct_bulk.cpu()
                        cell_type_losses[ct_name] = ct_loss

                locus_record = {
                    "locus": batch["locus"],
                    "fiber_features": fiber_features.cpu(),
                    "processed_fibers": processed_fibers.cpu(),
                    "pred_bulk": pred_composite_bulk.cpu(),
                    "target_bulk": target_bulk.cpu(),
                    "pred_cell_type_bulks": cell_type_preds,
                    "target_cell_type_bulks": {k: v.cpu() for k, v in ct_targets.items()} if is_mixed else {},
                    "cell_type_masks": batch.get("cell_type_masks", {}),
                    "loss": comp_loss,
                    "cell_type_losses": cell_type_losses,
                }
                valid_locus_count += 1

                if len(locus_records) < self.num_to_save:
                    locus_records.append(locus_record)
                else:
                    j = self.rng.randint(0, valid_locus_count - 1)
                    if j < self.num_to_save:
                        locus_records[j] = locus_record

                if save_path is not None:
                    ct_losses_fmt = {k: {"loss": v} for k, v in cell_type_losses.items()}
                    print(f"saving idx{batch_idx}")
                    fig = plot_evaluator_record_t(
                        record_t=locus_record,
                        input_flags=self.model.init_args["input_flags"],
                        loss=comp_loss,
                        ct_losses=ct_losses_fmt,
                        bulk_name=self.dataset.bulk_name,
                        mode="Test"
                    )
                    plt.savefig(f"{save_path}test_e_{batch_idx}.png")
                    plt.close()

        return {
            "composite": {"loss": composite_loss_meter.avg},
            "per_cell_type": {ct: {"loss": cell_type_loss_meters[ct].avg} for ct in cell_type_loss_meters},
            "locus_records": locus_records,
            "num_locus": valid_locus_count * self.batch_size,
        }
