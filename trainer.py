"""
Main training loop and execution manager.
"""

import os
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from utils import *
from vis_utils import plot_evaluation_dashboard_t, plot_loss

class Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, wandb_run=None,
                 epochs=10, batch_size=32, lr=1e-4, patience=5, config=None):

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.wandb_run = wandb_run
        self.epochs = epochs
        self.batch_size = batch_size
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=patience
        )
        self.criterion = nn.MSELoss()

        self.scaler = torch.amp.GradScaler(self.device)

    def train_step(self, batch):
        self.model.train()
        fiber_features, target, forward_kwargs = unpack_batch(batch, self.device)

        self.optimizer.zero_grad()

        # Execute forward pass under mixed precision context
        with torch.amp.autocast(self.device_type):
            output, processed_fibers = self.model(fiber_features, **forward_kwargs)
            loss = self.criterion(output, target)

        # Scale loss and backpropagate using the Gradient Scaler
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item(), output, processed_fibers

    def val_step(self, batch):
        self.model.eval()
        fiber_features, target, forward_kwargs = unpack_batch(batch, self.device)

        with torch.no_grad():
            # Run validation evaluation under mixed precision context
            with torch.amp.autocast(self.device_type):
                output, processed_fibers = self.model(fiber_features, **forward_kwargs)
                loss = self.criterion(output, target)

        return loss.item(), output, processed_fibers

    def train(self, save_dir):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            worker_init_fn=seed_worker
        )

        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                worker_init_fn=seed_worker
            )

        train_losses = []
        val_losses = []

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for epoch in range(self.epochs):
            if hasattr(self.train_dataset, "set_epoch"):
                self.train_dataset.set_epoch(epoch)
            if self.val_dataset is not None and hasattr(self.val_dataset, "set_epoch"):
                self.val_dataset.set_epoch(epoch)

            output_assays = getattr(self.config, "output_assays", ["atac"])

            # --- TRAINING PHASE ---
            train_meter = AverageMeter()
            train_assay_meters = {a: AverageMeter() for a in output_assays}
            last_t_batch, last_t_output, last_t_fibers = None, None, None

            for i, batch in enumerate(train_loader):
                t_loss, t_output, t_processed_fibers = self.train_step(batch)
                if epoch == 0 and i == 0 and self.device.type == "cuda":
                    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
                    alloc_mb = torch.cuda.memory_allocated() / 1024**2
                    print(f"GPU memory after first step — peak: {peak_mb:.0f} MB | current: {alloc_mb:.0f} MB", flush=True)
                    self.wandb_run.log({"gpu_peak_mb": peak_mb, "gpu_alloc_mb": alloc_mb})
                n = batch["fiber_features"].size(0)
                train_meter.update(t_loss, n=n)
                with torch.no_grad():
                    tgt = batch["target_bulk"].to(self.device)
                    for k, assay in enumerate(output_assays):
                        train_assay_meters[assay].update(
                            self.criterion(t_output.detach()[:, k], tgt[:, k]).item(), n=n
                        )
                last_t_batch = batch
                last_t_output = t_output
                last_t_fibers = t_processed_fibers

            avg_train_loss = train_meter.avg
            train_losses.append(avg_train_loss)

            # --- VALIDATION PHASE ---
            avg_val_loss = None
            val_assay_meters = {a: AverageMeter() for a in output_assays}
            last_v_batch, last_v_output, last_v_fibers = None, None, None

            if val_loader is not None:
                val_meter = AverageMeter()
                for v_batch in val_loader:
                    v_loss, v_output, v_processed_fibers = self.val_step(v_batch)
                    n = v_batch["fiber_features"].size(0)
                    val_meter.update(v_loss, n=n)
                    with torch.no_grad():
                        tgt = v_batch["target_bulk"].to(self.device)
                        for k, assay in enumerate(output_assays):
                            val_assay_meters[assay].update(
                                self.criterion(v_output[:, k], tgt[:, k]).item(), n=n
                            )
                    last_v_batch = v_batch
                    last_v_output = v_output
                    last_v_fibers = v_processed_fibers

                avg_val_loss = val_meter.avg
                val_losses.append(avg_val_loss)
                self.scheduler.step(avg_val_loss)
            else:
                self.scheduler.step(avg_train_loss)

            # --- LOGGING & DASHBOARDS ---
            log_dict = {"train_loss": avg_train_loss, "epoch": epoch}
            if avg_val_loss is not None:
                log_dict["val_loss"] = avg_val_loss
                print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            else:
                print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.6f}")

            for assay in output_assays:
                log_dict[f"train_loss_{assay}"] = train_assay_meters[assay].avg
                if avg_val_loss is not None:
                    log_dict[f"val_loss_{assay}"] = val_assay_meters[assay].avg

            train_assay_avg = {a: train_assay_meters[a].avg for a in output_assays}
            val_assay_avg   = {a: val_assay_meters[a].avg  for a in output_assays}

            # Generate & Log Train Dashboard Plot
            if last_t_batch is not None:
                fig_t = plot_evaluation_dashboard_t(
                    last_t_batch["fiber_features"],
                    self.train_dataset.input_flags,
                    last_t_output,
                    last_t_fibers,
                    last_t_batch["target_bulk"],
                    last_t_batch["locus"],
                    last_t_batch["cell_type"],
                    output_assays,
                    assay_avg_losses=train_assay_avg,
                    mode="Train"
                )
                log_dict["Train_Dashboard"] = wandb.Image(fig_t)
                plt.close(fig_t)

            # Generate & Log Val Dashboard Plot
            if last_v_batch is not None:
                fig_v = plot_evaluation_dashboard_t(
                    last_v_batch["fiber_features"],
                    self.train_dataset.input_flags,
                    last_v_output,
                    last_v_fibers,
                    last_v_batch["target_bulk"],
                    last_v_batch["locus"],
                    last_v_batch["cell_type"],
                    output_assays,
                    assay_avg_losses=val_assay_avg,
                    mode="Val"
                )
                log_dict["Val_Dashboard"] = wandb.Image(fig_v)
                plt.close(fig_v)

            self.wandb_run.log(log_dict)

        # Final loss summary curve & model save
        plot_loss(save_dir, train_losses, self.epochs, getattr(self.config, "output_assays", ["atac"]))
        self.model.save_model(save_dir, self.epochs, external_config=self.config)


#--------------------------------------------------------------------------------------------------
# Testing

def tester():
    pass

if __name__ == "__main__":
    tester()
