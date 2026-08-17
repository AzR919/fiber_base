"""
Main training loop and execution manager.
"""

import os
import sys
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluator import Evaluator, test_dataset_from_path_and_extra_args
from utils import *

class Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, eval_config_path=None,
                 epochs=10, batch_size=32, lr=1e-4, patience=5,
                 run_name="debug", config=None):

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.eval_config_path = eval_config_path
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

        # Initialize Gradient Scaler for Automatic Mixed Precision (AMP)
        self.scaler = torch.amp.GradScaler(self.device)

        if hasattr(config, "input_flags") and sum(config.input_flags) == 0:
            print("Encountered [0,0,0,0,0] run. Skipping training evaluation...")
            wandb.init(entity="liblab", project="Fiber", name=run_name, config=config)
            wandb.log({"train_loss": float('inf'), "val_loss": float('inf'), "epoch": 0})
            sys.exit(0)

        if "sweep" in run_name.lower() and hasattr(config, "input_flags"):
            run_name += self._build_run_name_suffix(config.input_flags)

        self.wandb_run = wandb.init(
            entity="liblab",
            project="fiber",
            name=run_name,
            config=config,
        )

        self.wandb_run.watch(self.model)

        wandb.define_metric("epoch")
        wandb.define_metric("train_loss", step_metric="epoch")
        wandb.define_metric("val_loss", step_metric="epoch")
        if eval_config_path is not None:
            wandb.define_metric("test_loss")
        wandb.watch(model, log="all")

    def _unpack_batch(self, batch):
        """Extracts batch dictionary elements and moves tensors to target device."""
        fiber_features = batch["fiber_features"].to(self.device)
        target = batch["target_bulk"].to(self.device)
        n_fibers = batch["n_fibers"].to(self.device)

        kwargs = {"n_fibers": n_fibers}

        if "genomic_dna" in batch:
            kwargs["dna"] = batch["genomic_dna"].to(self.device)
        if "fiber_dna" in batch:
            kwargs["dna_tensor"] = batch["fiber_dna"].to(self.device)

        return fiber_features, target, kwargs

    def train_step(self, batch):
        self.model.train()
        fiber_features, target, forward_kwargs = self._unpack_batch(batch)

        self.optimizer.zero_grad()
        # output, processed_fibers = self.model(fiber_features, **forward_kwargs)
        # loss = self.criterion(output, target)
        # loss.backward()
        # self.optimizer.step()

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
        fiber_features, target, forward_kwargs = self._unpack_batch(batch)

        # with torch.no_grad():
        #     output, processed_fibers = self.model(fiber_features, **forward_kwargs)
        #     loss = self.criterion(output, target)

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

        for epoch in range(self.epochs):
            if hasattr(self.train_dataset, "set_epoch"):
                self.train_dataset.set_epoch(epoch)
            if self.val_dataset is not None and hasattr(self.val_dataset, "set_epoch"):
                self.val_dataset.set_epoch(epoch)

            # --- TRAINING PHASE ---
            train_meter = AverageMeter()
            last_t_batch, last_t_output, last_t_fibers = None, None, None

            for batch in train_loader:
                t_loss, t_output, t_processed_fibers = self.train_step(batch)
                train_meter.update(t_loss, n=batch["fiber_features"].size(0))
                last_t_batch = batch
                last_t_output = t_output
                last_t_fibers = t_processed_fibers

            avg_train_loss = train_meter.avg
            train_losses.append(avg_train_loss)

            # --- VALIDATION PHASE ---
            avg_val_loss = None
            last_v_batch, last_v_output, last_v_fibers = None, None, None

            if val_loader is not None:
                val_meter = AverageMeter()
                for v_batch in val_loader:
                    v_loss, v_output, v_processed_fibers = self.val_step(v_batch)
                    val_meter.update(v_loss, n=v_batch["fiber_features"].size(0))
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

            # Generate & Log Train Dashboard Plot
            if last_t_batch is not None:
                fig_t = plot_evaluation_dashboard(
                    last_t_batch["fiber_features"],
                    self.train_dataset.input_flags,
                    last_t_output,
                    last_t_fibers,
                    last_t_batch["target_bulk"],
                    last_t_batch["locus"],
                    last_t_batch["cell_type"],
                    self.train_dataset.bulk_name,
                    avg_loss=avg_train_loss,
                    mode="Train"
                )
                log_dict["Train_Dashboard"] = wandb.Image(fig_t)
                plt.close(fig_t)

            # Generate & Log Val Dashboard Plot
            if last_v_batch is not None:
                fig_v = plot_evaluation_dashboard(
                    last_v_batch["fiber_features"],
                    self.train_dataset.input_flags,
                    last_v_output,
                    last_v_fibers,
                    last_v_batch["target_bulk"],
                    last_v_batch["locus"],
                    last_v_batch["cell_type"],
                    self.train_dataset.bulk_name,
                    avg_loss=avg_val_loss,
                    mode="Val"
                )
                log_dict["Val_Dashboard"] = wandb.Image(fig_v)
                plt.close(fig_v)

            self.wandb_run.log(log_dict)

        # Final loss summary curve & model save
        plot_loss(save_dir, train_losses, self.epochs, self.config.bulk_name)
        self.model.save_model(save_dir, self.epochs, external_config=self.config)

        if self.eval_config_path is not None:
            # -------------------------------------------------------------------------
            # Testing & WandB Logging
            # -------------------------------------------------------------------------
            print("\n" + "=" * 60)
            print(" Running Final Model Test & Deconvolution Dashboard...")
            print("=" * 60)

            extra_args = {
                "input_flags": self.config.input_flags,
                "return_dna": self.config.dna_type != "none",
            }
            test_set, eval_seed = test_dataset_from_path_and_extra_args(self.eval_config_path, extra_args)

            evaluator = Evaluator(self.model, test_set, batch_size=1, num_plots_to_log=5, device=self.device, seed=eval_seed)

            eval_results = evaluator.evaluate()
            test_log_dict = {"test_loss": eval_results["composite"]["loss"]}

            # Select locus records to visualize (e.g., top N samples or first N samples)
            locus_records = eval_results.get("locus_records", [])
            wandb_image_list = []

            for idx, record in enumerate(locus_records):

                # Generate the 2-column deconvolution plot
                fig = plot_evaluator_record(
                    record=record,
                    input_flags=test_set.input_flags,
                    loss=eval_results["composite"]["loss"],
                    ct_losses=eval_results["per_cell_type"],
                    bulk_name=test_set.bulk_name,
                    mode="Test"
                )

                # Extract locus info for clean WandB image captioning
                chr_name = record["locus"][0][0]
                start = record["locus"][1][0]
                end = record["locus"][2][0]
                num_locus = eval_results["num_locus"]
                caption = f"Locus {idx}/{num_locus}: {chr_name}:{start}-{end}"

                # Convert Matplotlib figure to wandb.Image
                wandb_image_list.append(
                    wandb.Image(fig, caption=caption)
                )

                # Always close local figures to prevent memory leaks in Matplotlib
                plt.close(fig)

            # Log all dashboard figures under a dedicated gallery panel in WandB
            test_log_dict["Evaluation/Deconvolution_Dashboards"] = wandb_image_list
            wandb.log(test_log_dict)

            print(f" Successfully logged {len(wandb_image_list)} evaluation dashboards to WandB!")

    def _build_run_name_suffix(self, input_flags):
        feature_names = ["m6a", "cpg", "msp", "nuc", "fire_msp"]
        sup_str = ""
        for name, flag in zip(feature_names, input_flags):
            if flag:
                sup_str += f"_{name}"
        return sup_str


#--------------------------------------------------------------------------------------------------
# Testing

def tester():
    pass

if __name__ == "__main__":
    tester()
