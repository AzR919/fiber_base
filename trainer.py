"""
Main train loop

"""

import wandb
import torch
import torch.nn as nn

from utils import *
from torch.utils.data import DataLoader

class Trainer:
    # 1. Added val_dataset to __init__ parameters
    def __init__(self, model, train_dataset, val_dataset=None, epochs=10,
                 batch_size=32, lr=1e-4, patience=5,
                 run_name="debug", config=None):

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset  # Save reference locally
        self.epochs = epochs
        self.iters_per_epoch = config.iters_per_epoch
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=patience
        )
        self.criterion = nn.MSELoss()

        if "sweep" in run_name.lower():
            run_name += self.run_name_sup(config.input_flags)

        self.wandb_run = wandb.init(
            entity="liblab",
            project="Fiber",
            name=run_name,
            config=config,
        )

        self.wandb_run.watch(self.model)

        # 2. Configure both loss profiles to index cleanly on the same x-axis step
        wandb.define_metric("epoch")
        wandb.define_metric("train_loss", step_metric="epoch")
        wandb.define_metric("val_loss", step_metric="epoch")

        self.config = config

        if sum(config.input_flags) == 0:
            print("Encountered [0,0,0,0,0] run. Skipping training evaluation...")
            # Assign matching maximum error ceilings to satisfy the sweep optimizer targets
            wandb.log({"train_loss": float('inf'), "val_loss": float('inf'), "epoch": 0})
            sys.exit(0)

    def train_step(self, batch):
        self.model.train()
        m6as, dna, target = [b.to(self.device) for b in batch[:3]]
        self.optimizer.zero_grad()
        output, processed_fibers = self.model(m6as, dna)
        loss = self.criterion(output, target)

        if torch.isnan(output).any().item() or torch.isnan(loss):
            torch.save(output, "./ignore/output.pt")
            torch.save(m6as, "./ignore/input.pt")
            torch.save(target, "./ignore/target.pt")
            exit(-1)

        loss.backward()
        self.optimizer.step()
        return loss.item(), output, processed_fibers

    # 3. Dedicated validation logic isolated from optimizer calls
    def val_step(self, batch):
        self.model.eval()
        m6as, dna, target = [b.to(self.device) for b in batch[:3]]
        with torch.no_grad():
            output, processed_fibers = self.model(m6as, dna)
            loss = self.criterion(output, target)
        return loss.item(), output, processed_fibers

    def train(self, save_dir):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size)

        # Only build evaluation dataloader if validation set was provided
        if self.val_dataset is not None:
            val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)

        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            # Rotate epoch seeds inside your dataset streams to change sample distributions
            self.train_dataset.set_epoch(epoch)
            if self.val_dataset is not None:
                self.val_dataset.set_epoch(epoch)

            # --- TRAINING PHASE ---
            total_train_loss = 0
            for batch in train_loader:
                t_loss, t_output, t_processed_fibers = self.train_step(batch)
                total_train_loss += t_loss
            avg_train_loss = total_train_loss / self.iters_per_epoch
            train_losses.append(avg_train_loss)

            # --- VALIDATION PHASE ---
            avg_val_loss = None
            if self.val_dataset is not None:
                total_val_loss = 0
                for v_batch in val_loader:
                    v_loss, v_output, v_processed_fibers = self.val_step(v_batch)
                    total_val_loss += v_loss
                avg_val_loss = total_val_loss / self.iters_per_epoch
                val_losses.append(avg_val_loss)
                # Update scheduler with the training baseline
                self.scheduler.step(avg_val_loss)
            else:
                # Update scheduler with the training baseline
                self.scheduler.step(avg_train_loss)

            # Log outputs to stdout and W&B tracking panel simultaneously
            if avg_val_loss is not None:
                print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
                self.wandb_run.log({
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "epoch": epoch
                })
            else:
                print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f}")
                self.wandb_run.log({
                    "train_loss": avg_train_loss,
                    "epoch": epoch
                })

            # Plot visual samples using the final batch evaluated
            plot_sample_out_fibers_wandb(
                self.wandb_run, save_dir, batch[0], self.config.input_flags,
                self.config.num_input_features, t_output, t_processed_fibers,
                batch[2], batch[3], epoch, avg_train_loss, mode="Train"
            )

            if avg_val_loss is not None:
                plot_sample_out_fibers_wandb(
                    self.wandb_run, save_dir, v_batch[0], self.config.input_flags,
                    self.config.num_input_features, v_output, v_processed_fibers,
                    v_batch[2], v_batch[3], epoch, avg_val_loss, mode="Val"
                )

        plot_loss(save_dir, train_losses, epoch+1)

    def run_name_sup(self, input_flags):
        feature_names = ["m6a", "cpg", "msp", "nuc", "fire_msp"]
        sup_str = ""
        for name, flag in zip(feature_names, input_flags):
            if not flag: continue
            sup_str += f"_{name}"
        return sup_str

#--------------------------------------------------------------------------------------------------
# testing

def tester():
    pass

if __name__=="__main__":

    tester()
