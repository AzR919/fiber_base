"""
Main train loop

"""

import torch
import torch.nn as nn

from utils import *
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, dataset, epochs=10,
                 batch_size=32, lr=1e-4, patience=5):

        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.iters_per_epoch = dataset.iters_per_epoch
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=patience
        )
        self.criterion = nn.MSELoss()

    def train_step(self, batch):

        self.model.train()
        m6as, dna, target = [b.to(self.device) for b in batch[:3]]
        self.optimizer.zero_grad()
        output = self.model(m6as, dna)
        loss = self.criterion(output, target)
        if torch.isnan(output).any().item() or torch.isnan(loss):
            torch.save(output, "./ignore/output.pt")
            torch.save(m6as, "./ignore/input.pt")
            torch.save(target, "./ignore/target.pt")
            exit(-1)
        loss.backward()
        self.optimizer.step()
        return loss.item(), output

    def train(self, save_dir):

        loader = DataLoader(self.dataset, batch_size=self.batch_size)

        losses = []
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                loss, output = self.train_step(batch)
                total_loss += loss
            avg_loss = total_loss / self.iters_per_epoch
            self.scheduler.step(avg_loss)
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
            losses.append(avg_loss)

            plot_sample(save_dir, batch[0], output, batch[2], batch[3], epoch)

        plot_loss(save_dir, losses, epoch+1)
