"""
Metrics for model evaluation.

Bulk tensors are (B, K, L); assay masks are (B, K) with 1 where that sample's cell type has data for
the assay. Masked-out assays contribute no loss and no gradient.
"""

import torch


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over the (sample, assay) pairs that have data. Equals plain MSE when mask is all ones."""
    sq = (pred.float() - target.float()) ** 2 * mask.float()[:, :, None]
    return sq.sum() / (mask.float().sum() * pred.shape[-1]).clamp(min=1)


def per_assay_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Per-assay MSE averaged over the samples that have that assay.
    Returns (losses (K,), counts (K,)); losses for assays with count 0 are meaningless (0)."""
    per_sample = ((pred.float() - target.float()) ** 2).mean(dim=-1)   # (B, K)
    mask = mask.float()
    counts = mask.sum(dim=0)                                          # (K,)
    losses = (per_sample * mask).sum(dim=0) / counts.clamp(min=1)
    return losses, counts
