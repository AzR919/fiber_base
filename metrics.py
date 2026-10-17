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


def per_window_pearson(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    """Pearson r across the positions of each window (profile shape), (B, K).
    NaN where the sample has no data for the assay or either profile is flat (std < eps)."""
    x, y = pred.detach().double(), target.detach().double()
    xc, yc = x - x.mean(dim=-1, keepdim=True), y - y.mean(dim=-1, keepdim=True)
    sx, sy = xc.pow(2).mean(dim=-1).sqrt(), yc.pow(2).mean(dim=-1).sqrt()
    r = (xc * yc).mean(dim=-1) / (sx * sy)
    invalid = (mask == 0) | (sx < eps) | (sy < eps)
    return r.masked_fill(invalid, float("nan")).float()


class PearsonMeter:
    """
    Streaming Pearson r per assay, pooled over every position of every window that has the assay —
    the correlation counterpart of the MSE we report. Batch statistics are merged with Chan et al.'s
    parallel update in float64, so it is exact (no catastrophic cancellation) over tens of millions of
    positions without storing them. Also tracks the MSE over the same positions (.mse), so a meter
    restricted to a subset of positions (e.g. bases in promoter cCREs) gives both metrics for it.
    """
    def __init__(self, num_assays: int):
        zeros = lambda: torch.zeros(num_assays, dtype=torch.float64)
        self.n, self.mean_x, self.mean_y = zeros(), zeros(), zeros()
        self.m2_x, self.m2_y, self.c_xy = zeros(), zeros(), zeros()
        self.sse = zeros()

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """pred/target: (B, K, L). mask: (B, K), 1 where the sample has data for the assay, or (B, K, L)
        to also select positions (e.g. assay mask × a per-bp region-type mask)."""
        x = pred.detach().double()
        y = target.detach().to(x.device).double()
        w = mask.detach().to(x.device).double()
        w = (w[:, :, None] if w.dim() == 2 else w).expand_as(x)
        self.sse += (w * (x - y) ** 2).sum(dim=(0, 2)).cpu()
        n_b = w.sum(dim=(0, 2))
        mean_x_b = (w * x).sum(dim=(0, 2)) / n_b.clamp(min=1)
        mean_y_b = (w * y).sum(dim=(0, 2)) / n_b.clamp(min=1)
        dx, dy = x - mean_x_b[None, :, None], y - mean_y_b[None, :, None]
        m2_x_b, m2_y_b, c_xy_b = (w * dx * dx).sum(dim=(0, 2)), (w * dy * dy).sum(dim=(0, 2)), (w * dx * dy).sum(dim=(0, 2))
        n_b, mean_x_b, mean_y_b, m2_x_b, m2_y_b, c_xy_b = (t.cpu() for t in (n_b, mean_x_b, mean_y_b, m2_x_b, m2_y_b, c_xy_b))

        n = self.n + n_b
        delta_x, delta_y = mean_x_b - self.mean_x, mean_y_b - self.mean_y
        cross = self.n * n_b / n.clamp(min=1)
        self.mean_x += delta_x * n_b / n.clamp(min=1)
        self.mean_y += delta_y * n_b / n.clamp(min=1)
        self.m2_x += m2_x_b + delta_x * delta_x * cross
        self.m2_y += m2_y_b + delta_y * delta_y * cross
        self.c_xy += c_xy_b + delta_x * delta_y * cross
        self.n = n

    @property
    def value(self) -> torch.Tensor:
        """(K,) pooled Pearson r; NaN for assays never seen or with zero variance."""
        denom = torch.sqrt(self.m2_x * self.m2_y)
        r = self.c_xy / denom
        return r.masked_fill((self.n == 0) | (denom == 0), float("nan")).float()

    @property
    def mse(self) -> torch.Tensor:
        """(K,) MSE over the same positions; NaN for assays never seen."""
        return (self.sse / self.n).masked_fill(self.n == 0, float("nan")).float()
