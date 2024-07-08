import torch
import torch.nn.functional as F
from torch import nn


def si_snr_loss(pred, target, eps=1e-8):
    # Mean subtraction
    target_mean = target.mean(dim=-1, keepdim=True)
    pred_mean = pred.mean(dim=-1, keepdim=True)
    target = target - target_mean
    pred = pred - pred_mean

    # Projection
    s_target = torch.sum(target * pred, dim=-1, keepdim=True) * target / (
                torch.sum(target ** 2, dim=-1, keepdim=True) + eps)
    e_noise = pred - s_target

    # SI-SNR
    si_snr = 10 * torch.log10((torch.sum(s_target ** 2, dim=-1) + eps) / (torch.sum(e_noise ** 2, dim=-1) + eps))
    return -si_snr.mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        si_snr = si_snr_loss(pred, target)
        mse = F.mse_loss(pred, target)
        return self.alpha * si_snr + (1 - self.alpha) * mse


# Example usage
criterion = CombinedLoss(alpha=0.5)
pred = torch.randn(10, 32000)  # Example prediction
target = torch.randn(10, 32000)  # Example target
loss = criterion(pred, target)
print("Combined Loss:", loss.item())