from torch import nn
import torch

class InstanceNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.randn(1, self.channels, 1, 1))
        self.beta = nn.Parameter(torch.randn(1, self.channels, 1, 1))

    def forward(self, x):
        x_mean = x.mean((-1, -2))
        x_std = torch.sqrt(torch.abs((x**2).mean((-1, -2)) - x_mean**2) + self.eps)
        return (x - x_mean.unsqueeze(-1).unsqueeze(-1)) / x_std.unsqueeze(-1).unsqueeze(-1) * self.gamma + self.beta
