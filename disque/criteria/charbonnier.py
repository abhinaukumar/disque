import torch
from torch import nn

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x1, x2):
        return torch.mean(torch.sqrt((x1 - x2)**2 + self.eps**2))
