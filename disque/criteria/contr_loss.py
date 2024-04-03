import torch
from torch import nn
from ..utils import distributed as dist

class InfoNCELoss(nn.Module):
    def __init__(self, temp=1) -> None:
        super().__init__()
        self.temp = temp
        self.cross_ent = nn.CrossEntropyLoss()

    def forward(self, x_1, x_2):
        '''
        x_i: N x F
        x_1[i], x_2[i] are +ve pairs
        x_1[i], x_2[j] are -ve pairs
        '''
        N = x_1.shape[0]
        # Apply L2 normalization
        x_1 = nn.functional.normalize(x_1, p=2, dim=1)
        x_2 = nn.functional.normalize(x_2, p=2, dim=1)

        x_2_all = dist.gather(x_2)

        # Compute predicted logits both ways for a symmetrized loss
        # Equivalent to logits = torch.matmul(x_1, x_2.T) without distributed or einsum
        logits = torch.einsum('nf,mf->nm', x_1, x_2_all)
        labels = torch.arange(x_1.shape[0], device=logits.device) + dist.rank() * N
        return self.cross_ent(logits / self.temp, labels)


class SymInfoNCELoss(nn.Module):
    def __init__(self, temp=1) -> None:
        super().__init__()
        self.temp = temp
        self.cross_ent = nn.CrossEntropyLoss()

    def forward(self, x_1, x_2):
        '''
        x_i: N x F
        x_1[i], x_2[i] are +ve pairs
        x_1[i], x_2[j] are -ve pairs
        '''
        N = x_1.shape[0]
        # Apply L2 normalization
        x_1 = nn.functional.normalize(x_1, p=2, dim=1)
        x_2 = nn.functional.normalize(x_2, p=2, dim=1)

        x_1_all = dist.gather(x_1)
        x_2_all = dist.gather(x_2)

        # Compute predicted logits both ways for a symmetrized loss
        logits_12 = torch.einsum('nf,mf->nm', x_1, x_2_all)
        logits_21 = torch.einsum('nf,mf->nm', x_2, x_1_all)
        labels = torch.arange(x_1.shape[0], device=logits_12.device) + dist.rank() * N
        crossent = self.cross_ent(logits_12 / self.temp, labels) + self.cross_ent(logits_21 / self.temp, labels)
        return crossent