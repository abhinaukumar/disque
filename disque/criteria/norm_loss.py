import torch
from torch import nn


class NormLoss(nn.Module):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('NormLoss')
        parser.add_argument('--lam_norm', help='Weight for norm loss term. (Default 1.0)', type=float, default=1.0)

    def __init__(self, args) -> None:
        super().__init__()
        self.lam_norm = args.lam_norm

    def forward(self, x, return_terms=False):
        norm_loss = torch.mean(torch.linalg.norm(x, dim=-1))
        loss = self.lam_norm*norm_loss

        if return_terms:
            loss_terms = {
                'norm_loss': norm_loss,
            }
            return loss, loss_terms
        else:
            return loss

    @staticmethod
    def get_zero_loss_terms():
        return {'norm_loss': torch.tensor(0.0, requires_grad=True)}