import torch
from torch import nn
from .contr_loss import SymInfoNCELoss
from lightning import pytorch as pl
from .charbonnier import CharbonnierLoss

class ReconLoss(nn.Module):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('ReconLoss')
        parser.add_argument('--lam_self_recon', help='Weight for self reconstruction term', type=float, default=10.0)
        parser.add_argument('--lam_cross_recon', help='Weight for cross reconstruction term', type=float, default=100.0)

    def __init__(self, args) -> None:
        super().__init__()
        self.l1 = CharbonnierLoss()
        self.lam_self_recon = args.lam_self_recon
        self.lam_cross_recon = args.lam_cross_recon

    def forward(self, x, y, y_cross, return_terms=False):
        x_11, x_12, x_21, x_22 = x
        y_11, y_12, y_21, y_22 = y
        y_cross_11, y_cross_12, y_cross_21, y_cross_22 = y_cross  # y_12 and y_21 are crossed predictions

        self_recon_loss = self.l1(x_11, y_11) + self.l1(x_12, y_12) + self.l1(x_21, y_21) + self.l1(x_22, y_22)
        cross_recon_loss = self.l1(x_11, y_cross_11) + self.l1(x_12, y_cross_12) + self.l1(x_21, y_cross_21) + self.l1(x_12, y_cross_22)

        loss = self.lam_self_recon*self_recon_loss + self.lam_cross_recon*cross_recon_loss

        if return_terms:
            loss_terms = {
                'self_recon_loss': self_recon_loss,
                'cross_recon_loss': cross_recon_loss
            }
            return loss, loss_terms
        else:
            return loss

class ContrastLoss(nn.Module):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('ContrastLoss')
        parser.add_argument('--lam_contrast', help='Weight for contrastive term. (Default 1.0)', type=float, default=1.0)
        parser.add_argument('--proj_dim', help='Projection dimension for contrastive loss. (Default 8.)', type=int, default=8)

    def __init__(self, args, ndim) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()
        if not hasattr(args, 'temp_contrast'):
            args.temp_contrast = 0.1
        self.symcont = SymInfoNCELoss(temp=args.temp_contrast)
        self.lam_contrast = args.lam_contrast
        self.proj_dim = args.proj_dim
        if ndim > 0:
            self.mlp = nn.Linear(ndim, self.proj_dim)
        else:
            self.mlp = None

    def forward(self, x_1, x_2, return_terms=False):
        if self.mlp:
            z_1, z_2 = self.mlp(x_1), self.mlp(x_2)
        else:
            z_1, z_2 = x_1, x_2
        contr_loss = self.symcont(z_1, z_2)
        loss = self.lam_contrast*contr_loss

        if return_terms:
            loss_terms = {
                'contr_loss': contr_loss,
            }
            return loss, loss_terms
        else:
            return loss

    @staticmethod
    def get_zero_loss_terms():
        return {'contr_loss': torch.tensor(0.0, requires_grad=True)}
