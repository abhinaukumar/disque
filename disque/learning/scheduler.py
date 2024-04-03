from typing import List
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from argparse import ArgumentParser

class WarmupCosineAnnealing(LRScheduler):
    def __init__(self, optimizer: Optimizer, args: ArgumentParser, steps_per_epoch: int, verbose: bool = False):
        self.epochs = args.epochs
        self.warmup_epochs = args.warmup_epochs
        self.warmup_steps = self.warmup_epochs * steps_per_epoch
        self.lr = args.lr
        self.eta_min = self.lr * (args.lr_decay ** 3)
        self.warmup_from = args.warmup_from
        self.warmup_to = self._get_cosine_lr(self.warmup_epochs)
        self.cosine_to = args.cosine_to
        super().__init__(optimizer, verbose=verbose)

    def _get_linear_lr(self, frac):
        return self.warmup_from + frac * (self.warmup_to - self.warmup_from)

    def _get_cosine_lr(self, epoch):
        return self.eta_min + (self.lr - self.eta_min) * (
                    1 + np.cos(np.pi * epoch / self.epochs)) / 2

    def get_lr(self) -> List[float]:
        warmup_frac = self._step_count / self.warmup_steps
        if warmup_frac < 1:
            lr = self._get_linear_lr(warmup_frac)
        else:
            epoch = (self._step_count - self.warmup_steps) + self.warmup_epochs
            lr = max(self.cosine_to, self._get_cosine_lr(epoch))

        return [lr for _ in self.optimizer.param_groups]
