from typing import Tuple
import torch
from torch import nn
from .norm import InstanceNorm
from .resnet import ResNetDownBlock

class ContentEncoder(nn.Module):
    def __init__(self, embed_total=2048) -> None:
        super().__init__()

        self.embed_total = embed_total
        self.block_out_channels = [self.embed_total//8, self.embed_total//8, self.embed_total//4, self.embed_total//2]
        self.block_in_channels = [64] + self.block_out_channels[:-1]
        self.n_sub_blocks = [3, 4, 6, 3]

        self.pre_filter = nn.Sequential(
            nn.Conv2d(3, 64, (7, 7), padding=(3, 3), bias=False),
            InstanceNorm(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.blocks = nn.ModuleList(
            nn.Sequential(
                ResNetDownBlock(in_channels, out_channels, factor=2, normalize='instance'),
                *[ResNetDownBlock(out_channels, out_channels, factor=1, normalize='instance') for _ in range(sub_blocks-1)]
            )
            for in_channels, out_channels, sub_blocks in zip(self.block_in_channels, self.block_out_channels, self.n_sub_blocks)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x_filt = self.pre_filter(x)
        ys = [x_filt]
        for block in self.blocks:
            ys.append(block(ys[-1]))
        return tuple(ys[1:])


class AppearanceEncoder(nn.Module):
    def __init__(self, embed_total=2048) -> None:
        super().__init__()

        self.embed_total = embed_total
        self.block_out_channels = [self.embed_total//8, self.embed_total//8, self.embed_total//4, self.embed_total//2]
        self.block_in_channels = [64] + self.block_out_channels[:-1]
        self.n_sub_blocks = [3, 4, 6, 3]

        self.pre_filter = nn.Sequential(
            nn.Conv2d(3, 64, (7, 7), padding=(3, 3), bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.blocks = nn.ModuleList(
            nn.Sequential(
                ResNetDownBlock(in_channels, out_channels, factor=2, normalize=None),
                *[ResNetDownBlock(out_channels, out_channels, factor=1, normalize=None) for _ in range(sub_blocks-1)]
            )
            for in_channels, out_channels, sub_blocks in zip(self.block_in_channels, self.block_out_channels, self.n_sub_blocks)
        )

    def forward(self, x: torch.Tensor, return_std: bool = False) -> Tuple[torch.Tensor]:
        x_filt = self.pre_filter(x)
        ys = [x_filt]
        for block in self.blocks:
            ys.append(block(ys[-1]))
        ret = tuple(y.mean((2, 3)) for y in ys[1:])
        if return_std:
            ret += tuple(torch.std(y, (2, 3)) for y in ys[1:])
        return ret
