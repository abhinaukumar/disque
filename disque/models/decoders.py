from typing import Tuple
import torch
from torch import nn
from .resnet import ResNetUpBlock

class Decoder(nn.Module):
    def __init__(self, embed_total=2048, add_one=False) -> None:
        super().__init__()
        self.embed_total = embed_total
        self.block_in_channels = [self.embed_total//2, self.embed_total//4, self.embed_total//8, self.embed_total//8]
        self.block_out_channels = self.block_in_channels[1:] + [64]
        self.n_sub_blocks = [3, 6, 4, 3]
        self.add = 1 if add_one else 0

        self.blocks = nn.ModuleList(
            nn.Sequential(
                ResNetUpBlock(in_channels, in_channels, factor=1, normalize=None),
                *[ResNetUpBlock(in_channels, in_channels, factor=1, normalize=None) for _ in range(sub_blocks-1)],
                ResNetUpBlock(in_channels, out_channels, factor=2, normalize=None)
            )
            for in_channels, out_channels, sub_blocks in zip(self.block_in_channels, self.block_out_channels, self.n_sub_blocks)
        )

        self.post_filter = nn.Conv2d(64, 3, (7, 7), padding=(3, 3), bias=False)

        self.embeds = nn.ModuleList(
            nn.Linear(in_channels, in_channels)
            for in_channels in self.block_in_channels
        )

    def forward(self, conts: Tuple[torch.Tensor], apps: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        conts = conts[::-1]
        apps = apps[::-1]
        z = 0
        for c, a, block, embed in zip(conts, apps, self.blocks, self.embeds):
            z = block((z + c)*embed(self.add + a).unsqueeze(-1).unsqueeze(-1))
        y = self.post_filter(z)
        return y
