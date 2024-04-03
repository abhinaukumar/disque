from torch import nn

from .norm import InstanceNorm

NormClass_dict = {
    'batch': nn.BatchNorm2d,
    'instance': InstanceNorm,
    None: nn.Identity(),
}

class ResNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, factor=1, k=3, normalize=None):
        super().__init__()
        inter_channels = out_channels // 4
        NormClass = NormClass_dict[normalize] if normalize is not None else None
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, (1, 1), bias=False),
            NormClass(inter_channels) if normalize is not None else nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inter_channels, inter_channels, (k, k), stride=(factor, factor), padding=k//2, bias=False),
            NormClass(inter_channels) if normalize is not None else nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inter_channels, out_channels, (1, 1), bias=False),
            NormClass(out_channels) if normalize is not None else nn.LeakyReLU(0.1, inplace=True),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

        if (factor == 1) and (in_channels == out_channels):
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), stride=(factor, factor), bias=False),
                NormClass(out_channels) if normalize is not None else nn.LeakyReLU(0.1, inplace=True),
            )

    def forward(self, x):
        return self.act(self.skip(x) + self.trunk(x))


class ResNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, factor=1, k=3, normalize=None):
        super().__init__()
        inter_channels = in_channels // 4
        NormClass = NormClass_dict[normalize] if normalize is not None else None
        self.trunk = nn.Sequential(
            nn.ConvTranspose2d(in_channels, inter_channels, (1, 1), bias=False),
            NormClass(inter_channels) if normalize is not None else nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(inter_channels, inter_channels, (k, k), stride=(factor, factor), padding=k//2, output_padding=factor-1, bias=False),
            NormClass(inter_channels) if normalize is not None else nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(inter_channels, out_channels, (1, 1), bias=False),
            NormClass(out_channels) if normalize is not None else nn.LeakyReLU(0.1, inplace=True),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

        if (factor == 1) and (in_channels == out_channels):
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, (1, 1), stride=(factor, factor), output_padding=factor-1, bias=False),
                NormClass(out_channels) if normalize is not None else nn.LeakyReLU(0.1, inplace=True),
            )

    def forward(self, x):
        return self.act(self.skip(x) + self.trunk(x))
