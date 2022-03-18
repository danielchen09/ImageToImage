import torch
import torch.nn as nn
from pix2pix.CBlock import CBlock


class Discriminator(nn.Module):
    """
    C64-C128-C256-C512

    After the last layer, a convolution is applied to map to a 1-dimensional output,
    followed by a Sigmoid function (or use logit in loss)

    As an exception to the above notation, BatchNorm is not applied to the first C64
    layer. All ReLUs are leaky, with slope 0.2
    """
    def __init__(self, in_channels=3, hidden_channels=None):
        super(Discriminator, self).__init__()
        if hidden_channels is None:
            hidden_channels = [64, 128, 256, 512]

        layers = []
        layers.append(CBlock(in_channels * 2, hidden_channels[0], batch_norm=False))
        for i in range(1, len(hidden_channels)):
            layers.append(CBlock(hidden_channels[i - 1], hidden_channels[i],
                                 stride=1 if i == len(hidden_channels) - 1 else 2))
        layers.append(nn.Conv2d(hidden_channels[-1], 1, 4, 1, 1, padding_mode='reflect'))
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))


if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    print(model(x, y).shape)