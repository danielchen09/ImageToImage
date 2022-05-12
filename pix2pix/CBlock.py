import torch.nn as nn


class CBlock(nn.Module):
    """
    Let Ck denote a Convolution-BatchNorm-ReLU layer
    with k filters. CDk denotes a Convolution-BatchNorm
    Dropout-ReLU layer with a dropout rate of 50%. All convolutions
    are 4x4 spatial filters applied with stride 2. Convolutions
    in the encoder, and in the discriminator, downsample
    by a factor of 2, whereas in the decoder they upsample by a
    factor of 2
    """
    def __init__(self, in_channels, out_channels, stride=2, padding=1, batch_norm=True, dropout=False, upsample=False, act='lrelu'):
        super(CBlock, self).__init__()
        self.use_dropout = dropout
        self.dropout = nn.Dropout(0.5)

        layers = []
        if upsample:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, stride, padding, bias=False))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, 4, stride, padding, bias=False, padding_mode='reflect'))

        if batch_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=False))

        if act == 'lrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif act == 'relu':
            layers.append(nn.ReLU())
        elif act == 'tanh':
            layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return self.dropout(x) if self.use_dropout else x

    def __str__(self):
        return f'{self.net}\ndropout={self.use_dropout}'