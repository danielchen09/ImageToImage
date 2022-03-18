import torch
import torch.nn as nn

from pix2pix.CBlock import CBlock


class Generator(nn.Module):
    """
    encoder:
        C64-C128-C256-C512-C512-C512-C512-C512
    decoder:
        CD512-CD512-CD512-C512-C256-C128-C64
        (including skip connection, it is:)
        CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128

    After the last layer in the decoder, a convolution is applied
    to map to the number of output channel, followed by a Tanh function

    Batch-Norm is not applied to the first C64 layer in the encoder

    All ReLUs in the encoder are leaky, with slope 0.2, while
    ReLUs in the decoder are not leaky
    """
    def __init__(self, in_channels=3, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [64, 128, 256, 512, 512, 512, 512, 512]
        self.encoder_blocks = [CBlock(in_channels, hidden_channels[0], batch_norm=False)]
        for i in range(1, len(hidden_channels) - 1):
            self.encoder_blocks.append(CBlock(hidden_channels[i - 1], hidden_channels[i]))
        self.encoder_blocks.append(CBlock(hidden_channels[-1], hidden_channels[-1], batch_norm=False, act='relu'))
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        self.decoder_blocks = []
        self.decoder_blocks.append(CBlock(hidden_channels[-1], hidden_channels[-1], upsample=True, act='relu', dropout=True))  # connecting bottleneck
        for i in range(len(hidden_channels) - 2, 0, -1):
            self.decoder_blocks.append(CBlock(hidden_channels[i] * 2, hidden_channels[i - 1], upsample=True, act='relu', dropout=True))
        self.decoder_blocks.append(CBlock(hidden_channels[0] * 2, in_channels, upsample=True, act='tanh', batch_norm=False))
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

    def forward(self, x):
        encoder_outputs = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_outputs.append(x)
        x = self.decoder_blocks[0](x)
        for i, decoder_block in enumerate(self.decoder_blocks[1:]):
            x = decoder_block(torch.cat([x, encoder_outputs[- i - 2]], 1))
        return x

    def __str__(self):
        s = ''
        s += str(self.initial_block) + '\n'
        for i, eb in enumerate(self.encoder_blocks):
            s += f'{i + 1}: {eb}\n'
        s += str(self.bottleneck) + '\n'
        for i, db in enumerate(self.decoder_blocks):
            s += f'{i + 1}: {db}\n'
        s += str(self.final_block)
        return s


if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256))
    model = Generator()
    preds = model(x)
    print(preds.shape)