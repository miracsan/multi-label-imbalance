#Copied from https://github.com/StefanDenn3r/Spatio-temporal-MS-Lesion-Segmentation/blob/master/model/FCDenseNet.py

from abc import abstractmethod

import torch.nn as nn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
        
#originally under /base/base_model.py 

class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
   

#Originally under model/FCDenseNet.py

class FCDenseNetEncoder(BaseModel):
    def __init__(
            self, in_channels=1, down_blocks=(4, 4, 4, 4, 4),
            bottleneck_layers=4, growth_rate=12, out_chans_first_conv=48
    ):
        super().__init__()
        self.down_blocks = down_blocks
        self.skip_connection_channel_counts = []

        self.add_module(
            'firstconv',
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_chans_first_conv,
                kernel_size=3, stride=1, padding=1, bias=True
            )
        )
        self.cur_channels_count = out_chans_first_conv

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(self.cur_channels_count, growth_rate, down_blocks[i])
            )
            self.cur_channels_count += (growth_rate * down_blocks[i])
            self.skip_connection_channel_counts.insert(
                0, self.cur_channels_count
            )
            self.transDownBlocks.append(TransitionDown(self.cur_channels_count))

        self.add_module(
            'bottleneck',
            Bottleneck(self.cur_channels_count, growth_rate, bottleneck_layers)
        )
        self.prev_block_channels = growth_rate * bottleneck_layers
        self.cur_channels_count += self.prev_block_channels

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        return out, skip_connections


class FCDenseNetDecoder(BaseModel):
    def __init__(
            self, prev_block_channels, skip_connection_channel_counts,
            growth_rate, n_classes, up_blocks, apply_softmax=True
    ):
        super().__init__()
        self.apply_softmax = apply_softmax
        self.up_blocks = up_blocks
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(self.up_blocks) - 1):
            self.transUpBlocks.append(
                TransitionUp(prev_block_channels, prev_block_channels)
            )
            cur_channels_count = prev_block_channels + \
                                 skip_connection_channel_counts[i]

            self.denseBlocksUp.append(
                DenseBlock(
                    cur_channels_count, growth_rate, self.up_blocks[i],
                    upsample=True
                )
            )
            prev_block_channels = growth_rate * self.up_blocks[i]
            cur_channels_count += prev_block_channels

        self.transUpBlocks.append(
            TransitionUp(prev_block_channels, prev_block_channels)
        )
        cur_channels_count = prev_block_channels + \
                             skip_connection_channel_counts[-1]
        self.denseBlocksUp.append(
            DenseBlock(
                cur_channels_count, growth_rate, self.up_blocks[-1],
                upsample=False
            )
        )
        cur_channels_count += growth_rate * self.up_blocks[-1]

        self.finalConv = nn.Conv2d(
            in_channels=cur_channels_count, out_channels=n_classes,
            kernel_size=1, stride=1, padding=0, bias=True
        )
        self.softmax = nn.Softmax2d()

    def forward(self, out, skip_connections):
        for i in range(len(self.up_blocks)):
            skip = skip_connections[-i - 1]
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        if self.apply_softmax:
            out = self.softmax(out)

        return out


class FCDenseNet(BaseModel):
    def __init__(
            self, in_channels=1, down_blocks=(4, 4, 4, 4, 4),
            up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4, growth_rate=12,
            out_chans_first_conv=48, n_classes=2, apply_softmax=True,
            encoder=None
    ):
        super().__init__()
        self.up_blocks = up_blocks
        self.encoder = encoder
        if not encoder:
            self.encoder = FCDenseNetEncoder(
                in_channels=in_channels, down_blocks=down_blocks,
                bottleneck_layers=bottleneck_layers, growth_rate=growth_rate,
                out_chans_first_conv=out_chans_first_conv
            )

        prev_block_channels = self.encoder.prev_block_channels
        skip_connection_channel_counts = self.encoder.skip_connection_channel_counts

        self.decoder = FCDenseNetDecoder(
            prev_block_channels, skip_connection_channel_counts, growth_rate,
            n_classes, up_blocks, apply_softmax
        )

    def forward(self, x, is_encoder_output=False):
        if is_encoder_output:
            out, skip_connections = x
        else:
            out, skip_connections = self.encoder(x)

        out = self.decoder(out, skip_connections)
        return out
             

#originally under /model/utils/layers.py

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()

        vectors = [torch.arange(0, s) for s in size]
        grid = torch.unsqueeze(torch.stack(torch.meshgrid(vectors)), dim=0).float()
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features, 1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=0, bias=True)

    def forward(self, x, skip_x):
        out = self.convTrans(x)
        out = center_crop(out, skip_x.size(2), skip_x.size(3))
        out = torch.cat([out, skip_x], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super().forward(x)


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]