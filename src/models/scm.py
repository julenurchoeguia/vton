# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from refiners.fluxion import layers as fl
from src.models.architecture_utils import Dropout


class SimpleGate(fl.Module):
    '''
    SimpleGate is a simplified version of the Squeeze-and-Excitation (SE) block.
    '''
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, fl.WeightedModule):
    def __init__(self, output_size):
        super().__init__(output_size)

class ElementwiseMultiply(fl.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y


class SimplifiedChannelAttention(fl.Chain):

    def __init__(self,dw_channel) -> None:
        super().__init__(
            fl.Parallel(
                fl.Chain(
                    AdaptiveAvgPool2d(1),
                    fl.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                            groups=1, bias=True),
                ),
                fl.Identity()
            ),
            ElementwiseMultiply()
        )

class ParametersMultiply(fl.Module):
    def __init__(self, params : fl.Parameter) -> None:
        super().__init__()
        self.params = params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.params

class NAFBlock(fl.Chain):
    
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        
        dw_channel = c * DW_Expand
        ffn_channel = FFN_Expand * c
        super().__init__(
            fl.Sum(
                fl.Chain(
                    fl.LayerNorm2d(c),
                    fl.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                    fl.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,bias=True),
                    SimpleGate(),
                    SimplifiedChannelAttention(dw_channel),
                    fl.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                    Dropout(probability=drop_out_rate),
                    ParametersMultiply(fl.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)),
                ),
                fl.Identity(),
            ),
            fl.Sum(
                fl.Chain(
                    fl.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                    SimpleGate(),
                    fl.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                    Dropout(probability=drop_out_rate),
                    ParametersMultiply(fl.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)),
                ),
                fl.Identity(),
            )
        )
        
    #     self.conv1 = fl.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
    #     self.conv2 = fl.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
    #                            bias=True)
    #     self.conv3 = fl.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
    #     # Simplified Channel Attention
    #     self.sca = fl.Chain(
    #         AdaptiveAvgPool2d(1),
    #         fl.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
    #                   groups=1, bias=True),
    #     )

    #     # SimpleGate
    #     self.sg = SimpleGate()

        
    #     self.conv4 = fl.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
    #     self.conv5 = fl.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    #     self.norm1 = fl.LayerNorm2d(c)
    #     self.norm2 = fl.LayerNorm2d(c)

    #     self.dropout1 = Dropout(probability=drop_out_rate) 
    #     self.dropout2 = Dropout(probability=drop_out_rate) 

    #     self.beta = fl.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    #     self.gamma = fl.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    # def forward(self, inp):
    #     x = inp

    #     x = self.norm1(x)

    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = self.sg(x)
    #     x = x * self.sca(x)
    #     x = self.conv3(x)

    #     x = self.dropout1(x)

    #     y = inp + x * self.beta

    #     x = self.conv4(self.norm2(y))
    #     x = self.sg(x)
    #     x = self.conv5(x)

    #     x = self.dropout2(x)

    #     return y + x * self.gamma

class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNet_Combine(NAFNet):
    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        # weight of SCM
        weight = 1
        x = x + weight * inp[:, 3:6, :, :]

        return x[:, :, :H, :W]

