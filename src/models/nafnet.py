### Python import ###
import torch
import torch.nn as nn
import torch.nn.functional as F

### Refiner import ###
from refiners.fluxion import layers as fl

### Local import ###
from models.architecture_utils import (
    LayerNorm2d, # function to rewrite using refiners
    Local_Base, # function to rewrite using refiners
    Dropout,
    AdaptiveAvgPool2d
)


"""
TODO :
    - Finish to rewrite the NAFBlock class using refiners (cf test_nafnet.ipynb for rewriting in progress)
    - Rewrite the NAFNet class using refiners

https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py

"""

class SimpleGate(fl.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        # the .chunk() method splits a tensor into a specified number of chunks along a given dimension
        return x1 * x2

class SimplifiedChannelAttention(fl.Module):
    def __init__(self, c, DW_Expand = 2) -> None:
        super().__init__(
            AdaptiveAvgPool2d(1),
            fl.Conv2d(in_channels=(c*DW_Expand)//2, out_channels=(c*DW_Expand)//2, kernel_size=1, padding=0, stride=1, groups=1, use_bias=True),
        )

class CustomConditionedDropout(fl.Module):
    def __init__(self, drop_out_rate) -> None:
        super().__init__()
        self.drop_out_rate = drop_out_rate
    def forward(self, x):
        if self.drop_out_rate > 0.:
            x = Dropout(x)
        else :
            x = fl.Identity()
        return x

class MultiplyLayers(fl.Module):
    def forward(self, x, layer):
        new_x = x * layer(x)
        return new_x

class NAFBlock(fl.Chain):
    def __init__(self, c, DW_Expand = 2, FFN_Expand = 2, drop_out_rate = 0.) -> None:
        super().__init__(
            # TODO : x = inp
            # x = self.norm1(x)
            # LayerNorm2d(c),

            fl.Conv2d(in_channels=c, out_channels=c*DW_Expand, kernel_size=1, padding=0, stride=1, groups=1, use_bias=True),
            fl.Conv2d(in_channels=c*DW_Expand, out_channels=c*DW_Expand, kernel_size=3, padding=1, stride=1, groups=c*DW_Expand, use_bias=True),
            SimpleGate(),

            # x = x * self.sca(x) (sca : simplified channel attention)
            # try with fl.Matmul() ? cf chain.py in refiners repo
            # MultiplyLayers(SimplifiedChannelAttention(c, DW_Expand)) ??

            fl.Conv2d(in_channels=(c*DW_Expand)//2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, use_bias=True),
            Dropout(drop_out_rate) if drop_out_rate > 0. else fl.Identity(),

            # TODO :  y = inp + x * self.beta
            #         x = self.conv4(self.norm2(y))

            SimpleGate(),
            fl.Conv2d(in_channels=FFN_Expand*c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, use_bias=True),
            Dropout(drop_out_rate) if drop_out_rate > 0. else fl.Identity(),

            # TODO : return y + x * self.gamma
        )

class NAFBlock_debase(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
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

