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
from refiners.fluxion.context import Contexts
from src.models.architecture_utils import Dropout
from torch.nn import PixelShuffle as _PixelShuffle
from typing import Iterable, cast





class PixelShuffle(_PixelShuffle, fl.Module):
    """Pixel Shuffle layer.
    """

    def __init__(self, upscale_factor: int):
        _PixelShuffle.__init__(self, upscale_factor=upscale_factor)

class SimpleGate(fl.Module):
    '''
    SimpleGate is a simplified version of the Squeeze-and-Excitation (SE) block.
    '''
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, fl.Module):
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
                            groups=1, use_bias=True),
                ),
                fl.Identity()
            ),
            ElementwiseMultiply()
        )

class NAFBlock(fl.Chain):
    
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        
        dw_channel = c * DW_Expand
        ffn_channel = FFN_Expand * c
        super().__init__(
            fl.Residual(
                fl.LayerNorm2d(c),
                fl.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, use_bias=True),
                fl.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,use_bias=True),
                SimpleGate(),
                SimplifiedChannelAttention(dw_channel),
                fl.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, use_bias=True),
                Dropout(probability=drop_out_rate),
                fl.Parallel(
                    fl.Parameter(c, 1, 1),
                    fl.Identity(),
                ),
                ElementwiseMultiply(),
            ),
            fl.Residual(
                fl.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, use_bias=True),
                SimpleGate(),
                fl.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, use_bias=True),
                Dropout(probability=drop_out_rate),
                fl.Parallel(
                    fl.Parameter(c, 1, 1),
                    fl.Identity(),
                ),
                ElementwiseMultiply(),
            )
        )
        
    #     self.conv1 = fl.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, use_bias=True)
    #     self.conv2 = fl.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
    #                            use_bias=True)
    #     self.conv3 = fl.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, use_bias=True)
        
    #     # Simplified Channel Attention
    #     self.sca = fl.Chain(
    #         AdaptiveAvgPool2d(1),
    #         fl.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
    #                   groups=1, use_bias=True),
    #     )

    #     # SimpleGate
    #     self.sg = SimpleGate()

        
    #     self.conv4 = fl.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, use_bias=True)
    #     self.conv5 = fl.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, use_bias=True)

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
                              use_bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              use_bias=True)

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
                    nn.Conv2d(chan, chan * 2, 1, use_bias=False),
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


class NAFBlockEnc(fl.Chain):

    def __init__(self, nb_blocks, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__(
            *[
                NAFBlock(c, DW_Expand, FFN_Expand, drop_out_rate) for _ in range(nb_blocks)
            ]
        )

class NAFBlockDec(fl.Chain):

    def __init__(self, nb_blocks, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__(
            *[
                NAFBlock(c, DW_Expand, FFN_Expand, drop_out_rate) for _ in range(nb_blocks)
            ]
        )

class DownBlocks(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        enc_blk_nums: Iterable[int] = [2, 2, 4, 8],
    ):
        self.in_channels_list = [in_channels * (2 ** i) for i in range(len(enc_blk_nums)+1)] 
        super().__init__(
            *[
                fl.Chain(
                    ## Encoding Blocks
                    NAFBlockEnc(enc_blk_num, self.in_channels_list[index]),
                    ## Downsampling Block
                    fl.Conv2d(in_channels=self.in_channels_list[index], out_channels=self.in_channels_list[index+1], kernel_size=2, stride=2)
                )
                for index, enc_blk_num in enumerate(enc_blk_nums)
            ]
        )

class UpBlocks(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        dec_blk_nums: Iterable[int] = [2, 2, 2, 2],
    ) -> None:

        self.in_channels_list = [in_channels // (2 ** i) for i in range(len(dec_blk_nums)+1)]
        super().__init__(
            *[
                fl.Chain(
                    ## Upsampling Block
                    fl.Chain(
                        fl.Conv2d(in_channels=self.in_channels_list[index], out_channels=self.in_channels_list[index+1]*4, kernel_size=1,use_bias=False),
                        PixelShuffle(2)
                    ),
                    ## Decoding Blocks
                    NAFBlockDec(dec_blk_num, self.in_channels_list[index+1])
                )
                for index, dec_blk_num in enumerate(dec_blk_nums)
            ]
        )

class MiddleBlock(fl.Chain):
    def __init__(
            self, 
            in_channels: int ,
            nb_middle_blocks: int = 1
        ) -> None:
        super().__init__(
            *[
                NAFBlock(in_channels) for _ in range(nb_middle_blocks)
            ]
        )

class EncoderAccumulator(fl.Passthrough):
    def __init__(self, n: int) -> None:
        self.n = n
        super().__init__(
            fl.SetContext(context="unet", key="naf_block_encoders", callback=self.update),
        )

    def update(self, naf_block_encoder: list[torch.Tensor | float], x: torch.Tensor) -> None:
        naf_block_encoder[self.n] = x

class DecoderAdditionner(fl.Chain):
    def __init__(self, n: int) -> None:
        self.n = n

        super().__init__(
            fl.Sum(
                fl.Identity(),
                fl.UseContext(context="unet", key="naf_block_encoders").compose(lambda naf_block_encoder: naf_block_encoder[self.n]),
            ),
        )

    def forward(self, *args: torch.Any) -> torch.Any:
        return super().forward(*args)

class NAFNet_UNet(fl.Chain):
    """NAFNet U-Net.

    This U-Net is based on the NAFNet architecture.
    """

    def __init__(
        self, in_channels=16, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]
    ) -> None:
        """Initialize the U-Net.

        Args:
            in_channels: The number of input channels.
            device: The PyTorch device to use for computation.
            dtype: The PyTorch dtype to use for computation.
        """
        self.in_channels = in_channels
        self.mid_channels = in_channels * (2 ** 4)
        super().__init__(
            DownBlocks(in_channels=self.in_channels, enc_blk_nums=enc_blk_nums),
            MiddleBlock(in_channels=self.mid_channels, nb_middle_blocks=middle_blk_num),
            UpBlocks(in_channels=self.mid_channels, dec_blk_nums=dec_blk_nums),
        )

        for n, block in enumerate(cast(Iterable[fl.Chain], self.DownBlocks)):
            # Inject an accumulator after each encoding block.
            block.insert(1, EncoderAccumulator(n))
        for n, block in enumerate(cast(Iterable[fl.Chain], self.UpBlocks)):
            # Inject an additioner before each decoding block.
            block.insert(1, DecoderAdditionner(3-n))

    def init_context(self) -> Contexts:
        return {
            "unet": {"naf_block_encoders": [0.0] * 4},
        }
    
class SCM(fl.Sum):
    def __init__(
        self, img_channel=6, width=16, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2], scm_weight=1.0
    ) -> None:
        super().__init__(
            fl.Chain(
                fl.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,use_bias=True),
                NAFNet_UNet(in_channels=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums),
                fl.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,use_bias=True),              
            ),
            fl.Chain(
                fl.Slicing(dim=1, start=3, end=6),
                fl.Multiply(scale = scm_weight),
            )
        )


if __name__ == "__main__":
    model = SCM()
    x = torch.randn(1, 6, 1024, 1024)
    y = model(x)
    print(y.shape)