## Imports
from torch import nn as nn
from refiners.fluxion import layers as fl
from refiners.fluxion.adapters.adapter import Adapter


class Dropout(nn.Dropout, fl.Module):
    def __init__(self, probability: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p=probability, inplace=inplace)


class DropoutAdapter(Adapter[fl.SiLU], fl.Chain):
    def __init__(self, target: fl.SiLU, dropout: float = 0.5):
        self.dropout = dropout
        with self.setup_adapter(target):
            super().__init__(target)

    def inject(self, parent: fl.Chain | None = None):
        self.append(Dropout(self.dropout))
        super().inject(parent)

    def eject(self):
        dropout = self.ensure_find(Dropout)
        #  ensure_find : meme chose que find mais ne peut pas renovyer None
        self.remove(dropout)
        super().eject()


class Resblock(fl.Sum):
    def __init__(self, in_channels: int=1, out_channels: int=1) -> None:
        super().__init__(
            fl.Chain(
                fl.Conv2d(in_channels, out_channels, 3, padding=1),
                fl.SiLU(),
                fl.Conv2d(out_channels, out_channels, 3, padding=1),
            ),
            fl.Conv2d(in_channels, out_channels, 3, padding=1),
        )


class MaxPool2d(nn.MaxPool2d, fl.Module):
    def __init__(self, factor: int = 2) -> None:
        super().__init__(factor)


class Encoder(fl.Chain):
    def __init__(self, input_channels: int = 3):
        super().__init__(
            fl.Conv2d(input_channels, 32, 1, padding=0),
            Resblock(32, 64),
            MaxPool2d(2),
            Resblock(64, 128),
            MaxPool2d(2),
            Resblock(128, 256),
            MaxPool2d(2),
            fl.Conv2d(256, 32, 1, padding=0),
        )

class Decoder(fl.Chain):
    def __init__(self, output_channels: int = 3):
        super().__init__(
            fl.Conv2d(32, 256, 1, padding=0),
            Resblock(256, 128),
            fl.Upsample(channels=128,upsample_factor=2),
            Resblock(128, 64),
            fl.Upsample(channels=64,upsample_factor=2),
            Resblock(64, 32),
            fl.Upsample(channels = 32,upsample_factor=2),
            Resblock(32, output_channels),
            fl.Conv2d(output_channels, output_channels, 1, padding=0),
        )


class AutoEncoder(fl.Chain):
    def __init__(self) -> None:
        super().__init__(
            Encoder(),
            Decoder(),
        )


def load_dropout(chain : fl.Chain, dropout : float = 0.5):
    for silu, parent in chain.walk(fl.SiLU):
        DropoutAdapter(silu, dropout).inject(parent)


if __name__ == "__main__":
    import torch 
    # Test the model instantiation   
    autoencoder = AutoEncoder()
    # Test the dropout injection
    load_dropout(autoencoder, 0.1)
    print(autoencoder.__repr__())
    # Test forward pass
    x = torch.randn(1, 3, 512, 512)
    print(autoencoder(x).shape)

    print("All okay!")
    