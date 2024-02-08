from torch import nn
import torch
from refiners.fluxion import layers as fl
from refiners.fluxion.adapters.adapter import Adapter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Resblock
class Resblock(fl.Sum):
    def __init__(self, in_channels: int=1, out_channels: int=1) -> None:
        super().__init__(
            fl.Chain(
                fl.Conv2d(in_channels, out_channels, 3, padding=1),
                fl.SiLU(),
                fl.Conv2d(out_channels, out_channels, 3, padding=1),
            ),
            fl.Conv2d(in_channels, out_channels, 1, padding=0),
        )

class MaxPool2d(nn.MaxPool2d, fl.Module):
    def __init__(self, factor: int = 2) -> None:
        super().__init__(factor)


class Flatten(fl.Module):
    def __init__(self, start_dim: int = 0, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(self.start_dim, self.end_dim)


class Unflatten(fl.Module):
    def __init__(self, dim: int, sizes : torch.Size) -> None:
        super().__init__()
        self.dim = dim
        self.sizes = sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(self.dim, self.sizes) 
    
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
            Flatten(),
        )

class Decoder(fl.Chain):
    def __init__(self, output_channels: int = 3):
        super().__init__(
            Unflatten(dim=0, sizes=torch.Size([1, 32, 128, 96])),
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

class NormalRepametrization(fl.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).to(device)
        z = mu + std * eps
        return z

class VAE(fl.Chain):
    def __init__(self,latent_dim) -> None:
        super().__init__(
            Encoder(),
            fl.Parallel(
                fl.Linear(32*128*96, latent_dim), 
                fl.Linear(32*128*96, latent_dim)
            ),
            fl.Parallel(
                fl.Chain(
                    NormalRepametrization(),
                    fl.Linear(latent_dim, 32*128*96),
                    Decoder(),
                ),
                fl.Distribute(
                    fl.Identity(),
                    fl.Identity(),
                )    
            )
        )
    



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