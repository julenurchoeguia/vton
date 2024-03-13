### External imports ###
import torch
### Refiner imports ###
from refiners.fluxion import layers as fl
### Local imports ###
from models.architecture_utils import (
    Encoder,
    Decoder
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    

