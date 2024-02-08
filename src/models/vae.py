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

class VAE(fl.Module):
    def __init__(self,latent_dim) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc1 = fl.Linear(8*128*96, latent_dim)
        self.fc2 = fl.Linear(8*128*96, latent_dim)
        self.fc3 = fl.Linear(latent_dim, 8*128*96)
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).to(device)
        z = mu + std * eps
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]
    
    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar
    