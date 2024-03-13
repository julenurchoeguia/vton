### Refiner imports ###
from refiners.fluxion import layers as fl
### Local imports ###
from models.architecture_utils import (
    Encoder,
    Decoder,
    DropoutAdapter
)

class AutoEncoder(fl.Chain):
    def __init__(self) -> None:
        super().__init__(
            Encoder(),
            Decoder(),
        )

#%% Fonctions

def load_dropout(chain : fl.Chain, dropout : float = 0.5):
    for silu, parent in chain.walk(fl.SiLU):
        DropoutAdapter(silu, dropout).inject(parent)


if __name__ == "__main__":
    autoencoder = AutoEncoder().to("cuda")
    print(autoencoder.device)