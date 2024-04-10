### Python imports ###
import torch

### Local import ###
from trainer.autoencoder_trainer import AutoEncoderConfig, AutoEncoderTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = AutoEncoderConfig.load_from_toml("configs/config_autoencoder.toml")

trainer = AutoEncoderTrainer(config=config)
trainer = trainer.train()
