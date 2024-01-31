import torch
from vae_trainer import VAEConfig, VAETrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = VAEConfig.load_from_toml("configs/config_vae.toml")

trainer = VAETrainer(config=config)
trainer = trainer.train()
