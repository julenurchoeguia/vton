import torch
from vton.training_utils.autoencodeur_trainer import AutoEncoderConfig, AutoEncoderTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = AutoEncoderConfig.load_from_toml("/home/laure/vton/configs/config_autoencodeur.toml")
trainer = AutoEncoderTrainer(config=config)
trainer = trainer.train()
