### Python imports ###
import torch

### Local import ###
from trainer.scm_trainer import SCMConfig, SCMTrainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = SCMConfig.load_from_toml("configs/config_scm.toml")

trainer = SCMTrainer(config=config)
trainer = trainer.train()
