### Python imports ###
import torch

### Local import ###
from src.trainer.scm_trainer import SCMConfig, SCMTrainer


config = SCMConfig.load_from_toml("configs/config_scm.toml")
config.training.device = "cuda" if torch.cuda.is_available() else "cpu"


trainer = SCMTrainer(config=config)
trainer = trainer.train()
