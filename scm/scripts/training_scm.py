### Python imports ###
import torch

### Local import ###
from scm.src.scm_trainer import SCMConfig, SCMTrainer


config = SCMConfig.load_from_toml("scm/configs/config_scm.toml")
config.training.device = "cuda" if torch.cuda.is_available() else "cpu"


trainer = SCMTrainer(config=config)
trainer = trainer.train()
