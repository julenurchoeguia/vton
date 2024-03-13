### External imports ###
from functools import cached_property
from torch import Tensor
from PIL import Image
import numpy as np
import PIL
import torch
import time
from torch.nn.functional import mse_loss

### Refiner imports ###
from refiners.fluxion import layers as fl
from refiners.training_utils.trainer import (
    Trainer,
    seed_everything
)
from refiners.fluxion.utils import tensor_to_image
from refiners.training_utils.config import BaseConfig

### Local imports ###
from models.vae import VAE
from image_dataset import VAEBatch, ImageDataset
from dataclasses import dataclass

seed = 42
seed_everything(seed)

class VAEConfig(BaseConfig):
    path_dataset_train : str
    path_dataset_test : str
    path_dataset_val : str

class VAETrainer(Trainer[VAEConfig, VAEBatch]):

    @cached_property # pour ne pas loader autoencoder à chaque fois
    def vae(self):
        return VAE(latent_dim=32).to(device=self.device)

    def load_models(self) -> dict[str, fl.Module]:
        return {"vae": self.vae}

    def load_dataset(self):
        dataset_train =  ImageDataset(self.config.path_dataset_train)
        return dataset_train
    
    # def load_dataset_test(self):
    #     dataset_test = ImageDataset(self.config.path_dataset_test)
    #     return dataset_test

    def compute_loss(self, batch: VAEBatch) -> Tensor:
        image = batch.to(device=self.device)
        y,mu,logvar = self.vae(image)
        loss = (y-image).norm() + 0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1 - logvar)
        return loss
    
    def compute_evaluation(self) -> None:
        loss_val = 0
        reconstructed_images = []
        dataset_val = ImageDataset(self.config.path_dataset_val)

        for k, image in enumerate(dataset_val):
            image = image.unsqueeze(0)
            image = image.to(device=self.device)
            y,mu,logvar  = self.vae(image)
            loss = (y-image).norm() + 0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1 - logvar)
            loss_val += loss
        
            if k < 20:
                image_shape = image.shape
                concat = Image.new('RGB', (image_shape[-1]*2, image_shape[-2]))
                concat.paste(tensor_to_image(image.data), (0, 0))
                concat.paste(tensor_to_image(y.data), (image_shape[-1], 0))
                reconstructed_images.append(concat)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        torch.save(self.vae.state_dict(), str(self.config.checkpointing.save_folder) + "/"+ str(self.config.wandb.project) + "/" + str(self.config.wandb.name)  + f"/vae_{timestamp}.pt")
        images = [PIL.Image.fromarray(np.array(image)) for image in reconstructed_images]
        # print(len(images))
        i = 0
        for image in images:
            self.log({f"reconstructed_images_" + str(i): image})
            i += 1
        self.log({'val_loss': (loss_val / len(dataset_val)).item()})



