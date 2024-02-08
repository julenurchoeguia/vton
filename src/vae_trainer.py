from dataclasses import dataclass
from functools import cached_property
from torch import Tensor
from PIL import Image
import numpy as np
import PIL
import torch
import time

from torch.nn.functional import mse_loss

from refiners.fluxion import layers as fl
from refiners.training_utils.trainer import (
    Trainer,
    seed_everything
)
from refiners.fluxion.utils import tensor_to_image
from refiners.training_utils.config import BaseConfig
from vae import VAE
from image_dataset import VAEBatch, ImageDataset
import math

seed = 42
seed_everything(seed)

class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = 'none'
            self.baseline = 0.0

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == 'linear':
            y = (self.current_step / self.total_steps)
        elif self.shape == 'cosine':
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == 'logistic':
            exponent = ((self.total_steps / 2) - self.current_step)
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == 'none':
            y = 1.0
        else:
            raise ValueError('Invalid shape for annealing function. Must be linear, cosine, or logistic.')
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError('Cyclical_setter method requires boolean argument (True/False)')
        else:
            self.cyclical = value
        return

class VAEConfig(BaseConfig):
    path_dataset_train : str
    path_dataset_test : str
    path_dataset_val : str

class VAETrainer(Trainer[VAEConfig, VAEBatch]):

    def __init__(self, config: VAEConfig):
        super().__init__(config)
        self.annealer = Annealer(total_steps=self.clock.num_steps, shape='linear')

    @cached_property # pour ne pas loader autoencoder à chaque fois
    def vae(self):
        return VAE(latent_dim=2**8).to(device=self.device)

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
        y,normal_dist = self.vae(image)
        mu, logvar = normal_dist
        mse = 1000 * mse_loss(y,image)
        kld = - 0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        loss = mse + kld
        return loss
    
    def compute_evaluation(self) -> None:
        loss_val = 0
        reconstructed_images = []
        dataset_val = ImageDataset(self.config.path_dataset_val)

        for k, image in enumerate(dataset_val):
            image = image.unsqueeze(0)
            image = image.to(device=self.device)
            y,normal_dist = self.vae(image)
            mu, logvar = normal_dist
            mse = 1000 * mse_loss(y,image)
            kld = - 0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
            loss = mse + kld
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



