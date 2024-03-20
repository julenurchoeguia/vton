### External imports ###
from functools import cached_property
from torch import Tensor
from PIL import Image
import numpy as np
import PIL
import torch
import time
from torch.nn.functional import mse_loss
import torch.nn as nn

### Refiner imports ###
from refiners.fluxion import layers as fl
from refiners.training_utils.trainer import (
    Trainer,
    seed_everything
)
from refiners.fluxion.utils import tensor_to_image
from refiners.training_utils.config import BaseConfig

### Local imports ###
from models.scm import NAFNet_Combine
from image_dataset import SCMBatch, ImageDataset
from models.dataset import load_image

seed = 42
seed_everything(seed)

class SCMConfig(BaseConfig):
    path_dataset_train : str
    path_dataset_test : str
    path_dataset_val : str

class SCMTrainer(Trainer[SCMConfig, SCMBatch]):

    @cached_property # pour ne pas loader NAFNet_Combine à chaque fois
    def scm(self):
        return NAFNet_Combine().to(device=self.device)

    def load_models(self) -> dict[str, nn.Module]:
        return {"scm": self.scm}

    def load_dataset(self):
        dataset_train =  ImageDataset(self.config.path_dataset_train)
        return dataset_train
    
    # def load_dataset_test(self):
    #     dataset_test = ImageDataset(self.config.path_dataset_test)
    #     return dataset_test

    def compute_loss(self, batch: SCMBatch) -> Tensor:
        image = batch.to(device=self.device)
        prediction = self.scm(image)
        loss = mse_loss(prediction,image)
        return loss
    
    def compute_evaluation(self) -> None:
        loss_val = 0
        reconstructed_images = []

        dataset_val = ImageDataset(self.config.path_dataset_val)

        for k, image in enumerate(dataset_val):
            image = image.unsqueeze(0)
            image = image.to(device=self.device)
            prediction = self.scm(image)
            loss = mse_loss(prediction,image)
            loss_val += loss
        
            if k < 5:
                image_shape = image.shape
                concat = Image.new('RGB', (image_shape[-1]*2, image_shape[-2]))
                concat.paste(tensor_to_image(image.data), (0, 0))
                concat.paste(tensor_to_image(prediction.data), (image_shape[-1], 0))
                reconstructed_images.append(concat)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        torch.save(self.scm.state_dict(), str(self.config.checkpointing.save_folder) + "/"+ str(self.config.wandb.project) + "/" + str(self.config.wandb.name)  + f"/ae_{timestamp}.pt")
        
        images = [PIL.Image.fromarray(np.array(image)) for image in reconstructed_images]
        # print(len(images))
        i = 0
        for image in images:
            self.log({f"reconstructed_images_" + str(i): image})
            i += 1
        self.log({'val_loss': (loss_val / len(dataset_val)).item()})

