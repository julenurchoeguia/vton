### External imports ###
from torch import Tensor
from PIL import Image
import tqdm
import logging
import numpy as np
import os
import torch
import time
from torch.nn.functional import mse_loss
from dataclasses import dataclass
from pydantic import BaseModel

### Refiner imports ###
from refiners.training_utils.trainer import Trainer
from refiners.fluxion.utils import tensor_to_image
from refiners.training_utils import ModelConfig, register_model, BaseConfig, TrainingConfig, OptimizerConfig, LRSchedulerConfig, Optimizers, LRSchedulerType
from refiners.training_utils.common import seed_everything, TimeValue
from refiners.training_utils.wandb import WandbLogger,WandbLoggable

### Local imports ###
from src.models.scm import SCM
from src.datasets.dataset import SCMDataset

seed = 42
seed_everything(seed)

        
class SCMModelConfig(ModelConfig):
    checkpoint_path: str
    img_channel:int = 6 
    width:int = 16
    middle_blk_num:int = 12
    enc_blk_nums: list[int] = [2, 2, 4, 8]
    dec_blk_nums: list[int] =[2, 2, 2, 2]
    scm_weight:float =1.0
    

class SCMInferenceConfig(BaseConfig):
    scm: SCMModelConfig
    device : str
    path_saved_images : str = "/var/hub/VITON-HD-results-scm"
    path_generated_images : str = "/var/hub/VITON-HD-results-ladi-vton"
    path_garment_images : str = "/var/hub/VITON-HD/test"
    saving_path : str = "/home/daniel/work/vton/models"


class SCMInference():

    def __init__(self, config: SCMInferenceConfig):
        self.dataset = self.load_dataset(config.path_generated_images,config.path_garment_images,config.device,"test")
        self.model = self.scm(config.scm)
        self.compute_reference_loss()
        if not os.path.exists(config.saving_path + "/inputs-output"):
            os.makedirs(config.saving_path + "/inputs-output")
        if not os.path.exists(config.saving_path + "/prediction"):
            os.makedirs(config.saving_path + "/prediction")


    def load_dataset(self,path_generated_images,path_garment_images, device,  mode = "train")  -> SCMDataset:
        dataset =  SCMDataset(
            path_generated_images = path_generated_images,
            path_garment_images = path_garment_images,
            mode = mode,
            device = device,
        )
        return dataset
    
    def scm(self, config: SCMModelConfig) -> SCM:
        model = SCM(
            img_channel=config.img_channel,
            width=config.width,
            middle_blk_num=config.middle_blk_num,
            enc_blk_nums=config.enc_blk_nums,
            dec_blk_nums=config.dec_blk_nums,
            scm_weight=config.scm_weight
        ).to(device=config.device)
        model.load_state_dict(torch.load(config.checkpoint_path))
        model.eval()
        return model
    
    def compute_reference_loss(self):
        loss_val = 0
        for k, element in enumerate(self.dataset):
            loss = torch.norm(element["input_model_generate"].unsqueeze(0) - element["target"].unsqueeze(0))
            loss_val += loss

        self.reference_loss = loss_val / len(self.val_dataset)
        

    def run(self) -> None:
        loss_val = 0

        with torch.no_grad():

            for  element in tqdm.tqdm(self.dataset):
                input_scm = element["input_scm"].unsqueeze(0)
                prediction = self.model(input_scm)
                loss = torch.norm(prediction - element["target"].unsqueeze(0))
                loss_val += loss

                final_target = element["model_real"].unsqueeze(0)
                context_input = element["input_warped_cloth"].unsqueeze(0)
                model_generated = element["model_generated"].unsqueeze(0)
                mask = element["model_mask"].unsqueeze(0)
                final_prediction = prediction * mask + model_generated * (1 - mask)
                image_shape = final_target.shape
                concat = Image.new('RGB', (image_shape[-1]*2, image_shape[-2]*2))
                concat.paste(tensor_to_image(context_input), (0, 0))
                concat.paste(tensor_to_image(final_prediction), (image_shape[-1], 0))
                concat.paste(tensor_to_image(model_generated), (0, image_shape[-2]))
                concat.paste(tensor_to_image(final_target), (image_shape[-1], image_shape[-2]))
                concat.save(f"{self.config.saving_path}/inputs-output/{element['file_name']}")
                tensor_to_image(final_prediction).save(f"{self.config.saving_path}/prediction/{element['file_name']}")


            logging.info(f"Reference loss: {self.reference_loss}")
            logging.info(f"Validation loss: {loss_val / len(self.dataset)}")





if __name__ == "__main__":
    


    scm_config = SCMModelConfig(
        checkpoint_path="models/vton_details_preservation/scm_spicy_jupyter_29_20240402-182613.pt",
        img_channel=6,
        width=16,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2,2, 2, 2],
    )

    config = SCMInferenceConfig(
        scm=scm_config,
        device= "cuda:0" if torch.cuda.is_available() else "cpu",
        path_saved_images = "/var/hub/VITON-HD-results-scm",
        path_generated_images = "/var/hub/VITON-HD-results-ladi-vton",
        path_garment_images = "/var/hub/VITON-HD/test",
    )
    inference = SCMInference(config)
    inference.run()
