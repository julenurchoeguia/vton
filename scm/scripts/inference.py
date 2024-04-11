### External imports ###
from torch import Tensor
from PIL import Image
import tqdm
import logging
import numpy as np
import os
import torch
import time
from dataclasses import dataclass
from pydantic import BaseModel

### Refiner imports ###
from refiners.fluxion.utils import tensor_to_image
from refiners.training_utils import ModelConfig
from refiners.training_utils.common import seed_everything

### Local imports ###
from scm.src.scm import SCM
from scm.src.dataset import SCMDataset

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
    

class SCMInferenceConfig(ModelConfig):
    scm: SCMModelConfig
    device : str
    path_output : str = "/var/hub/VITON-HD-results-scm"
    path_generated_images : str = "/var/hub/VITON-HD-results-ladi-vton"
    path_garment_images : str = "/var/hub/VITON-HD/test"


class SCMInference():

    def __init__(self, config: SCMInferenceConfig):
        self.config = config
        self.dataset = self.load_dataset(config.path_generated_images,config.path_garment_images,config.device,"test")
        self.model = self.scm(config.scm, config.device)
        self.compute_reference_loss()
        if not os.path.exists(config.path_output + "/inputs-output"):
            os.makedirs(config.path_output + "/inputs-output")
        if not os.path.exists(config.path_output + "/prediction"):
            os.makedirs(config.path_output + "/prediction")
        if not os.path.exists(config.path_output + "/vton"):
            os.makedirs(config.path_output + "/vton")
        if not os.path.exists(config.path_output + "/ground_truth"):
            os.makedirs(config.path_output + "/ground_truth")


    def load_dataset(self,path_generated_images,path_garment_images, device,  mode = "train")  -> SCMDataset:
        dataset =  SCMDataset(
            path_generated_images = path_generated_images,
            path_garment_images = path_garment_images,
            mode = mode,
            device = device,
        )
        return dataset
    
    def scm(self, scm_config: SCMModelConfig, device) -> SCM:
        model = SCM(
            img_channel=scm_config.img_channel,
            width=scm_config.width,
            middle_blk_num=scm_config.middle_blk_num,
            enc_blk_nums=scm_config.enc_blk_nums,
            dec_blk_nums=scm_config.dec_blk_nums,
            scm_weight=scm_config.scm_weight
        ).to(device=device)
        model.load_state_dict(torch.load(scm_config.checkpoint_path))
        model.eval()
        return model
    
    def compute_reference_loss(self):
        loss_val = 0
        for k, element in enumerate(self.dataset):
            loss = torch.norm(element["input_model_generate"].unsqueeze(0) - element["target"].unsqueeze(0))
            loss_val += loss

        self.reference_loss = loss_val / len(self.dataset)
        

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
                concat.save(f"{self.config.path_output}/inputs-output/{element['file_name']}")
                tensor_to_image(final_prediction).save(f"{self.config.path_output}/prediction/{element['file_name']}")
                tensor_to_image(model_generated).save(f"{self.config.path_output}/vton/{element['file_name']}")
                tensor_to_image(final_target).save(f"{self.config.path_output}/ground_truth/{element['file_name']}")


            logging.info(f"Reference loss: {self.reference_loss}")
            logging.info(f"Validation loss: {loss_val / len(self.dataset)}")





if __name__ == "__main__":
    

    print("Starting SCM Inference")
    print("Loading SCM model")
    scm_config = SCMModelConfig(
        checkpoint_path="models/vton_details_preservation/scm_spicy_pluton_24_20240403-151412.pt",
        img_channel=6,
        width=16,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )
    print("Loading SCM Inference Config")
    config = SCMInferenceConfig(
        scm=scm_config,
        device= "cuda:0" if torch.cuda.is_available() else "cpu",
        path_output = "/var/hub/results-scm-warped-vton-image",
        path_generated_images = "/var/hub/VITON-HD-warped",
        path_garment_images = "/var/hub/VITON-HD/test",
    )
    print("Instantiating SCM Inference")
    inference = SCMInference(config)
    print("Running SCM Inference")
    inference.run()
