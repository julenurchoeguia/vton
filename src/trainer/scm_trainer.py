### External imports ###
from torch import Tensor
from PIL import Image
from typing import  Literal
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


@dataclass
class SCMBatch:
    file_name : str
    model_mask : Tensor
    cloth_mask : Tensor
    model_generated : Tensor
    cloth : Tensor
    model_real: Tensor
    input_cloth: Tensor
    input_warped_cloth : Tensor
    input_model_generate: Tensor
    input_scm: Tensor
    target: Tensor
        
class SCMModelConfig(ModelConfig):
    img_channel:int = 6 
    width:int = 16
    middle_blk_num:int = 12
    enc_blk_nums: list[int] = [2, 2, 4, 8]
    dec_blk_nums: list[int] =[2, 2, 2, 2]
    scm_weight:float =1.0

class WandbConfig(BaseModel):
    mode: Literal["online", "offline", "disabled"] = "offline"
    project: str
    entity: str = "finegrain"
    name: str | None = None
    tags: list[str] = []
    group: str | None = None
    job_type: str | None = None
    notes: str | None = None

class SCMConfig(BaseConfig):
    scm: SCMModelConfig
    wandb : WandbConfig
    path_generated_images : str = "/var/hub/VITON-HD-results-ladi-vton"
    path_garment_images : str = "/var/hub/VITON-HD/test"
    saving_path : str = "/home/daniel/work/vton/models"


class SCMTrainer(Trainer[SCMConfig, SCMBatch]):

    def __init__(self, config: SCMConfig):
        self.train_dataset = self.load_dataset(config.path_generated_images,config.path_garment_images,config.training.device,"train")
        self.val_dataset = self.load_dataset(config.path_generated_images,config.path_garment_images,config.training.device,"val")
        super().__init__(config)
        self.load_wandb()
        self.compute_reference_loss()

    def log(self, data: dict[str, WandbLoggable]) -> None:
        self.wandb.log(data=data, step=self.clock.step)

    def load_wandb(self) -> None:
        init_config = {**self.config.wandb.model_dump(), "config": self.config.model_dump()}
        self.wandb = WandbLogger(init_config=init_config)

    def load_dataset(self,path_generated_images,path_garment_images, device,  mode = "train")  -> SCMDataset:
        dataset =  SCMDataset(
            path_generated_images = path_generated_images,
            path_garment_images = path_garment_images,
            mode = mode,
            device = device,
        )
        return dataset
    
    def get_item(self, index: int) -> SCMBatch:
        return SCMBatch(
            **self.train_dataset[index]
        )
    
    def collate_fn(self, batch: list[SCMBatch]) -> SCMBatch:
        return SCMBatch(
            file_name = [item.file_name for item in batch],
            model_mask = torch.stack([item.model_mask for item in batch]),
            cloth_mask = torch.stack([item.cloth_mask for item in batch]),
            model_generated = torch.stack([item.model_generated for item in batch]),
            cloth = torch.stack([item.cloth for item in batch]),
            model_real = torch.stack([item.model_real for item in batch]),
            input_cloth = torch.stack([item.input_cloth for item in batch]),
            input_warped_cloth = torch.stack([item.input_warped_cloth for item in batch]),
            input_model_generate = torch.stack([item.input_model_generate for item in batch]),
            input_scm = torch.stack([item.input_scm for item in batch]),
            target = torch.stack([item.target for item in batch]),
        )

    @property
    def dataset_length(self) -> int:
        return self.train_dataset.__len__()
    
    @register_model()
    def scm(self, config: SCMModelConfig) -> SCM:
        return SCM(
            img_channel=config.img_channel,
            width=config.width,
            middle_blk_num=config.middle_blk_num,
            enc_blk_nums=config.enc_blk_nums,
            dec_blk_nums=config.dec_blk_nums,
            scm_weight=config.scm_weight
        )

    def compute_loss(self, batch: SCMBatch) -> Tensor:
        target = batch.target
        input_scm = batch.input_scm
        prediction = self.scm(input_scm)
        loss = mse_loss(prediction,target)
        self.log({"learning_loss": loss, "iteration": self.clock.iteration, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        return loss
    
    def compute_reference_loss(self):
        loss_val = 0
        for k, element in enumerate(self.val_dataset):
            loss = mse_loss(element["input_model_generate"].unsqueeze(0),element["target"].unsqueeze(0))
            loss_val += loss

        self.reference_loss = loss_val / len(self.val_dataset)

    def save_model(self) -> None:
        training_name = self.config.wandb.name
        epoch = self.clock.epoch
        saving_path = self.config.saving_path
        project = self.config.wandb.project
        save_model_directory = saving_path + "/"+ project 
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        if not os.path.exists(save_model_directory):
            os.makedirs(save_model_directory)
        torch.save(self.scm.state_dict(), save_model_directory + f"/scm_{training_name}_{epoch}_{timestamp}.pt")


    def compute_evaluation(self) -> None:
        loss_val = 0
        log_images = []

        for k, element in enumerate(self.val_dataset):
            input_scm = element["input_scm"].unsqueeze(0)
            input_scm = input_scm.to(device=self.device)
            prediction = self.scm(input_scm)
            loss = mse_loss(prediction,element["target"].unsqueeze(0))
            loss_val += loss
        
            if k < 20:
                model_real = element["model_real"].unsqueeze(0)
                cloth = element["input_warped_cloth"].unsqueeze(0)
                model_generated = element["model_generated"].unsqueeze(0)
                image_shape = model_real.shape
                concat = Image.new('RGB', (image_shape[-1]*2, image_shape[-2]*2))
                concat.paste(tensor_to_image(cloth), (0, 0))
                concat.paste(tensor_to_image(prediction), (image_shape[-1], 0))
                concat.paste(tensor_to_image(model_generated), (0, image_shape[-2]))
                concat.paste(tensor_to_image(model_real), (image_shape[-1], image_shape[-2]))
                log_images.append(concat)
        
        images = [Image.fromarray(np.array(image)) for image in log_images]
        i = 0
        for image in images:
            self.log({f"reconstructed_images_" + str(i): image})
            i += 1

        self.log({'val_loss': (loss_val / len(self.val_dataset)).item(), "epoch": self.clock.epoch , "reference_loss": self.reference_loss.item()})
        self.save_model()





if __name__ == "__main__":
    

    training = TrainingConfig(
        duration=TimeValue({ "number": 15,"unit": "epoch"}),
        device="cuda" if torch.cuda.is_available() else "cpu",
        evaluation_interval= TimeValue({ "number": 1,"unit": "epoch"}),
        batch_size=4,
    )

    optimizer = OptimizerConfig(
        optimizer=Optimizers.Adam,
        learning_rate=1e-4,
    )

    lr_scheduler = LRSchedulerConfig(
        type=LRSchedulerType.COSINE_ANNEALING_LR,
    )

    wandb = WandbConfig(
        mode="online",
        project="vton_details_preservation",
        entity="finegrain-cs",
        name="sky_sikar",
    )

    scm_config = SCMModelConfig(
        img_channel=6, width=32, middle_blk_num=8, enc_blk_nums=[2, 4, 8], dec_blk_nums=[2, 2, 2]
    )

    config = SCMConfig(
        training=training,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scm=scm_config,
        wandb= wandb,
    )
    trainer = SCMTrainer(config=config)
    # trainer.train()
