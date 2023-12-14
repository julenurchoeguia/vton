import torch
from torch import nn
from PIL import Image
import os
from torch.utils.data import Dataset
from refiners.fluxion import layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.context import Contexts
from refiners.training_utils.trainer import Trainer
from typing import TypeVar
from refiners.training_utils.config import BaseConfig


#%% UNET
class DownBlock(fl.Chain):
    def __init__(self) -> None:
        super().__init__(
            fl.Conv2d(32, 64, 3, padding=1),
            fl.Lambda(self.append_residual),
            fl.Downsample(64, 2),
            fl.Conv2d(64, 128, 3, padding=1),
            fl.Lambda(self.append_residual),
            fl.Downsample(128, 2),
            fl.Conv2d(128, 256, 3, padding=1),
            fl.Lambda(self.append_residual),
        )

    def append_residual(self, x: torch.Tensor):
        self.use_context("unet")["residuals"].append(x)
        return x


class MiddleBlock(fl.Residual):
    def __init__(self) -> None:
        super().__init__(
            fl.Conv2d(256, 256, 3, padding=1),
        )


class UpBlock(fl.Chain):
    def __init__(self) -> None:
        super().__init__(
            fl.Concatenate(
                fl.Identity(),
                fl.UseContext(context="unet", key="residuals").compose(func=lambda residuals: residuals.pop(-1)),
                dim=1,
            ),
            fl.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1),
            fl.Upsample(channels=128),
            fl.Concatenate(
                fl.Identity(),
                fl.UseContext(context="unet", key="residuals").compose(func=lambda residuals: residuals.pop(-1)),
                dim=1,
            ),
            fl.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            fl.Upsample(channels=64),
            fl.Concatenate(
                fl.Identity(),
                fl.UseContext(context="unet", key="residuals").compose(func=lambda residuals: residuals.pop(-1)),
                dim=1,
            ),
            fl.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
        )


class UNet(fl.Chain):
    def __init__(self) -> None:
        super().__init__(
            fl.Conv2d(in_channels=3, out_channels=32, kernel_size=1),
            DownBlock(),
            MiddleBlock(),
            UpBlock(),
            fl.Conv2d(in_channels=32, out_channels=3, kernel_size=1),
        )

    def init_context(self) -> Contexts:
        return {"sampling": {"shapes": []}, "unet": {"residuals": []}}


#%% Resblock
class Resblock(fl.Sum):
    def __init__(self, in_channels: int=1, out_channels: int=1) -> None:
        super().__init__(
            fl.Chain(
                fl.Conv2d(in_channels, out_channels, 3, padding=1),
                fl.SiLU(),
                fl.Conv2d(out_channels, out_channels, 3, padding=1),
            ),
            fl.Conv2d(in_channels, out_channels, 3, padding=1),
        )

#%%
class Dropout(nn.Dropout, fl.Module):
    def __init__(self, probability: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p=probability, inplace=inplace)

class MaxPool2d(nn.MaxPool2d, fl.Module):
    def __init__(self, factor: int = 2) -> None:
        super().__init__(factor)

class DropoutAdapter(Adapter[fl.SiLU], fl.Chain):
    def __init__(self, target: fl.SiLU, dropout: float = 0.5):
        self.dropout = dropout
        with self.setup_adapter(target):
            super().__init__(target)

    def inject(self, parent: fl.Chain | None = None):
        self.append(Dropout(self.dropout))
        super().inject(parent)

    def eject(self):
        dropout = self.ensure_find(Dropout)
        #  ensure_find : meme chose que find mais ne peut pas renovyer None
        self.remove(dropout)
        super().eject()
        

#%% Autoencoder

class Encoder(fl.Chain):
    def __init__(self, input_channels: int = 3):
        super().__init__(
            fl.Conv2d(input_channels, 32, 1, padding=0),
            Resblock(32, 64),
            MaxPool2d(2),
            Resblock(64, 128),
            MaxPool2d(2),
            Resblock(128, 256),
            MaxPool2d(2),
<<<<<<<< HEAD:laure_bis/lib_refiner_autoencoder.py
            fl.Conv2d(256, 8, 1, padding=0),
========
            fl.Conv2d(256, 32, 1, padding=0),
>>>>>>>> daniel:notebooks/autoencodeur.py
        )

class Decoder(fl.Chain):
    def __init__(self, output_channels: int = 3):
        super().__init__(
<<<<<<<< HEAD:laure_bis/lib_refiner_autoencoder.py
            fl.Conv2d(8, 256, 1, padding=0),
========
            fl.Conv2d(32, 256, 1, padding=0),
>>>>>>>> daniel:notebooks/autoencodeur.py
            Resblock(256, 128),
            fl.Upsample(channels=128,upsample_factor=2),
            Resblock(128, 64),
            fl.Upsample(channels=64,upsample_factor=2),
            Resblock(64, 32),
            fl.Upsample(channels = 32,upsample_factor=2),
            Resblock(32, output_channels),
            fl.Conv2d(output_channels, output_channels, 1, padding=0),
        )

class AutoEncoder(fl.Chain):
    def __init__(self) -> None:
        super().__init__(
            Encoder(),
            Decoder(),
        )

#%% Fonctions

def load_dropout(chain : fl.Chain, dropout : float = 0.05):
    for silu, parent in chain.walk(fl.SiLU):
        DropoutAdapter(silu, dropout).inject(parent)


class Dataset:
    def __init__(self, path) -> None:
        self.data = list(range(100))
        self.path = path

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return f'Dataset(len={len(self)})'

    def __repr__(self) -> str:
        return str(self)
    
    def __getitem__(self, key : str|int) -> int:
        match key:
            case key if isinstance(key, str):
                raise ValueError('Dataset does not take string as index.')
            case _:
                return self.data[key]
            
class ImageDataset(Dataset):
    def __init__(self, path) -> None:
        self.path = path
        self.image_files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        # self.data = [self.load_image(file) for file in self.image_files]

    def __len__(self) -> int:
        return len(self.image_files)

    def __str__(self) -> str:
        return f'ImageDataset(len={len(self)})'

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, key: int) -> Image.Image:
        # return self.data[key]
        return self.load_image(self.image_files[key])

    def load_image(self, file: str) -> Image.Image:
        image_path = os.path.join(self.path, file)
        try:
<<<<<<<< HEAD:laure_bis/lib_refiner_autoencoder.py
            # with Image.open(image_path) as image:
            #     return image
            image = Image.open(image_path).convert('RGB')
========
            image = Image.open(image_path).convert("RGB")
>>>>>>>> daniel:notebooks/autoencodeur.py
            return image
        except Exception as e:
            print(f"Error loading image '{file}': {e}")
            return None
        


ConfigType = TypeVar("ConfigType", bound=BaseConfig)

class AutoEncoderTrainer(Trainer[ConfigType]):

    def load_models(self) -> dict[str, fl.Module]:
        return {"autoencodeur": self.autoencodeur}

    def load_dataset(self) -> Dataset[TextEmbeddingLatentsBatch]:
        return ImageDataset(self.config.dataset.path)


    def sample_timestep(self) -> Tensor:
        random_step = random.randint(a=self.config.latent_diffusion.min_step, b=self.config.latent_diffusion.max_step)
        self.current_step = random_step
        return self.ddpm_scheduler.timesteps[random_step].unsqueeze(dim=0)

    def sample_noise(self, size: tuple[int, ...], dtype: DType | None = None) -> Tensor:
        return sample_noise(
            size=size, offset_noise=self.config.latent_diffusion.offset_noise, device=self.device, dtype=dtype
        )

    def compute_loss(self, batch: TextEmbeddingLatentsBatch) -> Tensor:
        clip_text_embedding, latents = batch.text_embeddings, batch.latents
        timestep = self.sample_timestep()
        noise = self.sample_noise(size=latents.shape, dtype=latents.dtype)
        noisy_latents = self.ddpm_scheduler.add_noise(x=latents, noise=noise, step=self.current_step)
        self.unet.set_timestep(timestep=timestep)
        self.unet.set_clip_text_embedding(clip_text_embedding=clip_text_embedding)
        prediction = self.unet(noisy_latents)
        loss = mse_loss(input=prediction, target=noise)
        return loss

    def compute_evaluation(self) -> None:
        sd = StableDiffusion_1(
            unet=self.unet,
            lda=self.lda,
            clip_text_encoder=self.text_encoder,
            scheduler=DPMSolver(num_inference_steps=self.config.test_diffusion.num_inference_steps),
            device=self.device,
        )
        prompts = self.config.test_diffusion.prompts
        num_images_per_prompt = self.config.test_diffusion.num_images_per_prompt
        if self.config.test_diffusion.use_short_prompts:
            prompts = [prompt.split(sep=",")[0] for prompt in prompts]
        images: dict[str, WandbLoggable] = {}
        for prompt in prompts:
            canvas_image: Image.Image = Image.new(mode="RGB", size=(512, 512 * num_images_per_prompt))
            for i in range(num_images_per_prompt):
                logger.info(f"Generating image {i+1}/{num_images_per_prompt} for prompt: {prompt}")
                x = randn(1, 4, 64, 64, device=self.device)
                clip_text_embedding = sd.compute_clip_text_embedding(text=prompt).to(device=self.device)
                for step in sd.steps:
                    x = sd(
                        x,
                        step=step,
                        clip_text_embedding=clip_text_embedding,
                    )
                canvas_image.paste(sd.lda.decode_latents(x=x), box=(0, 512 * i))
            images[prompt] = canvas_image
        self.log(data=images)
