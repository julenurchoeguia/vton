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
