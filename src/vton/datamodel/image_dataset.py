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
        self.data = [self.load_image(file) for file in self.image_files]

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return f'ImageDataset(len={len(self)})'

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, key: int) -> Image.Image:
        return self.data[key]

    def load_image(self, file: str) -> Image.Image:
        image_path = os.path.join(self.path, file)
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image '{file}': {e}")
            return None
        
