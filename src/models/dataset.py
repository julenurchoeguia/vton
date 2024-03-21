### External imports ###
from PIL import Image
import os

### Refiner imports ###
from refiners.fluxion.utils import image_to_tensor, tensor_to_image
import torchvision.transforms as transforms
import torch

### Local imports ###
import sys
sys.path.append('../') # define relative path for local imports
from image_dataset import Dataset


class SCMDataset(Dataset):
    def __init__(self, path_person, path_cloth) -> None:
        self.path_person = path_person
        self.path_cloth = path_cloth
        self.image_person = [f for f in os.listdir(path_person) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.image_cloth = [f for f in os.listdir(path_cloth) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self) -> int:
        return len(self.image_person)
    
    def __str__(self) -> str:
        return f'SCMDataset(len={len(self)})'

    def __repr__(self) -> str:
        return str(self)
    
    def __getitem__(self, key: int) -> torch.Tensor:
        return self.concat_images(self.image_person[key], self.image_cloth[key])
        pass

    def load_image(self, file: str) -> Image.Image:
        image_path = os.path.join(self.path, file)
        try:
            image = Image.open(image_path).convert("RGB")
            image = image_to_tensor(image).squeeze(0)
            return image
        except Exception as e:
            print(f"Error loading image '{file}': {e}")
            return None
        
    def concat_images(self, image_person, image_cloth)-> torch.Tensor:
        transform = transforms.Compose([
        transforms.Resize((512, 384))  # Resize to 512x384
        ])
        resized_cloth = transform(image_cloth)
        #concatenate the two images for the NAFFNet_Combine model
        concat_tensor = torch.cat((image_person, resized_cloth),0)
        return concat_tensor

     
