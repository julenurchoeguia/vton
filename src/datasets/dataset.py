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




class SCMDataset:
    def __init__(self, path_generated_images, path_garment_images, mode, device, high = 1024 , width = 768) -> None:
        self.path_generated_images = path_generated_images
        self.path_garment_images = path_garment_images
        self.high = high
        self.width = width
        self.mode = mode
        self.device = device
        path_to_get_files_list = self.path_generated_images + "/" + self.mode + "_files.txt"
        with open(path_to_get_files_list, 'r') as file:
            self.images_files = file.readlines()
        self.images_files = [image_file.strip() for image_file in self.images_files]
        self.generated_images_files = [ path_generated_images + "/paired/upper_body/"+ image_file for image_file in self.images_files]
        self.garment_images_files = [ path_garment_images + "/cloth/" + image_file for image_file in self.images_files]
        self.original_images_files = [ path_garment_images + "/image/" + image_file for image_file in self.images_files]
        self.agnostic_mask_files = [ path_garment_images + "/agnostic-mask/" + image_file.replace(".jpg","_mask.png") for image_file in self.images_files]
        self.cloth_mask_files = [ path_garment_images + "/cloth-mask/" + image_file for image_file in self.images_files]
        self.resize = transforms.Resize((self.high, self.width))

    def __len__(self) -> int:
        return len(self.images_files)
    
    def __str__(self) -> str:
        return f'SCMDataset(len={len(self)})'

    def __repr__(self) -> str:
        return str(self)
    
    def __getitem__(self, key: int) -> torch.Tensor:
        model_mask = self.resize_image(self.load_image(self.agnostic_mask_files[key]))
        cloth_mask = self.resize_image(self.load_image(self.cloth_mask_files[key]))
        model_generated = self.resize_image(self.load_image(self.generated_images_files[key]))
        cloth = self.resize_image(self.load_image(self.garment_images_files[key]))
        model_real = self.resize_image(self.load_image(self.original_images_files[key]))
        input_cloth = cloth * cloth_mask
        input_model_generate = model_generated * model_mask
        input_scm = torch.cat((input_cloth, input_model_generate), dim=0)
        target = model_mask * model_real
        return {
            "file_name":self.images_files[key], 
            "model_mask": model_mask.to(self.device),
            "cloth_mask": cloth_mask.to(self.device),
            "model_generated": model_generated.to(self.device),
            "cloth": cloth.to(self.device),
            "model_real": model_real.to(self.device),
            "input_cloth": input_cloth.to(self.device),
            "input_model_generate": input_model_generate.to(self.device),
            "input_scm": input_scm.to(self.device),
            "target": target.to(self.device)
        }
    def load_image(self, path: str) -> Image.Image:
        try:
            image = Image.open(path).convert("RGB")
            image = image_to_tensor(image).squeeze(0)
            return image
        except Exception as e:
            print(f"Error loading image '{path}': {e}")
            return None
        
    def resize_image(self, image)-> torch.Tensor:
        if image.shape[-2:] != (self.high, self.width):
            image = self.resize(image)
        return image
        
    


if __name__ == "__main__":

    path_generated_images = "/var/hub/VITON-HD-results-ladi-vton"
    path_garment_images = "/var/hub/VITON-HD/test"
    mode = "train"

    dataset = SCMDataset(path_generated_images, path_garment_images, mode)
    elemt = dataset[0]
    for key, value in elemt.items():
        print(key, value.shape)
     
