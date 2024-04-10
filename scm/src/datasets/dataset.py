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
import tqdm

class GaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class SCMDataset:
    def __init__(self, path_generated_images, path_garment_images, mode, device = "cpu", high = 1024 , width = 768) -> None:
        self.path_generated_images = path_generated_images
        self.path_garment_images = path_garment_images
        self.high = high
        self.width = width
        self.mode = mode
        self.device = device
        self.gaussian_noise = GaussianNoise(0.0, 0.1)
        path_to_get_files_list = self.path_generated_images + "/" + self.mode + "_files_v2.txt"
        with open(path_to_get_files_list, 'r') as file:
            self.images_files = file.readlines()
        self.images_files = [image_file.strip() for image_file in self.images_files]
        self.images_parse_v3_files = [ path_garment_images + "/image-parse-v3/" + image_file.replace(".jpg",".png") for image_file in self.images_files]
        # self.update_image_files()
        self.generated_images_files = [ path_generated_images + "/paired/upper_body/"+ image_file for image_file in self.images_files]
        self.garment_images_files = [ path_garment_images + "/cloth/" + image_file for image_file in self.images_files]
        self.original_images_files = [ path_garment_images + "/image/" + image_file for image_file in self.images_files]
        self.agnostic_mask_files = [ path_garment_images + "/agnostic-mask/" + image_file.replace(".jpg","_mask.png") for image_file in self.images_files]
        self.cloth_mask_files = [ path_garment_images + "/cloth-mask/" + image_file for image_file in self.images_files]
        self.warped_cloth_files = [ path_generated_images + "/paired/warped_cloth/" + image_file for image_file in self.images_files]
        self.images_parse_v3_files = [ path_garment_images + "/image-parse-v3/" + image_file.replace(".jpg",".png") for image_file in self.images_files]
        self.resize = transforms.Resize((self.high, self.width))

    def update_image_files(self):
        wrong_files = []
        wrong_files_parse_v3 = []
        for i,file in tqdm.tqdm(enumerate(self.images_files)):
            mask = self.mask_orange_color(self.resize_image(self.load_image(self.images_parse_v3_files[i])))
            if mask.sum() < 0.05*mask.numel():
                wrong_files.append(file)
                wrong_files_parse_v3.append(self.images_parse_v3_files[i])

        for file in wrong_files:
            self.images_files.remove(file)

        for file in wrong_files_parse_v3:
            self.images_parse_v3_files.remove(file)

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
        model_real_noisy = self.gaussian_noise(model_real)
        warped_cloth = self.resize_image(self.load_image(self.warped_cloth_files[key]))
        parse_v3 = self.resize_image(self.load_image(self.images_parse_v3_files[key]))
        new_model_mask = self.mask_orange_color(parse_v3)
        input_cloth = cloth * cloth_mask
        input_model_generate = model_generated * new_model_mask
        input_scm = torch.cat((warped_cloth, input_model_generate), dim=0)
        target = model_real * new_model_mask
        return {
            "file_name":self.images_files[key], 
            "model_mask": new_model_mask.to(self.device),
            "cloth_mask": cloth_mask.to(self.device),
            "model_generated": model_generated.to(self.device),
            "cloth": cloth.to(self.device),
            "model_real_noisy": model_real_noisy.to(self.device),
            "model_real": model_real.to(self.device),
            "parse_v3": parse_v3.to(self.device),
            "input_cloth": input_cloth.to(self.device),
            "input_warped_cloth": warped_cloth.to(self.device), 
            "input_model_generate": input_model_generate.to(self.device),
            "input_scm": input_scm.to(self.device),
            "target": target.to(self.device)
        }
    

    def mask_orange_color(self,image: torch.Tensor, threshold: float = 0.5):
        # RGB of orange is (255, 165, 0)
        # Let's mask all the pixels around this color
        H,W = image.shape[1], image.shape[2]
        # Let's build tensor (3, H, W) with the RGB values of orange
        orange_tensor = torch.tensor([255, 165, 2]).unsqueeze(1).unsqueeze(2).repeat(1,H,W)/255
        white_tensor = torch.tensor([1, 1, 1]).unsqueeze(1).unsqueeze(2).repeat(1,H,W)
        # Let's compute the distance between the image and the orange color
        distance = torch.norm(image - orange_tensor, dim=0)
        # Let's build the mask
        mask = distance < threshold
        # image with only the orange pixels
        final_mask = white_tensor * mask
        
        return final_mask


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
     
