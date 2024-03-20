### External imports ###
from PIL import Image
import os

### Refiner imports ###
from refiners.fluxion.utils import image_to_tensor, tensor_to_image
import torchvision.transforms as transforms
import torch
import torchvision


def load_image(image_path: str) -> Image.Image:
        try:
            image = Image.open(image_path).convert("RGB")
            image = image_to_tensor(image).squeeze(0)
            return image
        except Exception as e:
            print(f"Error loading image '{file}': {e}")
            return None
                
def concatenate_dataset(path_dataset_model, path_dataset_cloth):
    items = os.listdir(path_dataset_model)

    transform = transforms.Compose([
    transforms.Resize((512, 384))  # Resize to 512x384
    ])

    for item in items:
        # get path of each image
        item_path_result = os.path.join(path_dataset_model, item)
        item_path_cloth = os.path.join(path_dataset_cloth, item)
        # load the image
        image_result = load_image(item_path_result)
        image_cloth = load_image(item_path_cloth)
        #resize the cloth image to image_result size (512x384)
        resized_cloth = transform(image_cloth)
        #concatenate the two images for the NAFFNet_Combine model
        concat_image = torch.cat((resized_cloth, image_result),0)
        print(type(concat_image))
        #save the concatenated image
        save_path = os.path.join("/var/hub/VITON-HD-results-concatenated/paired/upper_body/", item)

        # torchvision.utils.save_image(concat_image, save_path)


