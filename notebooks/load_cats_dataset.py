from datasets import load_dataset
import os
import tqdm
import logging

logging.basicConfig(level=logging.INFO)

def resize_image_to_eightdiv_sizes(image):
    """
    Resize the image to sizes that are divisible by 8 using PIL
    """
    # Get the current size of the image
    width, height = image.size
    # Resize the image using PIL
    resized_image = image.resize((width - width % 8, height - height % 8))
    return resized_image

# Load the dataset
logging.info("Loading dataset...")
dataset = load_dataset("cats_vs_dogs",split="train")

# Apply a filter to keep only cats using label 0
logging.info("Filtering dataset to have only cat photos...")
dataset = dataset.filter(lambda x: x['labels'] == 0)

# Select subset of the first 10_000 images
# dataset = dataset.select(range(10000))

# Set the path for the output folder
folder = "data_cats"
os.makedirs(folder, exist_ok=True)

# Iterate through the datasets and save images
logging.info("Saving images...")
idx=1
for data_point in tqdm.tqdm(dataset):
    try:
        image = data_point['image']
        image = resize_image_to_eightdiv_sizes(image)
        image_filename = f"image_{idx}.png"
        image_path = os.path.join(folder, image_filename)
        image.save(image_path)
        idx+=1
    except:
        pass