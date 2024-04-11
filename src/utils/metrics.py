from tabnanny import verbose
from torch import Tensor
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from cleanfid import fid
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from loguru import logger

### Refiner imports ###
from refiners.fluxion.utils import image_to_tensor


def compute_ssim(pred: Tensor, target: Tensor):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    return ssim(pred, target)

def compute_lpips(pred: Tensor, target: Tensor):
    lpips = LearnedPerceptualImagePatchSimilarity()
    return lpips(pred, target)

def make_custom_stats(custom_name, dataset_path):
    return fid.make_custom_stats(custom_name, dataset_path, mode="clean")

def compute_fid(gen_folder):
    fid_score = fid.compute_fid(gen_folder, dataset_name="viton-hd", mode='clean', dataset_split="custom",
                                        verbose=True, use_dataparallel=False)
    return fid_score

def compute_kid(gen_folder):
    kid_score = fid.compute_kid(gen_folder, dataset_name="viton-hd", mode='clean', dataset_split="custom",
                                        verbose=True, use_dataparallel=False)
    return kid_score

def compute_metrics_w_gt(pred_folder, target_folder):
    ssim_list = []
    lpips_list = []

    pred_folder_list = os.listdir(pred_folder)
    target_folder_list = os.listdir(target_folder)

    for pred, target in tqdm(zip(pred_folder_list, target_folder_list)):
        if pred == target:
            pred_path = os.path.join(pred_folder, pred)
            target_path = os.path.join(target_folder, target)

            image_pred = Image.open(pred_path).convert("RGB")
            image_target = Image.open(target_path).convert("RGB")

            tensor_pred = image_to_tensor(image_pred)
            tensor_target = image_to_tensor(image_target)

            ssim_list.append(compute_ssim(tensor_pred, tensor_target))
            lpips_list.append(compute_lpips(tensor_pred, tensor_target))
        else :
            logger.error("Mismatch in pred and target folder")
            return None

    return np.mean(ssim_list), np.mean(lpips_list)
