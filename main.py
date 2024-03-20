from src.models.dataset import concatenate_dataset

if __name__ == "__main__":
    path_result = '/var/hub/VITON-HD-results-ladi-vton/paired/upper_body'
    path_cloth = '/var/hub/VITON-HD/test/cloth'
    concatenate_dataset(path_dataset_model = path_result, path_dataset_cloth = path_cloth)
    