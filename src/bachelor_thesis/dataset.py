import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, List

class ImageFileDataset(Dataset):
    def __init__(self, image_dir: str, filenames: List[str], transform: transforms.Compose):
        """
        Args:
            image_dir (str): Directory with all the images.
            filenames (list): List of specific image filenames (e.g., 'class1_001.png') to load.
            transform (transforms.Compose): Torchvision transforms to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f) for f in filenames]
        self.filenames_no_ext = [f.split('.')[0] for f in filenames]

    def __len__(self) -> int:
        """Returns the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Fetches the image and its metadata at a given index.
        """
        img_path = self.image_paths[idx]
        filename = self.filenames_no_ext[idx]
        
        # Open image, convert to RGB, and apply transforms
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        
        return image_tensor, filename