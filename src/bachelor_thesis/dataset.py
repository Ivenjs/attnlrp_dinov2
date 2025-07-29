import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, List, Optional, Callable, Dict
from torch.utils.data.dataloader import default_collate

from mask_generator import MaskGenerator
from tqdm import tqdm

class GorillaReIDDataset(Dataset):
    """
    A dataset for gorilla Re-Identification tasks.
    This version is a lightweight loader, assuming masks are pre-generated.
    """
    def __init__(self, 
                 image_dir: str, 
                 filenames: List[str], 
                 transform: transforms.Compose, 
                 base_mask_dir: Optional[str] = None,
                 mask_transform: Optional[transforms.Compose] = None,
                 label_extractor: Callable[[str], str] = lambda f: f.split('_')[0],
                 dataset_name: Optional[str] = None):
        """
        Args:
            image_dir (str): Directory with all the images.
            filenames (list): List of specific image filenames to load.
            transform (transforms.Compose): Transforms for the input images.
            base_mask_dir (str, optional): Directory where pre-generated segmentation masks are stored.
            mask_transform (transforms.Compose, optional): Transforms for the masks.
            label_extractor (Callable, optional): Extracts label from filename.
            dataset_name (Optional[str], optional): Name used to construct mask path.
        """
        self.image_dir = image_dir
        self.filenames = filenames
        self.transform = transform
        self.mask_transform = mask_transform
        self.label_extractor = label_extractor
        self.dataset_name = dataset_name or os.path.basename(os.path.dirname(image_dir))
        self.split_name = os.path.basename(image_dir)

        self.image_paths = [os.path.join(image_dir, f) for f in self.filenames]
        self.labels = [self.label_extractor(f) for f in self.filenames]

        # This logic is now lightweight and fast. It just checks for files.
        if base_mask_dir:
            # Construct the full path to the specific mask directory
            mask_dir = os.path.join(base_mask_dir, self.dataset_name, self.split_name)
            self.mask_paths = [os.path.join(mask_dir, os.path.basename(f)) for f in self.filenames]
            self.has_mask = [os.path.exists(p) for p in self.mask_paths]
        else:
            self.mask_paths = [None] * len(self.filenames)
            self.has_mask = [False] * len(self.filenames)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        # This method remains largely the same, as it was already correct.
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        
        label = self.labels[idx]
        filename_no_ext = os.path.splitext(self.filenames[idx])[0]
        
        mask_tensor = None
        if self.has_mask[idx]: # We can rely on the pre-computed check
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert("L")
            if self.mask_transform:
                mask = self.mask_transform(mask)
            mask_tensor = (mask > 0).float()

        return {
            "image": image_tensor,
            "label": label,
            "mask": mask_tensor,
            "filename": filename_no_ext
        }

def custom_collate_fn(batch):
    """
    Custom collate function to handle None values for masks.
    'image', 'label', 'filename' are collated using the default collate.
    'mask' is returned as a list of tensors/Nones.
    """
    # Separate the masks from the rest of the data
    masks = [item.pop('mask') for item in batch]
    
    # Use the default collate for the rest of the items
    # This will create batched tensors for images, and lists for labels/filenames
    collated_batch = default_collate(batch)
    
    # Add the list of masks back into the collated batch
    collated_batch['mask'] = masks
    
    return collated_batch