import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, List, Optional, Callable, Dict
from torch.utils.data.dataloader import default_collate

from MaskGenerator import MaskGenerator
from tqdm import tqdm

class GorillaReIDDataset(Dataset):
    """
    A dataset for gorilla Re-Identification tasks that handles images, labels, 
    and optional segmentation masks.
    """
    def __init__(self, 
                 image_dir: str, 
                 filenames: List[str], 
                 transform: transforms.Compose, 
                 mask_dir: Optional[str] = None,
                 mask_transform: Optional[transforms.Compose] = None,
                 generate_masks_from: Optional[str] = None, 
                 mask_generator: Optional[MaskGenerator] = None,
                 mask_gen_batch_size: int = 32,
                 label_extractor: Callable[[str], str] = lambda f: f.split('_')[0],
                 dataset_name: Optional[str] = None):
        """
        Args:
            image_dir (str): Directory with all the images: '/some/path/eval_body_squared_cleaned_open_2024_bigval/train'
            filenames (list): List of specific image filenames (e.g., 'NN01.png') to load.
            transform (transforms.Compose): Torchvision transforms for the input images.
            mask_dir (str, optional): Directory with segmentation masks. Assumes mask filenames
                                      match image filenames. Defaults to None.
            mask_transform (transforms.Compose, optional): Torchvision transforms for the masks. 
                                                          IMPORTANT: Should not include normalization.
                                                          Defaults to None.
            generate_masks_from (Optional[str]): "None" for no masks, "cropped" for directly from images and "database" for getting frames and boxes from database
            mask_generator (Optional[MaskGenerator]): wraps a sam2 model to generate masks
            mask_gen_batch_size (int): Batch size for mask generation. Defaults to 32.
            label_extractor (Callable, optional): A function that extracts the individual ID (label)
                                                  from a filename. Defaults to taking the part before
                                                  the first underscore.
            dataset_name (Optional[str], optional): Name of the dataset. If not provided, it will be derived
                                                 from the image directory name.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = filenames
        
        self.transform = transform
        self.mask_transform = mask_transform
        self.label_extractor = label_extractor

        # If dataset_name is not provided, use the parent directory name of image_dir
        self.dataset_name = dataset_name or os.path.basename(os.path.dirname(image_dir))


        # --- Pre-process paths and labels for efficiency ---
        self.image_paths = [os.path.join(image_dir, f) for f in self.filenames]
        self.labels = [self.label_extractor(f) for f in self.filenames]

        if self.mask_dir and generate_masks_from == "cropped":
            assert mask_generator is not None, "A MaskGenerator instance must be provided."
            
            print(f"Mode 'cropped' is active. Checking for and generating masks in {self.mask_dir}...")
            #TODO: The saving logic should also take the dataset_name AND THE SPLIT (for closed set) into account.
            os.makedirs(self.mask_dir, exist_ok=True)
            
            generation_jobs = []
            for filename in self.filenames:
                mask_path = os.path.join(self.mask_dir, os.path.basename(filename))
                if not os.path.exists(mask_path):
                    img_path = os.path.join(self.image_dir, os.path.basename(filename))
                    generation_jobs.append({'img_path': img_path, 'mask_path': mask_path})
            
            if generation_jobs:
                print(f"Found {len(generation_jobs)} missing masks. Generating in batches of {mask_gen_batch_size}...")
                
                # 2. Process jobs in batches
                for i in tqdm(range(0, len(generation_jobs), mask_gen_batch_size), desc="Generating Mask Batches"):
                    batch_jobs = generation_jobs[i : i + mask_gen_batch_size]
                    
                    try:
                        image_batch_pil = [Image.open(job['img_path']).convert("RGB") for job in batch_jobs]
                        
                        generated_masks_np = mask_generator.generate_masks_from_crops_batch(image_batch_pil)
                        
                        for job, mask_np in zip(batch_jobs, generated_masks_np):
                            mask_pil = Image.fromarray(mask_np * 255)
                            mask_pil.save(job['mask_path'])
                    
                    except Exception as e:
                        print(f"Warning: Failed to process a batch. Error: {e}")

        
        if self.mask_dir:
            self.mask_paths = [os.path.join(mask_dir, os.path.basename(f)) for f in self.filenames]
            self.has_mask = [os.path.exists(p) for p in self.mask_paths]
        else:
            self.mask_paths = [None] * len(self.filenames)
            self.has_mask = [False] * len(self.filenames)

    def get_dataset_name(self) -> str:
        return self.dataset_name

    def __len__(self) -> int:
        """Returns the total number of images."""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Optional[torch.Tensor], str]:
        """
        Fetches the image, its label, an optional mask, and the original filename.
        
        Returns:
            Tuple[torch.Tensor, str, Optional[torch.Tensor], str]: A tuple containing:
                - image_tensor (torch.Tensor): The transformed image.
                - label (str): The individual ID of the gorilla (e.g., 'zola').
                - mask_tensor (torch.Tensor or None): The transformed binary mask (1s for gorilla, 0s for background).
                                                     None if mask_dir is not provided or mask is missing.
                - filename (str): The original filename without extension.
        """
        # 1. Load Image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        
        # 2. Get Label and Filename
        label = self.labels[idx]
        filename_no_ext = os.path.splitext(self.filenames[idx])[0]
        
        # 3. Load Mask (if applicable)
        mask_tensor = None
        if self.mask_dir and self.has_mask[idx]:
            mask_path = self.mask_paths[idx]
            # Masks are typically grayscale (L mode)
            mask = Image.open(mask_path).convert("L")
            
            if self.mask_transform:
                mask = self.mask_transform(mask)
            
            # Ensure the mask is a binary tensor of 0s and 1s
            mask_tensor = (mask > 0).float()

        return {
            "image": image_tensor,
            "label": label,
            "mask": mask_tensor,  # can be None
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