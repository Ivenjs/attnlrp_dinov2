import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, List, Optional, Callable, Dict
from torch.utils.data.dataloader import default_collate
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
from utils import deterministic_randperm, parse_encounter_id

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
                 video_extractor: Callable[[str], str] = lambda f: f.split('_')[1] + "_" + f.split('_')[2] + "_" + f.split('_')[3],
                 dataset_name: Optional[str] = None,
                 k: int = 5):
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
        self.video_extractor = video_extractor
        self.dataset_name = dataset_name or os.path.basename(os.path.dirname(image_dir))
        self.split_name = os.path.basename(image_dir)

        self.image_paths = [os.path.join(image_dir, f) for f in self.filenames]
        self.labels = [self.label_extractor(f) for f in self.filenames]
        self.videos = [self.video_extractor(f) for f in self.filenames]
        self.encounters = [parse_encounter_id(v) for v in self.videos]

        if base_mask_dir:
            # Construct the full path to the specific mask directory
            mask_dir = os.path.join(base_mask_dir, self.dataset_name, self.split_name)
            self.mask_paths = [os.path.join(mask_dir, os.path.basename(f)) for f in self.filenames]
            self.has_mask = [os.path.exists(p) for p in self.mask_paths]
        else:
            self.mask_paths = [None] * len(self.filenames)
            self.has_mask = [False] * len(self.filenames)

        self.k = k
        self._filter_images_for_knn()

    def _filter_images_for_knn(self):
        """
        Pre-computes lists of image indices that are suitable for
        Cross-Encounter and Standard KNN evaluation based on label distribution.
        An encounter is defined as a unique (camera, date) pair.
        """
        print(f"Filtering images for KNN evaluation with k={self.k}...")
        
        # 1. Build a map of labels to their encounters and image indices
        data_by_label_and_encounter = defaultdict(lambda: defaultdict(list))
        for i, (label, encounter) in enumerate(zip(self.labels, self.encounters)):
            if encounter[0] is not None:  # Ensure encounter was parsed correctly
                data_by_label_and_encounter[label][encounter].append(i)

        self.images_for_ce_knn = []  # CE: Cross-Encounter
        self.images_for_standard_knn = []
        
        # 2. Iterate through each image and apply filtering logic
        for idx, (label, current_encounter) in enumerate(zip(self.labels, self.encounters)):
            if current_encounter[0] is None:
                continue # Skip images that couldn't be parsed

            encounters_with_label = data_by_label_and_encounter[label]
            
            # --- Cross-Encounter KNN Filter ---
            # Count how many images with the same label exist in OTHER encounters
            other_encounters_count = sum(
                len(images) for enc, images in encounters_with_label.items() if enc != current_encounter
            )
            # To have a valid positive match, we need enough images in other encounters.
            # The k//2 + 1 heuristic ensures a majority vote is possible in KNN.
            if other_encounters_count >= self.k // 2 + 1:
                self.images_for_ce_knn.append(idx)

            # --- Standard KNN Filter ---
            # Total images for this label, minus the query image itself.
            total_images_with_label = sum(len(images) for images in encounters_with_label.values())
            # We need at least k//2 + 1 other images for a majority vote.
            if total_images_with_label - 1 >= self.k // 2 + 1:
                self.images_for_standard_knn.append(idx)


        print(f"Found {len(self.images_for_ce_knn)} / {len(self)} images suitable for Cross-Encounter KNN.")
        print(f"Found {len(self.images_for_standard_knn)} / {len(self)} images suitable for Standard KNN.")

        # Fallback: if no eligible images are found, use all images to avoid crashing.
        if not self.images_for_standard_knn:
            print("Warning: No images were found suitable for Standard KNN evaluation. Using all images instead.")
            self.images_for_standard_knn = list(range(len(self)))
        if not self.images_for_ce_knn:
            print("Warning: No images were found suitable for Cross-Encounter KNN evaluation. Using all images instead.")
            self.images_for_ce_knn = list(range(len(self)))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        # This method remains largely the same, as it was already correct.
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        
        label = self.labels[idx]
        video = self.videos[idx]
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
            "filename": filename_no_ext,
            "video": video,
            "original_index": idx
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

class PerturbedGorillaReIDDataset(Dataset):
    """
    A wrapper dataset that applies perturbations to images based on relevance maps.
    This is used to evaluate the impact of LRP-guided perturbations on downstream
    k-NN accuracy.

    This version filters the dataset at initialization to only include samples
    for which a relevance map is available, ensuring methodological purity.
    """
    def __init__(self,
                 base_dataset,
                 relevance_maps: Dict[str, torch.Tensor],
                 perturbation_mode: str, # 'morf', 'lerf', or 'random'
                 perturbation_fraction: float,
                 patch_size: int,
                 seed: int = 161,
                 baseline_value: str = "black"):
        
        self.base_dataset = base_dataset
        self.relevance_maps = relevance_maps
        self.perturbation_mode = perturbation_mode.lower()
        self.perturbation_fraction = perturbation_fraction
        self.patch_size = patch_size
        self.baseline_value = baseline_value
        self.seed = seed

        if self.perturbation_mode not in ['morf', 'lerf', 'random']:
            raise ValueError("perturbation_mode must be 'morf', 'lerf', or 'random'")
            
        # --- Pre-filter the dataset to only include valid samples ---
        self.valid_indices = []
        print("Filtering dataset for available relevance maps...")
        for i, filename_with_ext in enumerate(self.base_dataset.filenames):
            filename_no_ext = os.path.splitext(filename_with_ext)[0]
            if filename_no_ext in self.relevance_maps:
                self.valid_indices.append(i)
        
        original_count = len(self.base_dataset)
        valid_count = len(self.valid_indices)
        print(f"Found relevance maps for {valid_count} out of {original_count} images.")
        #print how many relevance maps are filled with 0 only
        zero_count = sum(1 for map in self.relevance_maps.values() if torch.all(map == 0))
        print(f"Found {zero_count} relevance maps filled with zeros.")
        if valid_count == 0:
            raise ValueError("No relevance maps found for any images in the dataset. Cannot proceed.")

        # --- Create a map from original index to new, filtered index ---
        self.original_to_new_idx_map = {original_idx: new_idx for new_idx, original_idx in enumerate(self.valid_indices)}

        # --- Create new proxy attributes, re-mapping indices where needed ---
        self.filenames = [self.base_dataset.filenames[i] for i in self.valid_indices]
        self.labels = [self.base_dataset.labels[i] for i in self.valid_indices]
        
        # Correctly re-map the k-NN index lists
        self.images_for_ce_knn = [
            self.original_to_new_idx_map[original_knn_idx]
            for original_knn_idx in self.base_dataset.images_for_ce_knn
            if original_knn_idx in self.original_to_new_idx_map
        ]
        
        self.images_for_standard_knn = [
            self.original_to_new_idx_map[original_knn_idx]
            for original_knn_idx in self.base_dataset.images_for_standard_knn
            if original_knn_idx in self.original_to_new_idx_map
        ]
        print(f"Re-mapped k-NN lists. CE-KNN valid queries: {len(self.images_for_ce_knn)}. Standard-KNN valid queries: {len(self.images_for_standard_knn)}")


    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict:
        original_idx = self.valid_indices[idx]
        
        original_data = self.base_dataset[original_idx]
        input_tensor = original_data["image"]
        filename = original_data["filename"]

        # If perturbation fraction is 0, just return the original image
        if self.perturbation_fraction == 0.0:
            return original_data
            
        relevance_map = self.relevance_maps[filename]

        # Note: A relevance_map of all zeros will result in a random-like perturbation
        patch_relevance = F.avg_pool2d(relevance_map, 
                                       kernel_size=self.patch_size, 
                                       stride=self.patch_size)
        patch_relevance_flat = patch_relevance.flatten()
        
        num_patches = len(patch_relevance_flat)
        
        if self.perturbation_mode == 'morf':
            order = torch.argsort(patch_relevance_flat, descending=True)
        elif self.perturbation_mode == 'lerf':
            order = torch.argsort(patch_relevance_flat, descending=False)
        else: # 'random'
            key = self.base_dataset.filenames[idx] 
            order = deterministic_randperm(num_patches, key, self.seed)
            
        num_patches_to_perturb = int(np.floor(self.perturbation_fraction * num_patches))
        patches_to_perturb = order[:num_patches_to_perturb]

        perturbed_tensor = input_tensor.clone()
        
        if self.baseline_value.lower() == "black":
            baseline_fill = torch.zeros_like(input_tensor)
        elif self.baseline_value.lower() == "mean":
            mean_color = input_tensor.mean(dim=[1, 2], keepdim=True)
            baseline_fill = mean_color.expand_as(input_tensor)
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_value}")

        h, w = input_tensor.shape[-2:]
        num_patches_w = w // self.patch_size

        for patch_idx in patches_to_perturb:
            row = (patch_idx // num_patches_w) * self.patch_size
            col = (patch_idx % num_patches_w) * self.patch_size
            
            perturbed_tensor[..., row:row+self.patch_size, col:col+self.patch_size] = \
                baseline_fill[..., row:row+self.patch_size, col:col+self.patch_size]
        
        original_data["image"] = perturbed_tensor
        return original_data