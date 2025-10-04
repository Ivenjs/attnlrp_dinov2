import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Optional, Callable, Dict
from torch.utils.data.dataloader import default_collate
from collections import defaultdict
from utils import parse_encounter_id

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
            if "zoo" in self.image_dir.lower():
                mask_dir = os.path.join(base_mask_dir, self.split_name)
            else:
                mask_dir = os.path.join(base_mask_dir, self.dataset_name, self.split_name)

            self.mask_paths = [os.path.join(mask_dir, os.path.basename(f)) for f in self.filenames]
            self.has_mask = [os.path.exists(p) for p in self.mask_paths]
            print(f"Mask directory: {mask_dir}")
            print(f"Found {sum(self.has_mask)} / {len(self.has_mask)} masks.")
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
        
        data_by_label_and_encounter = defaultdict(lambda: defaultdict(list))
        for i, (label, encounter) in enumerate(zip(self.labels, self.encounters)):
            if encounter[0] is not None: 
                data_by_label_and_encounter[label][encounter].append(i)
            else:
                print(f"Warning: Could not parse encounter for image {self.filenames[idx]}. Skipping KNN filtering for this image.")

        self.images_for_ce_knn = []
        self.images_for_standard_knn = []
        
        for idx, (label, current_encounter) in enumerate(zip(self.labels, self.encounters)):
            if current_encounter[0] is None:
                print(f"Warning: Could not parse encounter for image {self.filenames[idx]}. Skipping KNN filtering for this image.")
                continue 

            encounters_with_label = data_by_label_and_encounter[label]
            
            other_encounters_count = sum(
                len(images) for enc, images in encounters_with_label.items() if enc != current_encounter
            )
            # To have a valid positive match, we need enough images in other encounters.
            # The k//2 + 1 ensures a majority vote is possible in KNN.
            if other_encounters_count >= self.k // 2 + 1:
                self.images_for_ce_knn.append(idx)

            total_images_with_label = sum(len(images) for images in encounters_with_label.values())
            # We need at least k//2 + 1 other images for a majority vote.
            if total_images_with_label - 1 >= self.k // 2 + 1:
                self.images_for_standard_knn.append(idx)


        print(f"Found {len(self.images_for_ce_knn)} / {len(self)} images suitable for Cross-Encounter KNN.")
        print(f"Found {len(self.images_for_standard_knn)} / {len(self)} images suitable for Standard KNN.")

        if not self.images_for_standard_knn:
            print("Warning: No images were found suitable for Standard KNN evaluation. Using all images instead.")
            self.images_for_standard_knn = list(range(len(self)))
        if not self.images_for_ce_knn:
            print("Warning: No images were found suitable for Cross-Encounter KNN evaluation. Using all images instead.")
            self.images_for_ce_knn = list(range(len(self)))
        
        print(f"Total unique labels: {len(set(self.labels))}")
        print(f"Total unique encounters: {len(set(self.encounters))}")
        labels_ce = {self.labels[i] for i in self.images_for_ce_knn}
        encounters_ce = {self.encounters[i] for i in self.images_for_ce_knn}
        print(f"CE KNN unique labels: {len(labels_ce)}, unique encounters: {len(encounters_ce)}")


    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        
        label = self.labels[idx]
        video = self.videos[idx]
        filename_no_ext = os.path.splitext(self.filenames[idx])[0]
        
        mask_tensor = None
        if self.has_mask[idx]:
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert("L")
            if self.mask_transform:
                mask = self.mask_transform(mask)
            mask_tensor = (mask > 0).float()
        else:
            print(f"No mask found for {self.filenames[idx]}.")

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
    masks = [item.pop('mask') for item in batch]
    
    collated_batch = default_collate(batch)
    
    collated_batch['mask'] = masks
    
    return collated_batch