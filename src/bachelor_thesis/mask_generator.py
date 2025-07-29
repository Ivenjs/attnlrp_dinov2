import torch
import numpy as np
from PIL import Image
import hydra
from pathlib import Path
from typing import List
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class MaskGenerator:
    """
    A class to encapsulate SAM2 model loading and mask generation
    for pre-cropped images.
    """
    def __init__(self, model_checkpoint_path: str, model_config_dir: str):
        """
        Initializes the device and the SAM2 model.
        Args:
            model_checkpoint_path (str): Path to the SAM .pt file.
            model_config_dir (str): Path to the directory containing model YAMLs.
        """
        print("Initializing MaskGenerator...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        # Reset hydra to ensure it finds our configs correctly.
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize_config_dir(config_dir=str(Path(model_config_dir).resolve()), version_base="1.2")
        
        sam2_model = build_sam2("sam2.1_hiera_l.yaml", model_checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def generate_mask_from_crop(self, image_crop: Image.Image) -> np.ndarray:
        """
        Generates a binary mask for a single pre-cropped image.
        Args:
            image_crop (PIL.Image.Image): A PIL image of the cropped gorilla.
        Returns:
            np.ndarray: A binary (0 or 1) NumPy array of the mask.
        """
        image_np_rgb = np.array(image_crop.convert("RGB"))
        
        self.predictor.set_image(image_np_rgb)
        
        h, w, _ = image_np_rgb.shape
        # The inset helps SAM avoid edge cases.
        box_prompt = np.array([1, 1, w - 1, h - 1]) 
        
        # 4. Predict
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_prompt[None, :], # Predictor expects a batch dimension
            multimask_output=False,
        )
        
        # masks shape is [1, 1, H, W], we want [H, W]
        binary_mask = masks.squeeze().cpu().numpy().astype(np.uint8)
        
        return binary_mask

    def generate_masks_from_crops_batch(self, image_crops: list[Image.Image]) -> list[np.ndarray]:
        """
        Generates a batch of binary masks for a list of pre-cropped images.
        Args:
            image_crops (list[PIL.Image.Image]): A list of PIL images.
        Returns:
            list[np.ndarray]: A list of binary NumPy array masks.
        """
        if not image_crops:
            return []

        # 1. Convert all PIL images to NumPy and create box prompts
        image_np_batch = [np.array(img.convert("RGB")) for img in image_crops]
        box_prompts_batch = []
        for img_np in image_np_batch:
            h, w, _ = img_np.shape
            box_prompts_batch.append(np.array([1, 1, w - 1, h - 1]))

        # 2. Set the image batch in the predictor
        self.predictor.set_image_batch(image_np_batch)

        # 3. Predict the batch
        masks_batch, scores, _ = self.predictor.predict_batch(
            None,
            None,
            box_batch=box_prompts_batch,
            multimask_output=False,
        )

        # masks_batch is a tensor of shape [1, H, W]
        binary_masks = [m.squeeze().astype(np.uint8) for m in masks_batch]
        
        return binary_masks
        
    def generate_masks_from_boxes_batch(self, full_images: List[np.ndarray], box_prompts: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generates a batch of binary masks for a list of full images using box prompts.
        
        Args:
            full_images (List[np.ndarray]): A list of full-frame images as RGB NumPy arrays.
            box_prompts (List[np.ndarray]): A list of corresponding bounding boxes [x0, y0, x1, y1].

        Returns:
            List[np.ndarray]: A list of binary NumPy array masks.
        """
        if not full_images:
            return []

        self.predictor.set_image_batch(full_images)

        masks_batch, _, _ = self.predictor.predict_batch(
            None,
            None,
            box_batch=box_prompts,
            multimask_output=False,
        )

        binary_masks = [m.squeeze().astype(np.uint8) for m in masks_batch]
        
        return binary_masks