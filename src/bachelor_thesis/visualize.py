# visualize.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Tuple, Union

# Import imgify from zennit
from zennit.image import imgify

class Visualizer:
    """
    A class to handle visualization of LRP relevance maps, masks, and images.
    Uses zennit.image.imgify for high-quality relevance map plotting and
    matplotlib for creating overlays.
    """
    def __init__(self, save_dir: str, denorm_transform: transforms.Compose):
        """
        Initializes the Visualizer.
        """
        self.save_dir = save_dir
        self.denorm_transform = denorm_transform
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Visualizations will be saved to: {os.path.abspath(self.save_dir)}")

    def _preprocess_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """De-normalizes and prepares an image tensor for plotting."""
        # Squeeze the batch dimension if it exists
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
        img = self.denorm_transform(image_tensor.cpu()).permute(1, 2, 0)
        return torch.clamp(img, 0, 1).numpy()

    def _preprocess_mask(self, mask: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Ensures the mask is a 2D NumPy array."""
        if isinstance(mask, torch.Tensor):
            return mask.squeeze().cpu().numpy()
        return mask.squeeze()

    def plot_comparison(
        self,
        filename: str,
        image_tensor: torch.Tensor,
        mask: Union[torch.Tensor, np.ndarray],
        base_relevance: torch.Tensor,
        finetuned_relevance: torch.Tensor,
        base_scores: Tuple[float, float, float],
        finetuned_scores: Tuple[float, float, float]
    ):
        """
        Generates and saves a comprehensive comparison plot for a single image.
        """
        img_np = self._preprocess_image(image_tensor)
        mask_np = self._preprocess_mask(mask)

        fig, axs = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle(f"Relevance Comparison for: {filename}", fontsize=20)
        
        base_title = (f"Base Model Scores | "
                      f"Total: {base_scores[0]:.3f}, "
                      f"Positive: {base_scores[1]:.3f}, "
                      f"Negative: {base_scores[2]:.3f}")
        
        finetuned_title = (f"Finetuned Model Scores | "
                           f"Total: {finetuned_scores[0]:.3f}, "
                           f"Positive: {finetuned_scores[1]:.3f}, "
                           f"Negative: {finetuned_scores[2]:.3f}")

        norm = plt.Normalize(vmin=-1.0, vmax=1.0)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')

        # --- Row 1: Base Model ---
        axs[0, 0].set_ylabel(base_title, fontsize=14, labelpad=20)
        axs[0, 0].imshow(img_np); axs[0, 0].set_title("Original Image"); axs[0, 0].axis('off')
        axs[0, 1].imshow(mask_np, cmap='gray'); axs[0, 1].set_title("Segmentation Mask"); axs[0, 1].axis('off')

        # Heatmap Only (Column 3)
        base_heatmap_img = imgify(base_relevance.squeeze(), vmin=-1.0, vmax=1.0)
        axs[0, 2].imshow(base_heatmap_img)
        axs[0, 2].set_title("Relevance Heatmap"); axs[0, 2].axis('off')
        fig.colorbar(mappable, ax=axs[0, 2], orientation='vertical', fraction=0.046, pad=0.04)
        
        # --- MODIFIED: Create overlay manually using Matplotlib's alpha ---
        # Overlay (Column 4)
        axs[0, 3].imshow(img_np) # Plot base image first
        axs[0, 3].imshow(base_heatmap_img, alpha=0.6) # Plot heatmap on top with transparency
        axs[0, 3].set_title("Relevance Overlay"); axs[0, 3].axis('off')


        # --- Row 2: Finetuned Model ---
        axs[1, 0].set_ylabel(finetuned_title, fontsize=14, labelpad=20)
        axs[1, 0].imshow(img_np); axs[1, 0].set_title("Original Image"); axs[1, 0].axis('off')
        axs[1, 1].imshow(img_np); axs[1, 1].contour(mask_np, colors='lime', linewidths=1.5)
        axs[1, 1].set_title("Image with Mask Outline"); axs[1, 1].axis('off')

        # Heatmap Only (Column 3)
        ft_heatmap_img = imgify(finetuned_relevance.squeeze(), vmin=-1.0, vmax=1.0)
        axs[1, 2].imshow(ft_heatmap_img)
        axs[1, 2].set_title("Relevance Heatmap"); axs[1, 2].axis('off')
        fig.colorbar(mappable, ax=axs[1, 2], orientation='vertical', fraction=0.046, pad=0.04)

        # Overlay (Column 4)
        axs[1, 3].imshow(img_np) # Plot base image first
        axs[1, 3].imshow(ft_heatmap_img, alpha=0.6) # Plot heatmap on top with transparency
        axs[1, 3].set_title("Relevance Overlay"); axs[1, 3].axis('off')


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(self.save_dir, f"comparison_{filename.replace('.jpg', '.png')}")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)