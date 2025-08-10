# visualize.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Tuple, Union

# Import imgify from zennit
from zennit.image import imgify

class AttentionVisualizer:
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

    def plot_heatmap(
        self,
        filename: str,
        image_tensor: torch.Tensor,
        mask: Union[torch.Tensor, np.ndarray],
        relevance: torch.Tensor,
        scores: Tuple[float, float, float]
    ):
        """
        Generates and saves a comprehensive relevance plot for a single model and image.

        Args:
            filename (str): The original filename, used for titles and saving.
            image_tensor (torch.Tensor): The input image tensor.
            mask (Union[torch.Tensor, np.ndarray]): The ground truth segmentation mask.
            relevance (torch.Tensor): The relevance map from the model.
            scores (Tuple[float, float, float]): A tuple containing (total, positive, negative)
                                                 relevance scores.
        """
        img_np = self._preprocess_image(image_tensor)
        mask_np = self._preprocess_mask(mask)

        # Create a 1x4 plot grid. Adjust figsize for the new layout.
        fig, axs = plt.subplots(1, 4, figsize=(24, 7))
        fig.suptitle(f"Relevance for: {filename}", fontsize=20)
        
        # Create a single title string for the scores
        scores_title = (f"Scores | "
                        f"Total: {scores[0]:.3f}, "
                        f"Positive: {scores[1]:.3f}, "
                        f"Negative: {scores[2]:.3f}")
        
        # --- Create a shared color mapping for the heatmap ---
        norm = plt.Normalize(vmin=-1.0, vmax=1.0)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')

        # Use the scores title as a Y-axis label for the first plot for a clean look
        axs[0].set_ylabel(scores_title, fontsize=14, labelpad=20)

        # --- Column 1: Original Image ---
        axs[0].imshow(img_np)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        # --- Column 2: Image with Mask Outline ---
        axs[1].imshow(img_np)
        axs[1].contour(mask_np, colors='lime', linewidths=1.5)
        axs[1].set_title("Image with Mask Outline")
        axs[1].axis('off')

        # --- Column 3: Relevance Heatmap ---
        relevance = relevance.squeeze()
        relevance = relevance / abs(relevance).max()
        heatmap_img = imgify(relevance, vmin=-1.0, vmax=1.0)
        axs[2].imshow(heatmap_img)
        axs[2].set_title("Relevance Heatmap")
        axs[2].axis('off')
        # Add a colorbar to the heatmap plot
        fig.colorbar(mappable, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)

        # --- Column 4: Relevance Overlay ---
        axs[3].imshow(img_np)  # Plot base image first
        axs[3].imshow(heatmap_img, alpha=0.6)  # Plot heatmap on top with transparency
        axs[3].set_title("Relevance Overlay")
        axs[3].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjust rect to fit suptitle
        
        # Modify save path to include model name and avoid overwrites
        save_filename = f"relevance_{filename.replace('.jpg', '.png')}"
        save_path = os.path.join(self.save_dir, save_filename)
        
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)