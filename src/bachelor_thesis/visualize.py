# visualize.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Tuple

class Visualizer:
    """
    A class to handle visualization of LRP relevance maps, masks, and images.
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
        img = self.denorm_transform(image_tensor.cpu()).permute(1, 2, 0)
        return torch.clamp(img, 0, 1).numpy()

    def _preprocess_heatmap(self, relevance_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Prepares a relevance tensor as a heatmap for plotting.
        Uses a diverging colormap for positive and negative values.
        """
        heatmap = relevance_tensor.squeeze().cpu().numpy()
        # Find the maximum absolute value to center the colormap at zero
        vmax = np.abs(heatmap).max()
        return heatmap, vmax

    def _preprocess_mask(self, mask: torch.Tensor or np.ndarray) -> np.ndarray:
        """Ensures the mask is a 2D NumPy array."""
        if isinstance(mask, torch.Tensor):
            return mask.squeeze().cpu().numpy()
        return mask.squeeze()

    def plot_comparison(
        self,
        filename: str,
        image_tensor: torch.Tensor,
        mask: torch.Tensor or np.ndarray,
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
        base_heatmap, base_vmax = self._preprocess_heatmap(base_relevance)
        finetuned_heatmap, ft_vmax = self._preprocess_heatmap(finetuned_relevance)

        fig, axs = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle(f"Relevance Comparison for: {filename}", fontsize=20)
        
        # --- Titles with all scores ---
        base_title = (f"Base Model Scores | "
                      f"Total: {base_scores[0]:.3f}, "
                      f"Positive: {base_scores[1]:.3f}, "
                      f"Negative: {base_scores[2]:.3f}")
        
        finetuned_title = (f"Finetuned Model Scores | "
                           f"Total: {finetuned_scores[0]:.3f}, "
                           f"Positive: {finetuned_scores[1]:.3f}, "
                           f"Negative: {finetuned_scores[2]:.3f}")

        # --- Row 1: Base Model ---
        axs[0, 0].set_ylabel(base_title, fontsize=14, labelpad=20)
        axs[0, 0].imshow(img_np); axs[0, 0].set_title("Original Image"); axs[0, 0].axis('off')
        axs[0, 1].imshow(mask_np, cmap='gray'); axs[0, 1].set_title("Segmentation Mask"); axs[0, 1].axis('off')
        # Use a diverging colormap (e.g., coolwarm) to show pos/neg values
        im_base = axs[0, 2].imshow(base_heatmap, cmap='coolwarm', vmin=-base_vmax, vmax=base_vmax)
        axs[0, 2].set_title("Relevance Heatmap (Pos/Neg)"); axs[0, 2].axis('off')
        fig.colorbar(im_base, ax=axs[0, 2], orientation='vertical', fraction=0.046, pad=0.04)
        axs[0, 3].imshow(img_np)
        axs[0, 3].imshow(base_heatmap, cmap='coolwarm', vmin=-base_vmax, vmax=base_vmax, alpha=0.6)
        axs[0, 3].set_title("Relevance Overlay"); axs[0, 3].axis('off')

        # --- Row 2: Finetuned Model ---
        axs[1, 0].set_ylabel(finetuned_title, fontsize=14, labelpad=20)
        axs[1, 0].imshow(img_np); axs[1, 0].set_title("Original Image"); axs[1, 0].axis('off')
        axs[1, 1].imshow(img_np); axs[1, 1].contour(mask_np, colors='lime', linewidths=1.5)
        axs[1, 1].set_title("Image with Mask Outline"); axs[1, 1].axis('off')
        im_ft = axs[1, 2].imshow(finetuned_heatmap, cmap='coolwarm', vmin=-ft_vmax, vmax=ft_vmax)
        axs[1, 2].set_title("Relevance Heatmap (Pos/Neg)"); axs[1, 2].axis('off')
        fig.colorbar(im_ft, ax=axs[1, 2], orientation='vertical', fraction=0.046, pad=0.04)
        axs[1, 3].imshow(img_np)
        axs[1, 3].imshow(finetuned_heatmap, cmap='coolwarm', vmin=-ft_vmax, vmax=ft_vmax, alpha=0.6)
        axs[1, 3].set_title("Relevance Overlay"); axs[1, 3].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(self.save_dir, f"comparison_{filename.replace('.jpg', '.png')}")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)