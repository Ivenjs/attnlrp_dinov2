# visualize.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Tuple, Union, Dict, Any

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
        stats: Dict[str, Any]  
    )-> str:
        """
        Generates and saves a comprehensive relevance plot for a single model and image.

        Args:
            filename (str): A descriptive filename, e.g., "reason_originalfilename".
            image_tensor (torch.Tensor): The input image tensor.
            mask (Union[torch.Tensor, np.ndarray]): The ground truth segmentation mask.
            relevance (torch.Tensor): The relevance map from the model.
            stats (Dict[str, Any]): A dictionary containing all stats for the image
                                    from the masking experiment DataFrame.
        """
        img_np = self._preprocess_image(image_tensor)
        mask_np = self._preprocess_mask(mask)

        parts = os.path.splitext(filename)[0].split('_', 1)
        reason = parts[0].replace('-', ' ').title()
        clean_filename = parts[1] if len(parts) > 1 else filename

        fig, axs = plt.subplots(1, 4, figsize=(24, 8))
        fig.suptitle(f"Analysis for: {clean_filename}\n(Category: {reason})", fontsize=20, y=1.02)

        rank_orig = stats.get('rank_orig', 'N/A')
        rank_masked = stats.get('rank_masked', 'N/A')
        delta_proxy = stats.get('delta_proxy_score', 0.0)
        aogr_total = stats.get('AoGR_total', 0.0)

        stats_text = (
            f"--- Performance ---\n"
            f"Original Rank: {rank_orig}\n"
            f"Masked Rank:   {rank_masked}\n"
            f"Δ Proxy Score: {delta_proxy:+.3f}\n\n"
            f"--- Attention ---\n"
            f"Gorilla Attention (AoGR): {aogr_total:.2%}"
        )
        
        fig.text(
            0.01, 0.99, stats_text,
            fontsize=10, family='monospace',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8)
        )

        norm = plt.Normalize(vmin=-1.0, vmax=1.0)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')

        axs[0].imshow(img_np)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(img_np)
        axs[1].contour(mask_np, colors='lime', linewidths=1.5)
        axs[1].set_title("Mask Outline")
        axs[1].axis('off')

        relevance_norm = relevance.squeeze() / torch.abs(relevance).max()
        heatmap_img = imgify(relevance_norm, vmin=-1.0, vmax=1.0)
        axs[2].imshow(heatmap_img)
        axs[2].set_title("Relevance Heatmap")
        axs[2].axis('off')
        fig.colorbar(mappable, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)

        axs[3].imshow(img_np)
        axs[3].imshow(heatmap_img, alpha=0.6)
        axs[3].set_title("Relevance Overlay")
        axs[3].axis('off')
        

        plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjust rect to fit suptitle

        save_path = os.path.join(self.save_dir, f"{filename}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return save_path