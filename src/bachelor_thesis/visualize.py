# visualize.py

import argparse
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Tuple, Union, Dict, Any
import torch.nn.functional as F

# Import imgify from zennit
from zennit.image import imgify

from torch.utils.data import DataLoader
from lxt.efficient import monkey_patch_zennit
from dataset import GorillaReIDDataset, custom_collate_fn
from basemodel import get_model_wrapper
from knn_helpers import get_knn_db
from lrp_helpers import get_relevances
from utils import deterministic_randperm, get_db_path, get_mask_transform, load_config, get_denormalization_transform

class AttentionVisualizer:
    """
    A class to handle visualization of LRP relevance maps, masks, and images.
    Uses zennit.image.imgify for high-quality relevance map plotting and
    matplotlib for creating overlays.
    """
    def __init__(self, save_dir: str, denorm_transform: transforms.Compose, seed: int = 161):
        """
        Initializes the Visualizer.
        """
        self.save_dir = save_dir
        self.denorm_transform = denorm_transform
        self.seed = seed
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
        stats: Dict[str, Any],
        intensify: bool = False
    ) -> str:
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
        if intensify:
            relevance_norm = torch.tanh(3 * relevance_norm)

        heatmap_img = imgify(relevance_norm, vmin=-1.0, vmax=1.0)
        axs[2].imshow(heatmap_img)
        axs[2].set_title("Relevance Heatmap")
        axs[2].axis('off')
        fig.colorbar(mappable, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)

        axs[3].imshow(img_np)
        axs[3].imshow(heatmap_img, alpha=0.6)
        axs[3].set_title("Relevance Overlay")
        axs[3].contour(mask_np, colors='lime', linewidths=1.5)
        axs[3].axis('off')
        

        plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjust rect to fit suptitle

        save_path = os.path.join(self.save_dir, f"{filename}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return save_path
    
    def plot_perturbation(
        self,
        filename: str,
        image_tensor: torch.Tensor,
        relevance_map: torch.Tensor,
        patch_size: int,
        perturbation_fraction: float,
        perturbation_mode: str = 'morf',
        baseline_value: str = 'black',
        intensify: bool = True
    ) -> str:
        """
        Generates a plot showing the perturbed image alongside the original
        image with its relevance map overlaid.

        Args:
            filename (str): Base filename for the saved plot.
            image_tensor (torch.Tensor): The original input image tensor (B, C, H, W).
            relevance_map (torch.Tensor): The relevance map used to guide perturbation.
            patch_size (int): The size of square patches to perturb.
            perturbation_fraction (float): The fraction of patches to perturb (0.0 to 1.0).
            perturbation_mode (str): How to select patches.
                                     'morf': Most Relevant First.
                                     'lerf': Least Relevant First.
                                     'random': Random order.
            baseline_value (str): What to fill perturbed patches with.
                                  'black': Fills with zeros.
                                  'mean': Fills with the image's mean color.
        
        Returns:
            str: The full path to the saved visualization.
        """
        print(f"Plotting perturbation for {filename} with mode '{perturbation_mode}' and baseline '{baseline_value}'...")
        # --- 1. Perturbation Logic (Unchanged) ---
        patch_relevance = F.avg_pool2d(
            relevance_map,
            kernel_size=patch_size,
            stride=patch_size
        )
        patch_relevance_flat = patch_relevance.flatten()
        num_patches = len(patch_relevance_flat)

        if perturbation_mode == 'morf':
            order = torch.argsort(patch_relevance_flat, descending=True)
        elif perturbation_mode == 'lerf':
            order = torch.argsort(patch_relevance_flat, descending=False)
        elif perturbation_mode == 'random':
            order = deterministic_randperm(num_patches, filename, self.seed)
        else:
            raise ValueError(f"Unknown perturbation_mode: {perturbation_mode}")

        num_patches_to_perturb = int(np.floor(perturbation_fraction * num_patches))
        patches_to_perturb = order[:num_patches_to_perturb]

        perturbed_tensor = image_tensor.clone()

        if baseline_value.lower() == "black":
            baseline_fill = torch.zeros_like(image_tensor)
        elif baseline_value.lower() == "mean":
            mean_color = image_tensor.mean(dim=[1, 2], keepdim=True)
            baseline_fill = mean_color.expand_as(image_tensor)
        else:
            raise ValueError(f"Unknown baseline_value: {baseline_value}")

        h, w = image_tensor.shape[-2:]
        num_patches_w = w // patch_size
        for patch_idx in patches_to_perturb:
            row = (patch_idx // num_patches_w) * patch_size
            col = (patch_idx % num_patches_w) * patch_size
            perturbed_tensor[..., row:row+patch_size, col:col+patch_size] = \
                baseline_fill[..., row:row+patch_size, col:col+patch_size]

        # Prepare all necessary images for plotting
        perturbed_img_np = self._preprocess_image(perturbed_tensor)
        original_img_np = self._preprocess_image(image_tensor)
        
        # Create the relevance heatmap image using logic from plot_heatmap
        relevance_norm = relevance_map.squeeze() / torch.abs(relevance_map).max()
        #relevance_norm = torch.sign(relevance_norm ) * torch.abs(relevance_norm )**0.7
        if intensify:
            relevance_norm = torch.tanh(3 * relevance_norm)

        heatmap_img = imgify(relevance_norm, vmin=-1.0, vmax=1.0)

        # Create the plot
        fig, axs = plt.subplots(1, 3, figsize=(12, 6))
        
        title = (f"Perturbation Analysis: {perturbation_mode.upper()} "
                 f"({perturbation_fraction:.0%}, Patch Size: {patch_size})")
        fig.suptitle(title, fontsize=16)

        # Plot 1: Just the relevance map
        axs[0].imshow(heatmap_img)
        axs[0].set_title("Relevance Map")
        axs[0].axis('off')

        # Plot 2: The original image with relevance overlay
        axs[1].imshow(original_img_np)
        axs[1].imshow(heatmap_img, alpha=0.6)
        axs[1].set_title("Original with Relevance Overlay")
        axs[1].axis('off')

        # Plot 3: The perturbed image
        axs[2].imshow(perturbed_img_np)
        axs[2].set_title(f"Perturbed with '{baseline_value.title()}' Baseline")
        axs[2].axis('off')


        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle

        # --- 3. Save and Return (Unchanged) ---
        save_path = os.path.join(self.save_dir, f"{filename}_perturb_{perturbation_fraction}_{perturbation_mode}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return save_path
    
def main(cfg):
    monkey_patch_zennit(verbose=True)
    DEVICE = torch.device("cpu")
    MODE = cfg["lrp"]["mode"]
    print(f"\n--- RUNNING WITH MODE: {MODE} ---")

    DECISION_METRIC = MODE
    model_wrapper, image_transforms, model_data_config = get_model_wrapper(device=DEVICE, cfg=cfg["model"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    split_name = cfg["data"]["analysis_split"]
    split_dir = os.path.join(cfg["data"]["dataset_dir"], split_name)
    split_files = [f for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".png"))]
    model_type_str = "finetuned" if cfg["model"]["finetuned"] else "base"
    mask_transform = get_mask_transform(cfg["model"]["img_size"])

    dataset = GorillaReIDDataset(
        image_dir=split_dir,
        filenames=split_files,
        transform=image_transforms,
        base_mask_dir=cfg["data"]["base_mask_dir"],
        mask_transform=mask_transform
    )

    dataloader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"], num_workers=0, collate_fn=custom_collate_fn, shuffle=False)

    # --- 3. Compute k-NN Database ---
    db_path_knn = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name=dataset.dataset_name,
        split_name=split_name,
        db_dir=cfg["knn"]["db_embeddings_dir"]
    )
    db_embeddings, db_labels, db_filenames, db_video_ids = get_knn_db(
        db_path=db_path_knn,
        dataset=dataset,
        model_wrapper=model_wrapper,
        batch_size=cfg["data"]["batch_size"],
        device=DEVICE
    )

    # --- 4. Compute Relevances ---
    db_path_relevances = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name=dataset.dataset_name,
        split_name=split_name,
        db_dir=cfg["lrp"]["db_relevances_dir"],
        decision_metric=DECISION_METRIC
    )

    relevances_all = get_relevances(
        db_path=db_path_relevances,
        model_wrapper=model_wrapper,
        dataloader=dataloader,
        device=DEVICE,
        recompute=False,
        # All of these will be caught by **kwargs and passed to generate_relevances
        conv_gamma=cfg["lrp"]["conv_gamma"],           # Pass as single value (will be converted to list)
        lin_gamma=cfg["lrp"]["lin_gamma"],             # Pass as single value
        proxy_temp=cfg["knn"]["temp"],          # Pass as single value 
        distance_metric=cfg["knn"]["distance_metric"], #pass as single value
        topk_neg=cfg["knn"]["topk_neg"],  # Pass as single value
        mode=cfg["lrp"]["mode"],
        db_embeddings=db_embeddings,
        db_filenames=db_filenames,
        db_labels=db_labels,
        db_video_ids=db_video_ids,
        cross_video=cfg["lrp"]["cross_video"]
    )

    relevance_dict = {
        item['filename']: (item['relevance'], item['mask']) for item in relevances_all
    }
    print(f"length of relevance_dict: {len(relevance_dict)}")

    denorm_transform = get_denormalization_transform(mean=model_data_config['mean'], std=model_data_config['std'])

    visualizer = AttentionVisualizer(
        save_dir="./visualizations",
        denorm_transform=denorm_transform,
        seed=cfg["seed"]
    )

    # select 20 random images
    example_images = random.sample([os.path.splitext(f)[0] for f in split_files], 50)

    #print(f"Example images for visualization: {example_images}")

    fname_to_idx = {
        os.path.splitext(f)[0]: i for i, f in enumerate(dataset.filenames)
    }

    example_images = ["PL02_Tm002_20220706_015_2394_31676", "TU03_R118_20220912_096_42_851638"]
    for filename in example_images:

        # Get the necessary data for plotting
        if os.path.splitext(filename)[0] not in fname_to_idx:
            print(f"Warning: Filename '{filename}' not found in dataset. Skipping.")
            continue
            
        sample_data = dataset[fname_to_idx[os.path.splitext(filename)[0]]]
        image_tensor = sample_data["image"]
        
        relevance, mask = relevance_dict[filename]

        # Normalize relevance map for consistent visualization
        # This makes heatmaps comparable by scaling them to the [-1, 1] range
        #relevance = relevance / torch.abs(relevance).max()
        #relevance = torch.sign(relevance ) * torch.abs(relevance )**2
        frac = 0.75

        save_path = visualizer.plot_heatmap(
            filename=filename,
            image_tensor=image_tensor,
            mask=mask,
            relevance=relevance,
            stats={},
            intensify=True
        )

        """saved_path_morf = visualizer.plot_perturbation(
            filename=filename,
            image_tensor=image_tensor,
            relevance_map=relevance,
            patch_size=cfg["model"]["patch_size"],
            perturbation_fraction=frac, # Perturb the top 25% of patches
            perturbation_mode='morf',   # Most Relevant First
            baseline_value='mean'
        )
        print(f"MoRF perturbation plot saved to: {saved_path_morf}")

        saved_path_lerf = visualizer.plot_perturbation(
            filename=filename,
            image_tensor=image_tensor,
            relevance_map=relevance,
            patch_size=cfg["model"]["patch_size"],
            perturbation_fraction=frac,
            perturbation_mode='lerf',    # Least Relevant First
            baseline_value='mean'
        )
        print(f"LeRF perturbation plot saved to: {saved_path_lerf}")

        saved_path_lerf = visualizer.plot_perturbation(
            filename=filename,
            image_tensor=image_tensor,
            relevance_map=relevance,
            patch_size=cfg["model"]["patch_size"],
            perturbation_fraction=frac,
            perturbation_mode='random',    # Random
            baseline_value='mean'
        )
        print(f"Random perturbation plot saved to: {saved_path_lerf}")"""



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run faithfulness eval.")
    parser.add_argument(
        "--config_name", 
        type=str, 
        required=True,
        help="The name of the experiment config (e.g., 'finetuned', 'non_finetuned')."
    )   
    
    args, unknown_args = parser.parse_known_args()

    cfg = load_config(args.config_name, unknown_args)


    main(cfg)

