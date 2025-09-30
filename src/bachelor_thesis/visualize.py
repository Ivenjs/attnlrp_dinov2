import argparse
from collections import defaultdict
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Tuple, Union, Dict, Any
import torch.nn.functional as F
import json
from sklearn.model_selection import train_test_split

# Import imgify from zennit
from zennit.image import imgify

from torch.utils.data import DataLoader, ConcatDataset
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

    def plot_and_save_individual_overview(
        self,
        filename: str,
        image_tensor: torch.Tensor,
        mask: Union[torch.Tensor, np.ndarray],
        relevance: torch.Tensor,
        stats: Dict[str, Any],
        intensify: bool = False,
        show_stats: bool = False,
        category: str = "",
        heatmap_amp = 1.0,
        overlay_amp = 2.5,
    ) -> Dict[str, str]:
        """
        Generates and saves individual, themed plots for a single model and image.

        Each plot (original, mask, heatmap, overlay) is saved in a separate
        subdirectory within the main save directory.
        """
        img_np = self._preprocess_image(image_tensor)
        mask_np = self._preprocess_mask(mask)

        # Normalize relevance for visualization
        if torch.abs(relevance).max() < 1e-12:
            print(f"Empty relevance map for {filename}")
            relevance_norm = torch.zeros_like(relevance.squeeze())
        else:
            relevance_norm = relevance.squeeze() / torch.abs(relevance).max()

        relevance_intensified = torch.tanh(overlay_amp * relevance_norm)
        if intensify:
            relevance_norm = torch.tanh(heatmap_amp * relevance_norm)

        cmap = plt.get_cmap('coolwarm')
        norm_for_cmap = plt.Normalize(vmin=-1.0, vmax=1.0)
        heatmap_img = imgify(relevance_norm, vmin=-1.0, vmax=1.0)
        heatmap_intensified_img = imgify(relevance_intensified, vmin=-1.0, vmax=1.0)

        themes = {
            "original": "original_image",
            "masked": "mask_outline",
            "heatmap": "relevance_heatmap",
            "overlay": "relevance_overlay"
        }
        
        for theme_dir in themes.values():
            os.makedirs(os.path.join(self.save_dir, category, theme_dir), exist_ok=True)
            
        saved_paths = {}

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_np)
        #ax.set_title("Original Image")
        ax.axis('off')
        
        if show_stats:
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
            fig.text(0.01, 0.99, stats_text, fontsize=10, family='monospace',
                     va='top', ha='left', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))

        save_path = os.path.join(self.save_dir, category,themes["original"], f"{filename}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        saved_paths['original_image'] = save_path

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_np)
        ax.contour(mask_np, colors='lime', linewidths=1.5)
        #ax.set_title("Mask Outline")
        ax.axis('off')
        save_path = os.path.join(self.save_dir, category, themes["masked"], f"{filename}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        saved_paths['mask_outline'] = save_path

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(heatmap_img)
        #ax.set_title("Relevance Heatmap")
        ax.axis('off')
        mappable = plt.cm.ScalarMappable(norm=norm_for_cmap, cmap=cmap)
        fig.colorbar(mappable, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        save_path = os.path.join(self.save_dir, category, themes["heatmap"], f"{filename}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        saved_paths['relevance_heatmap'] = save_path
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(heatmap_intensified_img)
        ax.imshow(img_np, alpha=0.5)
        ax.contour(mask_np, colors='lime', linewidths=1.5)
        #ax.set_title("Relevance Overlay")
        ax.axis('off')
        save_path = os.path.join(self.save_dir, category, themes["overlay"], f"{filename}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        saved_paths['relevance_overlay'] = save_path

        return saved_paths
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
    

def get_intersected_categories(mode: str, is_zoo: bool) -> Dict[str, list]:
    category_to_filename = defaultdict(list)
    if not is_zoo:
        if mode == "proto_margin":
            category_to_filename["intersected_positive_lerf_flippers"] = [
                "DU40_R030_20211202_060_145_1172820",
                "GA41_R105_20220819_003_984_29862",
                "NN00_R018_20220711_050_2640_765937",
                "PL00_R103_20230204_008_1740_833501",
                "PL01_R185_20221218_099_642_137052",
            ]
            category_to_filename["intersected_robust_morf_successes"] = []
            category_to_filename["intersected_negative_morf_flippers"] = [
                "DU40_R030_20220325_020_21_1172823",
                "GA41_R105_20221226_204_636_838311",
                "NN00_R018_20220825_184_942_766159",
                "OE00_R019_20220824_127_42_240609",
                "TU03_R118_20221020_143_1590_261233",
            ]
            category_to_filename["intersected_hard_lerf_failures"] = [
                "PL02_R103_20230204_012_1578_833193",
                "RC21_R106_20221017_048_4100_1172721",
                "PL61_R465_20221029_008_5556_884817",
                "RC42_R108_20221127_293_7530_23115",
                "DU40_R030_20211202_060_3759_1172820",
            ]
        elif mode == "similarity":
            category_to_filename["intersected_positive_lerf_flippers"] = [
                "DU40_R030_20211202_064_1345_1172731",
                "OE00_R066_20211012_023_1024_1172780",
                "PL00_R103_20230227_095_48_288215",
                "PL61_R018_20220317_027_1128_1109080",
                "RC21_R108_20230128_153_1134_156578",
            ]
            category_to_filename["intersected_robust_morf_successes"] = []
            category_to_filename["intersected_negative_morf_flippers"] = [
                "DU40_R030_20211202_060_2687_1172820",
                "GA41_R105_20220819_003_1806_29862",
                "NN00_R018_20220825_184_1032_766159",
                "OE00_R019_20220512_133_5874_29277",
                "PL00_R018_20220317_027_4854_1109074",
            ]
            category_to_filename["intersected_hard_lerf_failures"] = [
                "PL61_R066_20220911_147_1350_830037",
                "PL02_Tm002_20220706_015_2502_31676",
                "PL01_R465_20221029_008_4128_884818",
                "OE00_R172_20221121_220_1026_220799",
                "NN00_R019_20221229_016_1620_774122",
            ]
        elif mode == "soft_knn_margin_all":
            category_to_filename["intersected_positive_lerf_flippers"] = [
                "DU40_R030_20220325_020_240_1172823",
                "GA41_R105_20220827_045_6030_836592",
                "NN00_R019_20221229_016_1578_774122",
                "OE00_R066_20211012_023_735_1172780",
                "PL00_R103_20230204_008_1320_833501",
            ]
            category_to_filename["intersected_robust_morf_successes"] = []
            category_to_filename["intersected_negative_morf_flippers"] = [
                "GA41_R105_20220819_003_1332_29862",
                "DU40_R030_20220325_020_79_1172823",
                "ME00_R465_20210924_023_100_1172856",
                "NN00_R019_20220107_235_606_604744",
                "OE00_R019_20220512_133_5280_29277",
            ]
            category_to_filename["intersected_hard_lerf_failures"] = [
                "RC42_R108_20221127_293_7530_23115",
                "RC21_R108_20230128_018_2634_847788",
                "PL61_R465_20221029_008_4380_884817",
                "PL02_Tm002_20220706_015_2316_31676",
                "OE00_R066_20220116_017_108_661909",
            ]
        
        elif mode == "soft_knn_margin_topk":
            category_to_filename["intersected_positive_lerf_flippers"] = [
                "PL00_R465_20220926_114_1818_884067",
                "PL01_R465_20220425_164_2094_878782",
                "PL02_R465_20221101_285_3234_886148",
                "RC21_R106_20221017_048_930_1172721",
                "RC42_R105_20230201_327_2220_329211",
            ]
            category_to_filename["intersected_robust_morf_successes"] = []
            category_to_filename["intersected_negative_morf_flippers"] = [
                "DU40_R030_20220325_020_150_1172823",
                "GA41_R185_20221030_060_216_870310",
                "ME00_R465_20210924_023_344_1172856",
                "NN00_R019_20220107_261_1224_1107407",
                "OE00_R019_20220512_133_5430_29277",
            ]
            category_to_filename["intersected_hard_lerf_failures"] = [
                "DU40_R030_20211202_060_3752_1172820",
                "NN00_R465_20220905_302_7230_885099",
                "PL00_R465_20220425_210_2310_878890",
                "PL01_R018_20220818_144_1002_765597",
                "RC21_R108_20230128_018_2760_847788",
            ]
    else:
        if mode == "similarity" or mode == "proto_margin":
            print(f"mode '{mode}' not supported for Zoo dataset.")

        if mode == "soft_knn_margin_all":
            category_to_filename["intersected_positive_lerf_flippers"] = [
                "Bibi_zoo_114_114_135_114Bibi",
                "Djambala_zoo_107_107_316_107Djambala",
                "MPenzi_zoo_136_136_269_136MPenzi",
                "Sango_zoo_39_39_2034_39Sango",
                "Tilla_zoo_101_101_456_101Tilla",
            ]
            category_to_filename["intersected_robust_morf_successes"] = []
            category_to_filename["intersected_negative_morf_flippers"] = [
                "Bibi_zoo_108_108_371_108Bibi",
                "Djambala_zoo_145_145_1080_145Djambala",
                "MPenzi_zoo_42_42_1516_42MPenzi",
                "Sango_zoo_109_109_308_109Sango",
                "Tilla_zoo_29_29_1128_29Tilla",
            ]
            category_to_filename["intersected_hard_lerf_failures"] = [
                "Bibi_zoo_100_100_1285_100Bibi",
                "Djambala_zoo_149_149_31_149Djambala",
                "MPenzi_zoo_65_65_361_65MPenzi",
                "Sango_zoo_23_23_612_23Sango",
                "Tilla_zoo_29_29_1128_29Tilla",
            ]
        elif mode == "soft_knn_margin_topk":
            category_to_filename["intersected_positive_lerf_flippers"] = [
                "Bibi_zoo_100_100_928_100Bibi",
                "Djambala_zoo_141_141_697_141Djambala",
                "MPenzi_zoo_59_59_358_59MPenzi",
                "Sango_zoo_13_13_487_13Sango",
                "Tilla_zoo_16_16_1051_16Tilla",
            ]
            category_to_filename["intersected_robust_morf_successes"] = []
            category_to_filename["intersected_negative_morf_flippers"] = [
                "Bibi_zoo_88_88_800_88Bibi",
                "Djambala_zoo_101_101_1366_101Djambala",
                "MPenzi_zoo_12_12_1543_12MPenzi",
                "MPenzi_zoo_30_30_2738_30MPenzi",
                "Sango_zoo_109_109_220_109Sango",
            ]
            category_to_filename["intersected_hard_lerf_failures"] = [
                "Bibi_zoo_53_53_419_53Bibi",
                "Djambala_zoo_145_145_296_145Djambala",
                "MPenzi_zoo_111_111_248_111MPenzi",
                "Sango_zoo_23_23_2669_23Sango",
                "Tilla_zoo_36_36_556_36Tilla",
            ]
    
    return category_to_filename

def sample_nonzero_relevance_diverse(filenames, relevance_dict, n=5):
    """
    Sample up to `n` filenames with non-zero relevance, trying to maximize class diversity.
    Class is determined by the first part of the filename: filename.split('_')[0].
    """
    valid_filenames = [f for f in filenames if f in relevance_dict and torch.abs(relevance_dict[f][0]).max() > 1e-12]
    
    class_to_files = defaultdict(list)
    for f in valid_filenames:
        class_name = f.split('_')[0]
        class_to_files[class_name].append(f)
    
    sampled = []
    classes = list(class_to_files.keys())
    random.shuffle(classes)
    
    while len(sampled) < n and classes:
        for cls in classes[:]:
            if class_to_files[cls]:
                chosen = random.choice(class_to_files[cls])
                sampled.append(chosen)
                class_to_files[cls].remove(chosen)
            if not class_to_files[cls]:
                classes.remove(cls)
            if len(sampled) >= n:
                break
    return sampled


def visualize_prediction_with_neighbors(
    prediction_info: Dict[str, Any],
    category: str,
    full_dataset: ConcatDataset,
    full_fname_to_idx: Dict[str, int],
    relevance_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    visualizer: AttentionVisualizer
):
    """
    Visualizes a query image and its top-k neighbors from a prediction entry.

    Args:
        prediction_info: A dictionary for a single prediction from the JSON file.
        category: The category of the prediction ('correct' or 'incorrect').
        full_dataset: The combined dataset to look up image tensors.
        full_fname_to_idx: A mapping from filename to index in the full_dataset.
        relevance_dict: Dictionary containing pre-computed relevance maps.
        visualizer: The AttentionVisualizer instance.
    """
    query_filename_ext = prediction_info["filename"]
    query_filename_base = os.path.splitext(query_filename_ext)[0]
    predicted_label = prediction_info["predicted_label"]
    
    print(f"\n--- Visualizing {category.upper()} prediction for: {query_filename_base} ---")

    neighbor_filenames_ext = prediction_info.get("top_k_neighbor_filenames", [])
    neighbor_filenames_base = [os.path.splitext(f)[0] for f in neighbor_filenames_ext]

    # Combine the query and its neighbors into a single list for processing
    all_filenames_to_plot = [query_filename_base] + neighbor_filenames_base

    # Create a unique prefix for this group of images to keep them together
    group_prefix = f"{category}_{query_filename_base}_pred_{predicted_label}"

    for i, filename in enumerate(all_filenames_to_plot):
        if i == 0:
            rank_label = "rank00_query"
        else:
            rank_label = f"rank{i:02d}_neighbor"

        # Construct a descriptive filename for saving
        # e.g., correct_DU40..._pred_DU40_rank00_query_DU40...
        save_filename = f"{group_prefix}_{rank_label}_{filename}"
        
        # --- Data Fetching ---
        if filename not in full_fname_to_idx:
            print(f"  [Warning] Filename '{filename}' not found in dataset. Skipping.")
            continue
        
        if filename not in relevance_dict:
            print(f"  [Warning] Relevance not found for '{filename}'. Skipping.")
            continue
            
        print(f"  -> Plotting {rank_label}: {filename}")

        # Get image tensor and mask from the dataset
        sample_idx = full_fname_to_idx[filename]
        sample_data = full_dataset[sample_idx]
        image_tensor = sample_data["image"]
        
        # Get relevance map from the pre-computed dictionary
        # Note: The mask from relevance_dict might be more accurate if it was 
        # computed on the same image transform, but using the one from the dataset
        # is also fine. Here we use the one from relevance_dict.
        relevance, cached_mask = relevance_dict[filename]

        if cached_mask is None:
            mask_to_use = sample_data["mask"] 
        else:
            mask_to_use = cached_mask

        # --- Visualization Call ---
        visualizer.plot_and_save_individual_overview(
            filename=save_filename,
            image_tensor=image_tensor,
            mask=mask_to_use,
            relevance=relevance,
            stats={},  # You can add stats here if available
            intensify=False,
            category=category, # Saves to the correct subdirectory (e.g., .../correct/)
        )

def main(cfg):
    monkey_patch_zennit(verbose=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    MODE = cfg["lrp"]["mode"]
    print(f"\n--- RUNNING WITH MODE: {MODE} ---")

    is_finetuned = cfg["model"]["finetuned"]
    model_type_str = "finetuned" if is_finetuned else "base"
    print(f"--- Running experiment for: {model_type_str.upper()} MODEL ---")


    model_wrapper, image_transforms, model_data_config = get_model_wrapper(device=DEVICE, cfg=cfg["model"])
    mask_transform = get_mask_transform(cfg["model"]["img_size"])

    # --- Prepare Datasets ---
    dataset_dir = cfg["data"]["dataset_dir"]
    if not "zoo" in dataset_dir:
        split_name = cfg["data"]["analysis_split"]
        split_dir = os.path.join(dataset_dir, split_name)
        split_files = [f for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".png"))]
        
        split_dataset = GorillaReIDDataset(
            image_dir=split_dir,
            filenames=split_files,
            transform=image_transforms,
            base_mask_dir=cfg["data"]["base_mask_dir"],
            mask_transform=mask_transform,
            k=cfg["knn"]["k"],
        )
        
        train_dir = os.path.join(cfg["data"]["dataset_dir"], "train")
        train_files = [f for f in os.listdir(train_dir) if f.lower().endswith((".jpg", ".png"))]
        train_dataset = GorillaReIDDataset(
            image_dir=train_dir, filenames=train_files, transform=image_transforms
        )
        

        datasets = [split_dataset, train_dataset]
        full_db_dataset = ConcatDataset(datasets)
        full_dataset_splits = "+".join([os.path.basename(d.image_dir) for d in datasets])


        query_dataset_offset = 0
        found = False
        for d in datasets:
            if d is split_dataset:
                found = True
                break
            query_dataset_offset += len(d)

        print("Query dataset offset in DB:", query_dataset_offset)

        if not found:
            raise RuntimeError("Query dataset (split_dataset) not found in db_constituents.")
        
        all_files_in_order = []
        for ds in full_db_dataset.datasets:
            all_files_in_order.extend(ds.filenames)
        
    else:
        print("Using Zoo dataset for evaluation.")
        all_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith((".jpg", ".png"))]

        subsample_fraction = cfg["data"].get("zoo_subsample_fraction", 1.0)

        if subsample_fraction < 1.0:
            print(f"Subsampling Zoo dataset to {subsample_fraction:.0%} of its original size.")
            labels = [f.split('_')[0] for f in all_files]

            discard_fraction = 1.0 - subsample_fraction

            subsampled_files, _ = train_test_split(
                all_files,
                test_size=discard_fraction,
                stratify=labels,
                random_state=cfg["seed"]
            )
            
            split_name_suffix = f"_subsampled_{int(subsample_fraction*100)}pct"

        else:
            print("Using the full Zoo dataset (no subsampling).")
            subsampled_files = all_files
            split_name_suffix = "_full"


        print(f"Using {len(subsampled_files)} images for the Zoo evaluation.")

        split_dataset = GorillaReIDDataset(
            image_dir=dataset_dir,
            filenames=subsampled_files,  
            transform=image_transforms,
            base_mask_dir=cfg["data"]["base_mask_dir"],
            mask_transform=mask_transform,
            k=cfg["knn"]["k"],
        )

        query_dataset_offset = 0
        full_db_dataset = split_dataset
        full_dataset_splits = os.path.basename(dataset_dir) + split_name_suffix
        split_name = full_dataset_splits
        split_files = subsampled_files
        all_files_in_order = split_dataset.filenames


    full_fname_to_idx = {os.path.splitext(f)[0]: i for i, f in enumerate(all_files_in_order)}

    db_path = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name=split_dataset.dataset_name,
        split_name=full_dataset_splits,
        bp_transforms=cfg["model"]["bp_transforms"],
        db_dir=cfg["knn"]["db_embeddings_dir"]
    )
    all_db_embeddings, all_db_labels, all_db_filenames, all_db_videos = get_knn_db(
        db_path=db_path,
        dataset=full_db_dataset,
        model_wrapper=model_wrapper,
        batch_size=cfg["data"]["batch_size"],
        device=DEVICE
    )

    local_query_indices = split_dataset.images_for_ce_knn

    split_db_path_knn = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name=split_dataset.dataset_name, split_name=split_name, bp_transforms=cfg["model"]["bp_transforms"], db_dir=cfg["knn"]["db_embeddings_dir"]
    )
    split_embeddings, split_labels, split_filenames, split_video_ids = get_knn_db(
        db_path=split_db_path_knn, dataset=split_dataset, model_wrapper=model_wrapper,
        batch_size=cfg["data"]["batch_size"], device=DEVICE
    )

    split_query_subset = torch.utils.data.Subset(split_dataset, local_query_indices)

    split_dataloader = DataLoader(split_query_subset, batch_size=cfg["data"]["batch_size"], num_workers=0, shuffle=False, collate_fn=custom_collate_fn)

    if cfg["lrp"]["eval_db"] == "test":
        relevance_split_name = split_name
        relevance_db_embeddings = split_embeddings
        relevance_db_labels = split_labels
        relevance_db_filenames = split_filenames
        relevance_db_videos = split_video_ids
    elif cfg["lrp"]["eval_db"] == "all":
        relevance_split_name = full_dataset_splits
        relevance_db_embeddings = all_db_embeddings
        relevance_db_labels = all_db_labels
        relevance_db_filenames = all_db_filenames
        relevance_db_videos = all_db_videos

    # --- Get path for current model's relevances ---
    db_path_relevances = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name=split_dataset.dataset_name, 
        split_name=relevance_split_name, 
        bp_transforms=cfg["model"]["bp_transforms"], 
        db_dir=cfg["lrp"]["db_relevances_dir"],
        decision_metric=MODE,
        lrp_params={
            "conv_gamma": cfg["lrp"]["conv_gamma"],
            "lin_gamma": cfg["lrp"]["lin_gamma"],
            "proxy_temp": cfg["lrp"]["temp"],
            "topk": cfg["lrp"]["topk"],
        }
    )
    
    relevances_all = get_relevances(
        db_path=db_path_relevances, 
        model_wrapper=model_wrapper, 
        dataloader=split_dataloader,
        device=DEVICE, 
        recompute=False, 
        conv_gamma=cfg["lrp"]["conv_gamma"], 
        lin_gamma=cfg["lrp"]["lin_gamma"],
        proxy_temp=cfg["lrp"]["temp"], 
        distance_metric=cfg["lrp"]["distance_metric"], 
        mode=cfg["lrp"]["mode"],
        topk=cfg["lrp"]["topk"], 
        db_embeddings=relevance_db_embeddings, 
        db_filenames=relevance_db_filenames,
        db_labels=relevance_db_labels, 
        db_video_ids=relevance_db_videos, 
        cross_encounter=cfg["lrp"]["cross_encounter"]
    )

    relevance_dict = {
        item['filename']: (item['relevance'], item['mask']) for item in relevances_all
    }

    print(f"Loaded relevance maps for {len(relevance_dict)} images.")

    denorm_transform = get_denormalization_transform(mean=model_data_config['mean'], std=model_data_config['std'])

    if "zoo" in cfg["data"]["dataset_dir"].lower():
        save_dir = f"./visualizations_zoo/{model_type_str}/{cfg['lrp']['mode']}"
    else:
        save_dir = f"./visualizations/{model_type_str}/{cfg['lrp']['mode']}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    visualizer = AttentionVisualizer(
        save_dir=save_dir,
        denorm_transform=denorm_transform,
        seed=cfg["seed"]
    )

    base_path = "/sc/home/iven.schlegelmilch/bachelor_thesis_code" #base_zoo_predictions.json

    if "zoo" in cfg["data"]["dataset_dir"].lower():
        filename = model_type_str + "_zoo_predictions.json"
        prediction_json_path = os.path.join(base_path, filename)
    else:
        filename = model_type_str + "_predictions.json"
        prediction_json_path = os.path.join(base_path, filename)

    prediction_data = None
    correct_prediction_lookup = {}
    incorrect_prediction_lookup = {}

    if os.path.exists(prediction_json_path):
        with open(prediction_json_path, 'r') as f:
            prediction_data = json.load(f)
        
        # Create efficient lookup maps (filename without extension -> prediction data)
        correct_prediction_lookup = {
            os.path.splitext(p['filename'])[0]: p 
            for p in prediction_data.get('correct_predictions', [])
        }
        incorrect_prediction_lookup = {
            os.path.splitext(p['filename'])[0]: p 
            for p in prediction_data.get('incorrect_predictions', [])
        }
        print(f"Loaded {len(correct_prediction_lookup)} correct and {len(incorrect_prediction_lookup)} incorrect predictions for neighbor lookup.")
    else:
        print(f"Warning: Prediction JSON not found at '{prediction_json_path}'. Skipping neighbor visualizations.")

    

    random_images = random.sample([os.path.splitext(f)[0] for f in split_files], 5)
    images_to_visualize = defaultdict(list)

    images_to_visualize["random"] = random_images

    if "zoo" in cfg["data"]["dataset_dir"].lower():
        images_to_visualize["correct_with_neighbors"] = ["Sango_zoo_98_98_1555_98Sango"]
        images_to_visualize["incorrect_with_neighbors"] = ["Bibi_zoo_51_51_2779_51Bibi"]
    else:
        images_to_visualize["correct_with_neighbors"] = ["GA41_R105_20220819_003_1650_29862"]
        images_to_visualize["incorrect_with_neighbors"] = ["PL02_R465_20220228_205_54_724627"]

    analysis_json_path = f"./visualizations/{os.path.basename(db_path_relevances)}.json"
    if os.path.exists(analysis_json_path):
        with open(analysis_json_path, 'r') as f:
            analysis_data = json.load(f)
    else:
        analysis_data = {}

    """for category, filenames in analysis_data.items():
        if category not in images_to_visualize:
            images_to_visualize[category] = []
        images_to_visualize[category].extend(
            sample_nonzero_relevance_diverse(filenames, relevance_dict, n=5) #this is dependant on base/finetuned model anyway. so no point in trying to have the same images
        )"""

    is_zoo = "zoo" in cfg["data"]["dataset_dir"].lower()
    intersected_categories = get_intersected_categories(mode=MODE, is_zoo=is_zoo)
    images_to_visualize.update(intersected_categories)



    fname_to_idx = {
        os.path.splitext(f)[0]: i for i, f in enumerate(split_dataset.filenames)
    }

    if prediction_data:
        # Process CORRECT examples
        for filename in images_to_visualize["correct_with_neighbors"]:
            prediction_info = correct_prediction_lookup.get(filename)
            if prediction_info:
                visualize_prediction_with_neighbors(
                    prediction_info=prediction_info,
                    category="correct", # This determines the subfolder
                    full_dataset=full_db_dataset,
                    full_fname_to_idx=full_fname_to_idx,
                    relevance_dict=relevance_dict,
                    visualizer=visualizer,
                )
            else:
                print(f"[Warning] Showcase file '{filename}' not found in the 'correct_predictions' list of your JSON.")
        
        # Process INCORRECT examples
        for filename in images_to_visualize["incorrect_with_neighbors"]:
            prediction_info = incorrect_prediction_lookup.get(filename)
            if prediction_info:
                visualize_prediction_with_neighbors(
                    prediction_info=prediction_info,
                    category="incorrect", # This determines the subfolder
                    full_dataset=full_db_dataset,
                    full_fname_to_idx=full_fname_to_idx,
                    relevance_dict=relevance_dict,
                    visualizer=visualizer,
                )
            else:
                print(f"[Warning] Showcase file '{filename}' not found in the 'incorrect_predictions' list of your JSON.")
    special_categories = ["correct_with_neighbors", "incorrect_with_neighbors"]

    for category, filenames in images_to_visualize.items():
        if category in special_categories:
            continue
        for filename in filenames:
            if os.path.splitext(filename)[0] not in fname_to_idx:
                print(f"Warning: Filename '{filename}' not found in dataset. Skipping.")
                continue
            
            sample_data = split_dataset[fname_to_idx[os.path.splitext(filename)[0]]]
            image_tensor = sample_data["image"]
            
            if filename not in relevance_dict:
                print(f"Warning: Relevance not found for '{filename}'. Skipping.")
                continue

            relevance, cached_mask = relevance_dict[filename]

            if cached_mask is None:
                mask_to_use = sample_data["mask"] 
            else:
                mask_to_use = cached_mask

            visualizer.plot_and_save_individual_overview(
                filename=filename,
                image_tensor=image_tensor,
                mask=mask_to_use,
                relevance=relevance,
                stats={},
                intensify=False,
                category=category,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LRP relevance visualization.")
    parser.add_argument(
        "--config_name", 
        type=str, 
        required=True,
        help="The name of the experiment config (e.g., 'finetuned', 'non_finetuned')."
    )   
    
    args, unknown_args = parser.parse_known_args()
    cfg = load_config(args.config_name, unknown_args)
    main(cfg)