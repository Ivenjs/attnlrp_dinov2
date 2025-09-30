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
        ax.set_title("Original Image")
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
        ax.set_title("Mask Outline")
        ax.axis('off')
        save_path = os.path.join(self.save_dir, category, themes["masked"], f"{filename}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        saved_paths['mask_outline'] = save_path

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(heatmap_img)
        ax.set_title("Relevance Heatmap")
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
        ax.set_title("Relevance Overlay")
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
    split_name = cfg["data"]["analysis_split"]
    split_dir = os.path.join(cfg["data"]["dataset_dir"], split_name)
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

    db_path = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name=train_dataset.dataset_name,
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
        dataset_name=train_dataset.dataset_name, 
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

    # <<< MODIFICATION START: QUICK & DIRTY INTERSECTION LOGIC >>>
    print("\n--- Finding intersection of valid relevances between models ---")


    base_model_checkpoint = "base"
    finetuned_model_checkpoint = "/workspaces/vast-gorilla/gorillawatch/models/ViTG-body_face-spac23+24v3-dedup.pth"

    # 2. GET DB PATH FOR THE "OTHER" MODEL
    other_model_path = base_model_checkpoint if is_finetuned else finetuned_model_checkpoint
    
    print(f"Current model is: {model_type_str.upper()}")
    
    # The `db_path_relevances` for the current model is already computed.
    db_path_relevances_current = db_path_relevances
    
    # Now compute the path for the other model, keeping all other params the same.
    # We create a simplified lrp_params dict to avoid potential key errors if cfg differs.
    simple_lrp_params = {
        "conv_gamma": cfg["lrp"]["conv_gamma"],
        "lin_gamma": cfg["lrp"]["lin_gamma"],
        "proxy_temp": cfg["lrp"]["temp"],
        "topk": cfg["lrp"]["topk"],
    }
    db_path_relevances_other = get_db_path(
        model_checkpoint_path=other_model_path, # <-- The only change is here
        dataset_name=train_dataset.dataset_name, 
        split_name=relevance_split_name, 
        bp_transforms=cfg["model"]["bp_transforms"], 
        db_dir=cfg["lrp"]["db_relevances_dir"],
        decision_metric=MODE,
        lrp_params=simple_lrp_params,
    )

    # 3. HELPER FUNCTION TO GET VALID FILENAMES
    def get_valid_filenames_from_db(db_path):
        if not os.path.exists(db_path):
            print(f"WARNING: Relevance DB not found at {db_path}")
            return set()
        print(f"Loading relevances from: {db_path}")
        relevance_data = torch.load(db_path, map_location='cpu') # Load to CPU to save GPU memory
        valid_files = {
            item['filename'] for item in relevance_data 
            if torch.abs(item['relevance']).max() > 1e-12
        }
        print(f" -> Found {len(valid_files)} files with non-zero relevance.")
        return valid_files

    # 4. GET THE INTERSECTION
    valid_files_current = get_valid_filenames_from_db(db_path_relevances_current)
    valid_files_other = get_valid_filenames_from_db(db_path_relevances_other)
    
    safe_filenames_for_comparison = valid_files_current.intersection(valid_files_other)
    print(f"\nFound {len(safe_filenames_for_comparison)} files with valid relevance in BOTH models.")
    # <<< MODIFICATION END >>>
    
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

    visualizer = AttentionVisualizer(
        save_dir=f"./visualizations/{model_type_str}/{cfg['lrp']['mode']}",
        denorm_transform=denorm_transform,
        seed=cfg["seed"]
    )

    images_to_visualize = defaultdict(list)
    all_base_filenames = [os.path.splitext(f)[0] for f in split_files]

    # <<< MODIFICATION: Filter the random sampling pool >>>
    # Create a pool of random candidates that are guaranteed to have valid relevance in BOTH models.
    safe_random_pool = [fname for fname in all_base_filenames if fname in safe_filenames_for_comparison]
    print(f"\nCreating a 'random' sample pool with {len(safe_random_pool)} images valid for both models.")
    
    images_to_visualize["random"] = sample_nonzero_relevance_diverse(safe_random_pool, relevance_dict, n=5)

    images_to_visualize["correct"] = [
        "TU03_R118_20220912_096_1860_851638",
        "PL02_R465_20220422_128_678_18108",
        "PL61_R465_20220926_114_2868_884075",
        "PL01_R465_20220629_005_1272_882125",
        "NN00_R018_20220825_184_336_766159",
    ]

    images_to_visualize["incorrect"] = [
        "AP03_R198_20220929_158_112_1172980",
        "OE00_Tm002_20220227_002_180_912465",
        "DU40_R030_20211202_064_3217_1172731",
        "GA41_R105_20221226_204_384_838311",
        "ME00_R465_20211104_151_450_1172834",
    ]

    analysis_json_path = f"./visualizations/{os.path.basename(db_path_relevances)}.json"
    if os.path.exists(analysis_json_path):
        with open(analysis_json_path, 'r') as f:
            analysis_data = json.load(f)
    else:
        analysis_data = {}

    for category, filenames in analysis_data.items():
        if category not in images_to_visualize:
            images_to_visualize[category] = []
        images_to_visualize[category].extend(
            sample_nonzero_relevance_diverse(filenames, relevance_dict, n=5)
        )

    # <<< MODIFICATION START: FILTER PAIRED DATA BEFORE SAMPLING >>>
    paired_json_path = f"./visualizations/{MODE}_PAIRED_intersection.json"
    if os.path.exists(paired_json_path):
        with open(paired_json_path, 'r') as f:
            paired_data = json.load(f)
            
        print("\nFiltering analysis data with the safe intersection set...")
        safe_paired_data = {}
        for category, filenames in paired_data.items():
            filtered_list = [fname for fname in filenames if fname in safe_filenames_for_comparison]
            print(f"Category '{category}': {len(filenames)} original -> {len(filtered_list)} safe files.")
            safe_paired_data[category] = filtered_list
        
        # Now use the filtered data for sampling
        for category, filenames in safe_paired_data.items():
            if category not in images_to_visualize:
                images_to_visualize[category] = []
            images_to_visualize[category].extend(
                fname for fname in sample_nonzero_relevance_diverse(filenames, relevance_dict, n=5)
                if fname not in images_to_visualize[category]
            )
    else:
        print(f"WARNING: Paired JSON file not found at {paired_json_path}")
    # <<< MODIFICATION END >>>

    fname_to_idx = {
        os.path.splitext(f)[0]: i for i, f in enumerate(split_dataset.filenames)
    }

    for category, filenames in images_to_visualize.items():
        for filename in filenames:
            if os.path.splitext(filename)[0] not in fname_to_idx:
                print(f"Warning: Filename '{filename}' not found in dataset. Skipping.")
                continue
            
            sample_data = split_dataset[fname_to_idx[os.path.splitext(filename)[0]]]
            image_tensor = sample_data["image"]
            
            if filename not in relevance_dict:
                print(f"Warning: Relevance not found for '{filename}'. Skipping.")
                continue
            relevance, mask = relevance_dict[filename]

            visualizer.plot_and_save_individual_overview(
                filename=filename,
                image_tensor=image_tensor,
                mask=mask,
                relevance=relevance,
                stats={},
                intensify=True,
                category=category,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LRP relevance visualization.")
    parser.add_argument(
        "--config_name", 
        type=str, 
        required=True,
        help="The name of the experiment config (e.g., 'finetuned', 'base')."
    )   
    
    args, unknown_args = parser.parse_known_args()
    cfg = load_config(args.config_name, unknown_args)
    main(cfg)