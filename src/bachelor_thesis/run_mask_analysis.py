from utils import get_db_path, get_mask_transform, load_config
from dataset import GorillaReIDDataset, custom_collate_fn
from lxt.efficient import monkey_patch_zennit
from basemodel import get_model_wrapper
from knn_helpers import get_knn_db
from typing import Dict, Tuple, List
from lrp_helpers import get_relevances
from torch.utils.data import DataLoader, ConcatDataset
from eval_helpers import attention_inside_mask
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf
import subprocess
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
import os
import random
import torch
import argparse 
import wandb
from collections import defaultdict
import pandas as pd

def get_combined_analysis_categories(
    base_analysis_path: str,
    predictions_json_path: str
) -> Dict[str, List[str]]:
    """
    Creates a new analysis JSON by combining and filtering two sources.
    It robustly normalizes all filenames by removing their extensions.
    """
    print(f"\n--- Combining analysis categories ---")

    def normalize_filename(fname: str) -> str:
        return os.path.splitext(fname)[0]

    combined_categories = {}
    categories_to_exclude = {"negative_morf_flippers", "robust_morf_successes"}

    try:
        with open(base_analysis_path, 'r') as f:
            base_data = json.load(f)
        
        print(f"Loaded base analysis from: {base_analysis_path}")
        for category, filenames in base_data.items():
            if category not in categories_to_exclude:
                normalized_filenames = [normalize_filename(f) for f in filenames]
                combined_categories[category] = normalized_filenames
                print(f"  - Kept and normalized category '{category}' with {len(normalized_filenames)} images.")
            else:
                print(f"  - Excluded category '{category}'.")

    except FileNotFoundError:
        print(f"Warning: Base analysis file not found at '{base_analysis_path}'. Continuing without it.")
    except Exception as e:
        print(f"Error reading base analysis file '{base_analysis_path}': {e}")
        return {} 

    try:
        with open(predictions_json_path, 'r') as f:
            predictions_data = json.load(f)
        
        print(f"Loaded predictions from: {predictions_json_path}")
        
        correct_filenames = [normalize_filename(item['filename']) for item in predictions_data.get('correct_predictions', [])]
        incorrect_filenames = [normalize_filename(item['filename']) for item in predictions_data.get('incorrect_predictions', [])]

        combined_categories["correct_predictions"] = correct_filenames
        combined_categories["incorrect_predictions"] = incorrect_filenames
        print(f"  - Added and normalized category 'correct_predictions' with {len(correct_filenames)} images.")
        print(f"  - Added and normalized category 'incorrect_predictions' with {len(incorrect_filenames)} images.")

    except FileNotFoundError:
        print(f"Error: Predictions JSON not found at '{predictions_json_path}'. Cannot add new categories.")
        return {} 
    except Exception as e:
        print(f"Error reading predictions file '{predictions_json_path}': {e}")
        return {}

    print("Successfully created combined analysis categories in memory.")
    return combined_categories

def _get_relevance_component(relevance_map: np.ndarray, component: str) -> np.ndarray:
    """
    Isolates a specific component of the relevance map.

    Args:
        relevance_map: The original relevance map.
        component: One of 'absolute', 'positive', or 'negative'.

    Returns:
        The processed relevance map component. For 'negative', it returns the absolute values.
    """
    if component == 'absolute':
        return np.abs(relevance_map)
    elif component == 'positive':
        return np.maximum(0, relevance_map)
    elif component == 'negative':
        return np.abs(np.minimum(0, relevance_map))
    else:
        raise ValueError(f"Unknown relevance component: {component}")

def analyze_era_in_mask_ratio(
    dataloader: DataLoader,
    relevance_dict: Dict[str, torch.Tensor],
    analysis_categories: Dict[str, List[str]],
    thresholds: List[float] = [0.10, 0.25, 0.50]
) -> Dict[str, Dict[str, float]]:
    """
    Measures the "Hotspot Placement Accuracy" for absolute, positive, and negative relevance.

    This calculates the percentage of the "Effective Relevance Area" (the hotspot)
    that falls correctly inside the segmentation mask for each relevance type.

    Args:
        dataloader: Dataloader that provides images, masks, and filenames.
        relevance_dict: Dictionary of filenames to their relevance maps.
        thresholds: List of fractions of the max relevance to define hotspots.

    Returns:
        A dictionary with aggregated statistics for each category and relevance type.
    """
    print(f"\n--- Running Hotspot Placement Accuracy (ERA-in-Mask) Analysis ---")
    analysis_results = analysis_categories
    if not analysis_results:
        print("Error: Received empty or invalid analysis categories. Skipping mask analysis.")
        return {}

    filename_to_categories = defaultdict(list)
    for category, filenames in analysis_categories.items():
        for filename in filenames:
            filename_to_categories[filename].append(category)

    category_ratios = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    baseline_category_name = "all_queries_baseline"
    relevance_components = ['absolute', 'positive', 'negative']

    skipped_masks = 0
    for batch in tqdm(dataloader, desc="Analyzing ERA-in-Mask"):
        masks_batch = batch["mask"]
        filenames_batch = batch["filename"]

        for i, filename in enumerate(filenames_batch):
            relevance_map_orig = relevance_dict.get(filename)
            if relevance_map_orig is None:
                continue

            if masks_batch[i] is None:
                skipped_masks += 1
                continue

            if isinstance(relevance_map_orig, torch.Tensor):
                relevance_map_orig = relevance_map_orig.cpu().numpy()
            mask_np = masks_batch[i].cpu().numpy().astype(bool)

            for component in relevance_components:
                relevance_component_map = _get_relevance_component(relevance_map_orig, component)
                max_relevance = np.max(relevance_component_map)

                if max_relevance < 1e-7:
                    continue

                for t in thresholds:
                    threshold_value = t * max_relevance
                    hotspot_mask = relevance_component_map > threshold_value
                    
                    total_hotspot_area = np.sum(hotspot_mask)

                    if total_hotspot_area == 0:
                        ratio = np.nan
                    else:
                        intersection_area = np.sum(hotspot_mask & mask_np)
                        ratio = intersection_area / total_hotspot_area
                    
                    category_ratios[baseline_category_name][component][t].append(ratio)
                    categories_for_file = filename_to_categories.get(filename, [])
                    for category in categories_for_file:
                        category_ratios[category][component][t].append(ratio)


    
    print(f"Skipped {skipped_masks} images due to missing masks.")
    print("\n--- Hotspot Placement Accuracy (ERA-in-Mask) Results ---")
    aggregated_stats = {}
    sorted_categories = sorted(category_ratios.keys(), key=lambda x: x != baseline_category_name)

    for category in sorted_categories:
        print(f"\nCategory: '{category}'")
        aggregated_stats[category] = {}
        
        for component in relevance_components:
            num_images = len(category_ratios[category][component][thresholds[0]])
            if num_images == 0: continue
            
            print(f"  Relevance Type: {component.capitalize()} ({num_images} images)")
            
            for t in thresholds:
                ratios = category_ratios[category][component][t]
                if not ratios: continue
                
                avg_ratio = np.nanmean(ratios) * 100
                std_ratio = np.nanstd(ratios) * 100

                print(f"    - Avg. In-Mask Ratio for Hotspot > {int(t*100)}%: {avg_ratio:.2f}% (±{std_ratio:.2f}%)")
                
                key_prefix = f'era_in_mask_{component}'
                aggregated_stats[category][f'avg_{key_prefix}_{int(t*100)}p'] = avg_ratio
                aggregated_stats[category][f'std_{key_prefix}_{int(t*100)}p'] = std_ratio
        
        aggregated_stats[category]['image_count'] = num_images

    return aggregated_stats

def analyze_effective_relevance_area(
    relevance_dict: Dict[str, torch.Tensor],
    analysis_categories: Dict[str, List[str]],
    thresholds: List[float] = [0.10, 0.25, 0.50]
) -> Dict[str, Dict[str, float]]:
    """
    Measures the "Effective Relevance Area" (ERA) for absolute, positive, and negative relevance.

    ERA is the percentage of the image area where the relevance magnitude exceeds
    a certain fraction of the maximum relevance for that image and component.

    Args:
        relevance_dict: Dictionary of filenames to their relevance maps.
        analysis_json_path: Path to the JSON file with image categories.
        thresholds: A list of fractions of the max relevance to use as thresholds.

    Returns:
        A dictionary with aggregated statistics for each category and relevance type.
    """
    print(f"\n--- Running Effective Relevance Area (ERA) Analysis ---")
    analysis_results = analysis_categories
    if not analysis_results:
        print("Error: Received empty or invalid analysis categories. Skipping mask analysis.")
        return {}

    filename_to_categories = defaultdict(list)
    for category, filenames in analysis_categories.items():
        for filename in filenames:
            filename_to_categories[filename].append(category)

    category_areas = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    baseline_category_name = "all_queries_baseline"
    relevance_components = ['absolute', 'positive', 'negative']

    for filename, relevance_map_orig in tqdm(relevance_dict.items(), desc="Analyzing ERA"):
        if isinstance(relevance_map_orig, torch.Tensor):
            relevance_map_orig = relevance_map_orig.cpu().numpy()
        
        total_pixels = relevance_map_orig.size

        for component in relevance_components:
            relevance_component_map = _get_relevance_component(relevance_map_orig, component)
            max_relevance = np.max(relevance_component_map)

            if max_relevance < 1e-7:
                continue
            
            for t in thresholds:
                threshold_value = t * max_relevance
                hot_pixel_count = np.sum(relevance_component_map > threshold_value)
                area_percent = hot_pixel_count / total_pixels
                
                category_areas[baseline_category_name][component][t].append(area_percent)
                categories_for_file = filename_to_categories.get(filename, [])
                for category in categories_for_file:
                    category_areas[category][component][t].append(area_percent)

    print("\n--- Effective Relevance Area (ERA) Results ---")
    aggregated_stats = {}
    sorted_categories = sorted(category_areas.keys(), key=lambda x: x != baseline_category_name)

    for category in sorted_categories:
        print(f"\nCategory: '{category}'")
        aggregated_stats[category] = {}
        
        for component in relevance_components:
            num_images = len(category_areas[category][component][thresholds[0]])
            if num_images == 0: continue
            
            print(f"  Relevance Type: {component.capitalize()} ({num_images} images)")

            for t in thresholds:
                areas = category_areas[category][component][t]
                if not areas: continue
                
                avg_area = np.mean(areas) * 100
                std_area = np.std(areas) * 100

                print(f"    - Avg. Area > {int(t*100)}% of Max: {avg_area:.2f}% (±{std_area:.2f}%)")
                
                key_prefix = f'era_{component}'
                aggregated_stats[category][f'avg_{key_prefix}_{int(t*100)}p'] = avg_area
                aggregated_stats[category][f'std_{key_prefix}_{int(t*100)}p'] = std_area
        
        aggregated_stats[category]['image_count'] = num_images

    return aggregated_stats

def analyze_relevance_concentration(
    relevance_dict: Dict[str, torch.Tensor],
    analysis_categories: Dict[str, List[str]],
    top_k_percent: float = 0.01
) -> Dict[str, Dict[str, float]]:
    """
    Analyzes concentration for absolute, positive, and negative relevance using Gini and RCI.

    Args:
        relevance_dict: A dictionary mapping filenames to their relevance maps.
        analysis_json_path: Path to the JSON file containing categorized filenames.
        top_k_percent: The percentage of pixels to consider for the RCI.

    Returns:
        A dictionary with aggregated statistics for each category and relevance type.
    """
    print(f"\n--- Running Relevance Concentration Analysis ---")
    analysis_results = analysis_categories
    if not analysis_results:
        print("Error: Received empty or invalid analysis categories. Skipping mask analysis.")
        return {}

    filename_to_categories = defaultdict(list)
    for category, filenames in analysis_categories.items():
        for filename in filenames:
            filename_to_categories[filename].append(category)

    category_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    baseline_category_name = "all_queries_baseline"
    relevance_components = ['absolute', 'positive', 'negative']

    def gini(x):
        x = x.flatten()
        if np.sum(x) < 1e-9:
            return np.nan
        x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(x, dtype=float)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

    for filename, relevance_map_orig in tqdm(relevance_dict.items(), desc="Analyzing relevance concentration"):
        if isinstance(relevance_map_orig, torch.Tensor):
            relevance_map_orig = relevance_map_orig.cpu().numpy()

        for component in relevance_components:
            relevance_component_map = _get_relevance_component(relevance_map_orig, component)
            
            gini_coeff = gini(relevance_component_map)
            
            flat_relevance = relevance_component_map.flatten()
            total_relevance = np.sum(flat_relevance)
            
            rci = np.nan
            if total_relevance > 1e-7:
                sorted_relevance = np.sort(flat_relevance)[::-1]
                k_pixels = max(1, int(len(sorted_relevance) * top_k_percent))
                top_k_sum = np.sum(sorted_relevance[:k_pixels])
                rci = top_k_sum / total_relevance

            # --- Store results ---
            category_metrics[baseline_category_name][component]['ginis'].append(gini_coeff)
            category_metrics[baseline_category_name][component]['rcis'].append(rci)

            categories_for_file = filename_to_categories.get(filename, [])
            for category in categories_for_file:
                category_metrics[category][component]['ginis'].append(gini_coeff)
                category_metrics[category][component]['rcis'].append(rci)

    print("\n--- Relevance Concentration Results ---")
    aggregated_stats = {}
    sorted_categories = sorted(category_metrics.keys(), key=lambda x: x != baseline_category_name)

    for category in sorted_categories:
        print(f"\nCategory: '{category}'")
        aggregated_stats[category] = {}
        
        for component in relevance_components:
            ginis = category_metrics[category][component]['ginis']
            rcis = category_metrics[category][component]['rcis']
            if not ginis: continue
            
            num_images = len(ginis)
            print(f"  Relevance Type: {component.capitalize()} ({num_images} images)")

            avg_gini = np.nanmean(ginis)
            std_gini = np.nanstd(ginis)
            avg_rci = np.nanmean(rcis)
            std_rci = np.nanstd(rcis)

            print(f"    - Avg. Gini Coefficient: {avg_gini:.3f} (±{std_gini:.3f})")
            print(f"    - Avg. RCI (Top {top_k_percent*100}%): {avg_rci:.3f} (±{std_rci:.3f})")
            
            key_prefix = component
            aggregated_stats[category][f'avg_gini_{key_prefix}'] = avg_gini
            aggregated_stats[category][f'std_gini_{key_prefix}'] = std_gini
            aggregated_stats[category][f'avg_rci_top_{top_k_percent*100}p_{key_prefix}'] = avg_rci
            aggregated_stats[category][f'std_rci_top_{top_k_percent*100}p_{key_prefix}'] = std_rci
        
        aggregated_stats[category]['image_count'] = num_images
        
    return aggregated_stats

def analyze_relevance_composition(
    relevance_dict: Dict[str, torch.Tensor],
    analysis_categories: Dict[str, List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Analyzes relevance maps for two key properties:
    1.  Composition: The ratio of positive relevance to the total absolute relevance.
    2.  Sparsity: The fraction of the map with zero (or near-zero) relevance.

    Args:
        relevance_dict: A dictionary mapping filenames to their relevance maps.
        analysis_json_path: Path to the JSON file containing categorized filenames.

    Returns:
        A dictionary with aggregated statistics for each category.
    """
    print(f"\n--- Running Relevance Composition and Sparsity Analysis ---")
    analysis_results = analysis_categories
    if not analysis_results:
        print("Error: Received empty or invalid analysis categories. Skipping mask analysis.")
        return {}



    filename_to_categories = defaultdict(list)
    for category, filenames in analysis_categories.items():
        for filename in filenames:
            filename_to_categories[filename].append(category)


    category_ratios = defaultdict(list)
    category_sparsity = defaultdict(list)
    baseline_category_name = "all_queries_baseline"
    for filename, relevance_map in tqdm(relevance_dict.items(), desc="Analyzing relevance composition"):
        if isinstance(relevance_map, torch.Tensor):
            relevance_map = relevance_map.cpu().numpy()


        pos_relevance = np.sum(relevance_map[relevance_map > 0])
        neg_relevance_abs = np.abs(np.sum(relevance_map[relevance_map < 0]))
        total_abs_relevance = pos_relevance + neg_relevance_abs

        if total_abs_relevance < 1e-9: 
            pos_ratio = np.nan
        else:
            pos_ratio = pos_relevance / total_abs_relevance
        
        total_pixels = relevance_map.size
        # Count pixels that are effectively zero (or small)
        zero_pixels = np.sum(np.abs(relevance_map) < 1e-7) 
        sparsity = zero_pixels / total_pixels
        
        category_ratios[baseline_category_name].append(pos_ratio)
        category_sparsity[baseline_category_name].append(sparsity)

        categories_for_file = filename_to_categories.get(filename, [])
        for category in categories_for_file:
            category_ratios[category].append(pos_ratio)
            category_sparsity[category].append(sparsity)


    print("\n--- Relevance Composition and Sparsity Results ---")
    aggregated_stats = {}
    sorted_categories = sorted(category_ratios.keys(), key=lambda x: x != baseline_category_name)
    print(f"sorted categories are: {sorted_categories}")
    for category in sorted_categories:
        ratios = category_ratios[category]
        sparsities = category_sparsity[category]
        if not ratios: 
            continue
        
        avg_ratio = np.nanmean(ratios)
        std_ratio = np.nanstd(ratios)
        
        avg_sparsity = np.mean(sparsities)
        std_sparsity = np.std(sparsities)

        print(f"\nCategory: '{category}' ({len(ratios)} images)")
        print(f"  - Avg. Positive Relevance Ratio: {avg_ratio:.3f} (±{std_ratio:.3f})")
        print(f"  - Avg. Relevance Sparsity (Zero-pixels): {avg_sparsity:.3f} (±{std_sparsity:.3f})")
        
        aggregated_stats[category] = {
            'avg_positive_relevance_ratio': avg_ratio,
            'std_positive_relevance_ratio': std_ratio,
            'avg_relevance_sparsity': avg_sparsity,
            'std_relevance_sparsity': std_sparsity,
            'image_count': len(ratios)
        }
        

def analyze_relevance_with_masks(
    dataloader: DataLoader,
    relevance_dict: Dict[str, torch.Tensor],
    analysis_categories: Dict[str, List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Analyzes how much of the LRP relevance falls inside/outside a segmentation mask
    for specific categories and for the entire dataset as a baseline.

    Args:
        dataloader: A DataLoader that yields batches containing images, masks, and filenames.
        relevance_dict: A dictionary mapping filenames to their relevance maps.
        analysis_json_path: Path to the JSON file containing the categorized image filenames.

    Returns:
        A dictionary with aggregated statistics for each category, including the baseline.
    """
    print(f"\n--- Running Relevance-in-Mask Analysis (with Baseline) ---")
    analysis_results = analysis_categories
    if not analysis_results:
        print("Error: Received empty or invalid analysis categories. Skipping mask analysis.")
        return {}

    filename_to_categories = defaultdict(list)
    for category, filenames in analysis_categories.items():
        for filename in filenames:
            filename_to_categories[filename].append(category)

    category_fractions = defaultdict(lambda: {'total': [], 'positive': [], 'negative': []})

    baseline_category_name = "all_queries_baseline"
    skipped_masks = 0
    for batch in tqdm(dataloader, desc="Analyzing relevance in masks"):
        masks_batch = batch["mask"]
        filenames_batch = batch["filename"]

        for i, filename in enumerate(filenames_batch):
            relevance_map = relevance_dict.get(filename)
            if relevance_map is None:
                print(f"Warning: Relevance map for '{filename}' not found. Skipping.")
                continue

            if masks_batch[i] is None:
                print(f"Warning: Mask for '{filename}' is None. Skipping.")
                skipped_masks += 1
                continue

            mask_np = masks_batch[i].cpu().numpy()
            total_frac, pos_frac, neg_frac = attention_inside_mask(relevance_map, mask_np)

            category_fractions[baseline_category_name]['total'].append(total_frac)
            category_fractions[baseline_category_name]['positive'].append(pos_frac)
            category_fractions[baseline_category_name]['negative'].append(neg_frac)

            categories_for_file = filename_to_categories.get(filename, [])
            for category in categories_for_file:
                category_fractions[category]['total'].append(total_frac)
                category_fractions[category]['positive'].append(pos_frac)
                category_fractions[category]['negative'].append(neg_frac)


    print(f"Skipped {skipped_masks} images due to missing masks.")
    print("\n--- Relevance-in-Mask Analysis Results ---")
    aggregated_stats = {}
    sorted_categories = sorted(category_fractions.keys(), key=lambda x: x != baseline_category_name)

    for category in sorted_categories:
        fractions = category_fractions[category]
        if not fractions['total']:
            continue

        avg_total = np.mean(fractions['total'])
        std_total = np.std(fractions['total'])
        avg_pos = np.mean(fractions['positive'])
        avg_neg = np.mean(fractions['negative'])

        print(f"\nCategory: '{category}' ({len(fractions['total'])} images)")
        print(f"  - Avg. Total Relevance in Mask: {avg_total:.3f} (±{std_total:.3f})")
        print(f"  - Avg. Positive Relevance in Mask: {avg_pos:.3f}")
        print(f"  - Avg. Negative Relevance in Mask: {avg_neg:.3f}")

        aggregated_stats[category] = {
            'avg_total_frac_in_mask': avg_total,
            'std_total_frac_in_mask': std_total,
            'avg_pos_frac_in_mask': avg_pos,
            'avg_neg_frac_in_mask': neg_frac,
            'image_count': len(fractions['total'])
        }
    
    return aggregated_stats


def main(cfg: Dict):
    """
    Runs a faithfulness evaluation by perturbing images based on LRP relevance maps
    and measuring the impact on k-NN Re-ID accuracy.
    """
    monkey_patch_zennit(verbose=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODE = cfg["lrp"]["mode"]
    print(f"\n--- RUNNING FAITHFULNESS EVALUATION WITH LRP MODE: {MODE} ---")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    model_wrapper, image_transforms, _ = get_model_wrapper(device=DEVICE, cfg=cfg["model"])
    mask_transform = get_mask_transform(cfg["model"]["img_size"])

    print(f"the dtype of the model is: {next(model_wrapper.model.parameters()).dtype}")

    model_type_str = "finetuned" if cfg["model"]["finetuned"] else "base"
    run_name = f"masking_analysis_{model_type_str}_{MODE}"
    wandb.init(
        project="Thesis-Iven",
        entity="gorillawatch",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        job_type="analysis"
    )

    dataset_dir = cfg["data"]["dataset_dir"]
    if not "zoo" in dataset_dir:
        predictions_json_path = "/sc/home/iven.schlegelmilch/bachelor_thesis_code/finetuned_predictions.json" if cfg["model"]["finetuned"] else "/sc/home/iven.schlegelmilch/bachelor_thesis_code/base_predictions.json"
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
        
    else:
        print("Using Zoo dataset for evaluation.")
        predictions_json_path = "/sc/home/iven.schlegelmilch/bachelor_thesis_code/finetuned_zoo_predictions.json" if cfg["model"]["finetuned"] else "/sc/home/iven.schlegelmilch/bachelor_thesis_code/base_zoo_predictions.json"
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



    local_query_indices = split_dataset.images_for_ce_knn

    global_query_indices = [idx + query_dataset_offset for idx in local_query_indices]

    print("Preparing the main KNN database (gallery)...")
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
    relevance_dict = {item['filename']: item['relevance'] for item in relevances_all}

    base_analysis_json_path = f"./visualizations/{os.path.basename(db_path_relevances)}.json"

    final_analysis_categories = get_combined_analysis_categories(
        base_analysis_path=base_analysis_json_path,
        predictions_json_path=predictions_json_path
    )

    for category, filenames in final_analysis_categories.items():
        if filenames:
            print(f"Category '{category}' example filename: {filenames[0]}")
        else:
            print(f"Category '{category}' has no filenames.")

    if final_analysis_categories is None:
        print("\nHalting analysis due to failure in creating combined category dictionary.")
        wandb.finish()
        return

    era_in_mask_stats = analyze_era_in_mask_ratio(
        dataloader=split_dataloader,
        relevance_dict=relevance_dict,
        analysis_categories=final_analysis_categories,
        thresholds=[0.01, 0.10, 0.25, 0.50, 0.75, 0.95, 0.99]
    )


    wandb.log({
        "era_in_mask_stats": wandb.Table(dataframe=pd.DataFrame(era_in_mask_stats))
    })



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DINOv2 AttnLRP experiment.")
    parser.add_argument(
        "--config_name", 
        type=str, 
        required=True,
        help="The name of the experiment config (e.g., 'finetuned', 'non_finetuned')."
    )
    args, unknown_args = parser.parse_known_args()

    cfg = load_config(args.config_name, unknown_args)

    command = [
        "python", 
        "/workspaces/bachelor_thesis_code/src/bachelor_thesis/generate_masks.py", 
        "--config_name", 
        args.config_name
    ] + unknown_args

    result = subprocess.run(
        command,
        check=True
    )

    main(cfg)