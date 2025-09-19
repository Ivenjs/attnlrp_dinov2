from utils import get_db_path, get_hpi_colors, get_mask_transform, load_config, get_denormalization_transform
from dataset import GorillaReIDDataset, custom_collate_fn
from lxt.efficient import monkey_patch_zennit
from basemodel import get_model_wrapper
from knn_helpers import get_knn_db
from typing import Dict, Tuple, List
from lrp_helpers import get_relevances
from torch.utils.data import DataLoader, ConcatDataset
from eval_helpers import attention_inside_mask
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


def analyze_relevance_with_masks(
    dataloader: DataLoader,
    relevance_dict: Dict[str, torch.Tensor],
    analysis_json_path: str
) -> Dict[str, Dict[str, float]]:
    """
    Analyzes how much of the LRP relevance falls inside/outside a segmentation mask
    for specific categories of images identified during perturbation analysis.

    Args:
        dataloader: A DataLoader that yields batches containing images, masks, and filenames.
        relevance_dict: A dictionary mapping filenames to their relevance maps.
        analysis_json_path: Path to the JSON file containing the categorized image filenames.

    Returns:
        A dictionary with aggregated statistics for each category.
    """
    print(f"\n--- Running Relevance-in-Mask Analysis ---")
    print(f"Loading analysis categories from: {analysis_json_path}")
    try:
        with open(analysis_json_path, 'r') as f:
            analysis_results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Analysis JSON file not found at {analysis_json_path}. Skipping mask analysis.")
        return {}

    # Create a reverse mapping for efficient lookup: {filename: category}
    filename_to_category = {
        filename: category
        for category, filenames in analysis_results.items()
        for filename in filenames
    }

    category_fractions = defaultdict(lambda: {'total': [], 'positive': [], 'negative': []})

    for batch in tqdm(dataloader, desc="Analyzing relevance in masks"):
        masks_batch = batch["mask"]
        filenames_batch = batch["filename"]

        for i, filename in enumerate(filenames_batch):
            category = filename_to_category.get(filename)
            if not category:
                continue

            relevance_map = relevance_dict.get(filename)
            if relevance_map is None:
                print(f"Warning: Relevance map for '{filename}' not found. Skipping.")
                continue

            # The mask from the dataloader is a tensor, but attention_inside_mask expects numpy
            mask_np = masks_batch[i].cpu().numpy()

            total_frac, pos_frac, neg_frac = attention_inside_mask(relevance_map, mask_np)

            # Store the results
            category_fractions[category]['total'].append(total_frac)
            category_fractions[category]['positive'].append(pos_frac)
            category_fractions[category]['negative'].append(neg_frac)

    # --- Aggregate and Print Results ---
    print("\n--- Relevance-in-Mask Analysis Results ---")
    aggregated_stats = {}
    for category, fractions in category_fractions.items():
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
            'avg_neg_frac_in_mask': avg_neg,
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
    run_name = f"faithfulness_eval_{model_type_str}_{MODE}"
    wandb.init(
        project="Thesis-Iven",
        entity="gorillawatch",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        job_type="analysis"
    )

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
        
    else:
        print("Using Zoo dataset for evaluation.")
        split_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith((".jpg", ".png"))]
        split_dataset = GorillaReIDDataset(
            image_dir=dataset_dir,
            filenames=split_files,
            transform=image_transforms,
            base_mask_dir=cfg["data"]["base_mask_dir"],
            mask_transform=mask_transform,
            k=cfg["knn"]["k"],
        )

        query_dataset_offset = 0
        full_db_dataset = split_dataset
        full_dataset_splits = os.path.basename(dataset_dir)
        split_name = full_dataset_splits



    local_query_indices = split_dataset.images_for_ce_knn

    global_query_indices = [idx + query_dataset_offset for idx in local_query_indices]

    # --- Create the KNN Search Database ---
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


    # --- Generate or Load Relevance Maps (for test images only) ---
    split_db_path_knn = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name=split_dataset.dataset_name, split_name=split_name, bp_transforms=cfg["model"]["bp_transforms"], db_dir=cfg["knn"]["db_embeddings_dir"]
    )
    split_embeddings, split_labels, split_filenames, split_video_ids = get_knn_db(
        db_path=split_db_path_knn, dataset=split_dataset, model_wrapper=model_wrapper,
        batch_size=cfg["data"]["batch_size"], device=DEVICE
    )

    # this loader only contains the subset of images that are used as queries for cross-encounter knn
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

    analysis_json_path = f"./visualizations/{os.path.basename(db_path_relevances)}.json"

    mask_analysis_stats = analyze_relevance_with_masks(
        dataloader=split_dataloader,
        relevance_dict=relevance_dict,
        analysis_json_path=analysis_json_path
    )

    wandb.log({"mask_analysis_stats": mask_analysis_stats})
    wandb.finish()



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