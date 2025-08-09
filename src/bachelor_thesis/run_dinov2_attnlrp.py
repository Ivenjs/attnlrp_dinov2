from mask_generator import MaskGenerator
from utils import load_all_configs
from dataset import GorillaReIDDataset, custom_collate_fn
from lxt.efficient import monkey_patch_zennit
from torchvision import transforms
from basemodel import get_model_wrapper
from knn_helpers import get_knn_db, get_query_performance_metrics, compute_knn_proxy_score
from eval_helpers import attention_inside_mask
from tqdm import tqdm
from typing import Dict, Tuple, Any, List
from collections import defaultdict
from lrp_helpers import compute_simple_attnlrp_pass, compute_knn_attnlrp_pass
from dino_patcher import DINOPatcher
from basemodel import TimmWrapper
from torch.utils.data import DataLoader
from dinov2_attnlrp_sweep import run_gamma_sweep
from visualize import Visualizer 
import matplotlib.pyplot as plt
from zennit.image import imgify
import pandas as pd
from PIL import Image
import gc
import numpy as np
import subprocess
import os
import random
import torch
# 1) run sam to get segmentation masks of data split, if not already saved
# 2) create dataset with masks
# 3) run lrp with swept parameters on images in dataset and compute the mask score
# 3a) compare basemodel vs finetuned model on val (maybe try train to see of results are better?)
# 3b) compare finetuned model on train vs val (overfitting?)
# 4) save worst performing images and mask their background. How does the knn score change? can I also recompute accuracy with only these few images?
# mean faithfullness curves for finetuned model vs base model

#faithfullnes score durchschnittskurve berechnen
#knn score weird?
#implement relevance chcker correctly to validate relevance maps, like in github issue
def get_denormalization_transform(mean: tuple, std: tuple) -> transforms.Compose:
    """Creates a transform to de-normalize image tensors using a lambda function."""
    # Convert to tensors for broadcasting
    mean_tensor = torch.tensor(mean)
    std_tensor = torch.tensor(std)

    return transforms.Compose([
        # Reshape to (C, 1, 1) to work with image tensors (C, H, W)
        transforms.Lambda(lambda x: x * std_tensor.view(3, 1, 1)),
        transforms.Lambda(lambda x: x + mean_tensor.view(3, 1, 1)),
    ])

def run_masking_experiment(
    model_wrapper: TimmWrapper,
    dataset: GorillaReIDDataset,
    db_embeddings: torch.Tensor,
    db_labels: List[str],
    db_filenames: List[str],
    mask_scores: Dict[str, Tuple], # The output from attention_inside_mask
    batch_size: int,
    device: torch.device,
    cfg: Dict
) -> pd.DataFrame:
    """
    Performs the causal masking experiment systematically on the entire dataset.

    For each image, it calculates performance metrics before and after masking
    the background, allowing for a quantitative analysis of the background's role.
    """
    print("\n--- Starting Systematic Masking Experiment ---")
    results_list = []
    
    # Use the original dataloader without shuffling to iterate through the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running masking experiment"):
            # --- ROBUST BATCH HANDLING START ---
            
            # Move the entire batch of images to the device at once
            images_orig_batch = batch["image"].to(device)
            masks_batch = batch["mask"] # This is a list of tensors/Nones
            labels_batch = batch["label"]
            filenames_batch = batch["filename"]
            
            # --- Process each item within the batch individually ---
            for i in range(len(filenames_batch)):
                # Isolate the data for the i-th sample
                image_orig = images_orig_batch[i].unsqueeze(0) # Keep batch dim -> [1, C, H, W]
                mask_tensor = masks_batch[i]
                label = labels_batch[i]
                filename = filenames_batch[i]

                # The experiment cannot proceed without a valid mask.
                if mask_tensor is None:
                    print(f"Warning: Skipping {filename} in masking experiment, no mask available.")
                    continue
                
                mask = mask_tensor.to(device)

                # --- 1. Baseline Performance ---
                embedding_orig = model_wrapper(image_orig)
                
                baseline_metrics = get_query_performance_metrics(
                    query_embedding=embedding_orig, query_label=label, query_filename=filename,
                    db_embeddings=db_embeddings, db_labels=db_labels, db_filenames=db_filenames,
                    distance_metric=cfg["knn"]["distance_metric"], k=cfg["knn"]["k"]
                )
                
                baseline_proxy_score = compute_knn_proxy_score(
                    query_embedding=embedding_orig, query_label=label, query_filename=filename,
                    db_embeddings=db_embeddings, db_labels=db_labels, db_filenames=db_filenames,
                    k=cfg["knn"]["k"], distance_metric=cfg["knn"]["distance_metric"]
                ).item()

                # --- 2. Masked Performance ---
                image_masked = image_orig * mask
                embedding_masked = model_wrapper(image_masked)

                masked_metrics = get_query_performance_metrics(
                    query_embedding=embedding_masked, query_label=label, query_filename=filename,
                    db_embeddings=db_embeddings, db_labels=db_labels, db_filenames=db_filenames,
                    distance_metric=cfg["knn"]["distance_metric"], k=cfg["knn"]["k"]
                )

                masked_proxy_score = compute_knn_proxy_score(
                    query_embedding=embedding_masked, query_label=label, query_filename=filename,
                    db_embeddings=db_embeddings, db_labels=db_labels, db_filenames=db_filenames,
                    k=cfg["knn"]["k"], distance_metric=cfg["knn"]["distance_metric"]
                ).item()

                # --- 3. Aggregate Results ---
                AoGR_total = mask_scores[filename][0] 
                AoGR_positive = mask_scores[filename][1]
                AoGR_negative = mask_scores[filename][2]
                
                results_list.append({
                    'filename': filename, 'label': label,
                    'AoGR_total': AoGR_total,
                    'AoGR_positive': AoGR_positive,
                    'AoGR_negative': AoGR_negative,
                    'background_attention_total': 1 - AoGR_total,
                    'background_attention_positive': 1 - AoGR_positive,
                    'background_attention_negative': 1 - AoGR_negative,
                    'rank_orig': baseline_metrics['rank'], 'rank_masked': masked_metrics['rank'],
                    'gt_sim_orig': baseline_metrics['gt_similarity'], 'gt_sim_masked': masked_metrics['gt_similarity'],
                    'recall_at_k_orig': baseline_metrics['recall_at_k'], 'recall_at_k_masked': masked_metrics['recall_at_k'],
                    'proxy_score_orig': baseline_proxy_score, 'proxy_score_masked': masked_proxy_score,
                    'delta_rank': baseline_metrics['rank'] - masked_metrics['rank'],
                    'delta_gt_sim': baseline_metrics['gt_similarity'] - masked_metrics['gt_similarity'],
                    'delta_recall': baseline_metrics['recall_at_k'] - masked_metrics['recall_at_k'],
                    'delta_proxy_score': masked_proxy_score - baseline_proxy_score
                })

    print("--- Masking Experiment Complete ---")
    return pd.DataFrame(results_list)

def compute_relevances(
    model_wrapper: TimmWrapper, 
    conv_gamma: float, 
    lin_gamma: float, 
    dataloader: DataLoader, 
    device: torch.device,
    db_embeddings: torch.Tensor = None, 
    db_labels: List[str] = None, 
    db_filenames: List[str] = None, 
    k_neighbors: int = 5, 
    mode: str = "knn", 
    verbose: bool = False, 
    distance_metric: str = "cosine"
) -> Dict[float, Tuple[torch.Tensor, np.ndarray]]:
    relevance_mask_dict = defaultdict(dict)
    with DINOPatcher(model_wrapper):
        for batch in tqdm(dataloader, desc="Processing batches"):
            input_batch = batch["image"].to(device)
            labels_batch = batch["label"]
            filenames_batch = batch["filename"]
            mask_batch = batch["mask"]
            for j, filename in enumerate(filenames_batch):
                # Slice the data for the j-th sample
                input_tensor_single = input_batch[j].unsqueeze(0) 
                label_single = labels_batch[j]

                if mode == "simple":
                    # Call the non-batched LRP function directly
                    relevance_single = compute_simple_attnlrp_pass(
                        conv_gamma=conv_gamma,
                        lin_gamma=lin_gamma,
                        model_wrapper=model_wrapper,
                        input_tensor=input_tensor_single,  
                        verbose=verbose  
                    )

                elif mode == "knn":
                    assert db_embeddings is not None, "db_embeddings must be provided for 'knn' mode."
                    assert db_filenames is not None, "db_filenames must be provided for 'knn' mode."
                    # Call the non-batched LRP function directly
                    relevance_single = compute_knn_attnlrp_pass(
                        conv_gamma=conv_gamma,
                        lin_gamma=lin_gamma,
                        model_wrapper=model_wrapper,
                        input_tensor=input_tensor_single,
                        query_label=label_single,
                        query_filename=filename,
                        db_embeddings=db_embeddings,
                        db_labels=db_labels,
                        db_filenames=db_filenames,
                        distance_metric=distance_metric,
                        k_neighbors=k_neighbors,
                        verbose=verbose, 
                    )
                mask_tensor_single = mask_batch[j]
                assert mask_tensor_single is not None, f"Mask for {filename} is None. Cannot compute relevance."
                mask_np_copy = mask_tensor_single.cpu().numpy()

                # Store the final, safe data
                # relevance_single is detached from graph and moved to CPU
                # mask_np_copy is a plain NumPy array with no ties to the DataLoader
                relevance_mask_dict[filename] = (relevance_single.detach().cpu(), mask_np_copy)
    return relevance_mask_dict

def analyze_masking_exp(masking_results_df, cfg: Dict) -> None:
    # --- Basic Analysis of the Results ---
    print("\n--- Masking Experiment Summary ---")
    print(masking_results_df.describe())
    
    # Calculate overall change in Rank-1 accuracy
    rank1_orig = (masking_results_df['rank_orig'] == 1).mean()
    rank1_masked = (masking_results_df['rank_masked'] == 1).mean()
    print(f"\nOverall Rank-1 Accuracy (Orig):   {rank1_orig:.4f}")
    print(f"Overall Rank-1 Accuracy (Masked): {rank1_masked:.4f}")
    
    # --- Correlation Plot ---
    plt.figure(figsize=(10, 6))
    plt.scatter(
        masking_results_df['background_attention_total'],
        masking_results_df['delta_proxy_score'],
        alpha=0.6
    )
    plt.title('Performance Change vs. Background Attention')
    plt.xlabel('Background Attention (1 - AoGR)')
    plt.ylabel('Change in k-NN Proxy Score (Masked - Original)')
    plt.grid(True)
    correlation_plot_path = os.path.join(cfg["data"]["visualization_dir"], "correlation_plot.png")
    plt.savefig(correlation_plot_path)

    # do another plot for positive and negative AoGR
    plt.figure(figsize=(10, 6))
    plt.scatter(
        masking_results_df['background_attention_positive'],
        masking_results_df['delta_proxy_score'],
        alpha=0.6, label='Positive AoGR'
    )
    plt.scatter(
        masking_results_df['background_attention_negative'],
        masking_results_df['delta_proxy_score'],
        alpha=0.6, label='Negative AoGR', color='orange'
    )
    plt.title('Performance Change vs. Positive/Negative Background Attention')
    plt.xlabel('Background Attention (1 - AoGR)')
    plt.ylabel('Change in k-NN Proxy Score (Masked - Original)')
    plt.legend()
    plt.grid(True)
    correlation_plot_path_pos_neg = os.path.join(cfg["data"]["visualization_dir"], "correlation_plot_pos_neg.png")
    plt.savefig(correlation_plot_path_pos_neg)
    print(f"Saved correlation plot to: {correlation_plot_path}")

if __name__ == "__main__":

    result = subprocess.run(
        ["python", "/workspaces/bachelor_thesis_code/src/bachelor_thesis/generate_masks.py"],
        check=True  
    )

    monkey_patch_zennit(verbose=True)  

    LOG_TO_WANDB = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VERBOSE = False  
    random.seed(27)  
    torch.manual_seed(27)  

    model_wrapper_finetuned, image_transforms, data_config = get_model_wrapper(device=DEVICE, finetuned=True)

    config_dir = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/configs"
    cfg = load_all_configs(config_dir)
    MODE = cfg["lrp"]["mode"]


    root_dir = cfg["data"]["dataset_dir"]
    val_dir = os.path.join(root_dir, "val")

    val_files = [f for f in os.listdir(val_dir) if f.lower().endswith((".jpg", ".png"))]


    mask_transform = transforms.Compose([
        transforms.Resize(
            size=cfg["model"]["img_size"], # e.g., (518, 518)
            interpolation=transforms.InterpolationMode.NEAREST # Use NEAREST for masks!
        ),
        transforms.ToTensor(), # Converts mask to a [1, H, W] tensor of floats (0.0 or 1.0)
    ])
    # IMPORTANT: The interpolation mode for the mask must be NEAREST.
    # Using BILINEAR or BICUBIC would create intermediate values (like 0.5)
    # along the edges, blurring the mask.

    val_dataset = GorillaReIDDataset(
        image_dir=val_dir,
        filenames=val_files,
        transform=image_transforms,      # The full transform for the image
        base_mask_dir=cfg["data"]["base_mask_dir"],
        mask_transform=mask_transform  # The spatial-only transform for the mask
    )


    val_db_embeddings_finetuned, val_db_labels_finetuned, val_db_filenames_finetuned = get_knn_db(
        db_dir=cfg["knn"]["db_embeddings_dir"], split_name="val", dataset=val_dataset,
        model_wrapper=model_wrapper_finetuned, model_checkpoint_path=cfg["model"]["checkpoint_path"], batch_size=cfg["data"]["batch_size"], device=DEVICE
    )
    val_dataloader = DataLoader(val_dataset, batch_size=cfg["data"]["batch_size"], num_workers=4, collate_fn=custom_collate_fn,shuffle=False)

    conv_gamma = cfg["lrp"]["conv_gamma"]
    lin_gamma = cfg["lrp"]["lin_gamma"]
    distance_metric = cfg["knn"]["distance_metric"]
    k_neighbors = cfg["knn"]["k"]

    relevance_mask_dict_finetuned = compute_relevances(
        model_wrapper=model_wrapper_finetuned,
        conv_gamma=conv_gamma,
        lin_gamma=lin_gamma,
        dataloader=val_dataloader,
        device=DEVICE,
        db_embeddings=val_db_embeddings_finetuned,
        db_labels=val_db_labels_finetuned,
        db_filenames=val_db_filenames_finetuned,
        k_neighbors=k_neighbors,
        mode=MODE,
        verbose=VERBOSE,
        distance_metric=distance_metric
    )

    scores_finetuned = {}
    for filename, (relevance, mask) in relevance_mask_dict_finetuned.items():
        scores_finetuned[filename] = attention_inside_mask(relevance, mask)


    masking_results_df = run_masking_experiment(
        model_wrapper=model_wrapper_finetuned,
        dataset=val_dataset,
        db_embeddings=val_db_embeddings_finetuned,
        db_labels=val_db_labels_finetuned,
        db_filenames=val_db_filenames_finetuned,
        mask_scores=scores_finetuned, # Pass the AoGR scores
        batch_size=cfg["data"]["batch_size"],
        device=DEVICE,
        cfg=cfg
    )

    analyze_masking_exp(masking_results_df, cfg)

    results_path = os.path.join(cfg["data"]["visualization_dir"], "masking_experiment_results.csv")
    masking_results_df.to_csv(results_path, index=False)
    print(f"\nSaved systematic masking experiment results to: {results_path}")


    print(f"Clearing model finetuned from GPU memory...")
    del model_wrapper_finetuned
    gc.collect() # Trigger Python's garbage collection
    torch.cuda.empty_cache() # Release cached memory back to the OS

    #TODO: does the base model have the same transforms and data config?
    model_wrapper_base, _, _ = get_model_wrapper(device=DEVICE, finetuned=False)
    val_db_embeddings_base, val_db_labels_base, val_db_filenames_base = get_knn_db(
        db_dir=cfg["knn"]["db_embeddings_dir"], split_name="val", dataset=val_dataset,
        model_wrapper=model_wrapper_base, model_checkpoint_path="/workspaces/bachelor_thesis_code/base", batch_size=cfg["data"]["batch_size"], device=DEVICE
    )

    relevance_mask_dict_base = compute_relevances(
        model_wrapper=model_wrapper_base,
        conv_gamma=conv_gamma,
        lin_gamma=lin_gamma,
        dataloader=val_dataloader,
        device=DEVICE,
        db_embeddings=val_db_embeddings_base,
        db_labels=val_db_labels_base,
        db_filenames=val_db_filenames_base,
        k_neighbors=k_neighbors,
        mode=MODE,
        verbose=VERBOSE,
        distance_metric=distance_metric
    )

    scores_base = {}
    for filename, (relevance, mask) in relevance_mask_dict_base.items():
        scores_base[filename] = attention_inside_mask(relevance, mask)

    def print_stats(name, scores_dict):
        totals = torch.tensor([v[0] for v in scores_dict.values()])
        positives = torch.tensor([v[1] for v in scores_dict.values()])
        negatives = torch.tensor([v[2] for v in scores_dict.values()])
        print(f"\n--- {name} Stats ---")
        print(f"  Total Frac   - Mean: {totals.mean():.4f}, Max: {totals.max():.4f}, Min: {totals.min():.4f}")
        print(f"  Positive Frac - Mean: {positives.mean():.4f}, Max: {positives.max():.4f}, Min: {positives.min():.4f}")
        print(f"  Negative Frac - Mean: {negatives.mean():.4f}, Max: {negatives.max():.4f}, Min: {negatives.min():.4f}")

    print_stats("Finetuned Model", scores_finetuned)
    print_stats("Base Model", scores_base)

    # ====================================================================
    # UPDATED VISUALIZATION SECTION
    # ====================================================================
    print("\n--- Starting Visualization ---")
    
    # 1. Instantiate the Visualizer with the correct denorm transform
    denorm_transform = get_denormalization_transform(mean=data_config['mean'], std=data_config['std'])
    visualizer = Visualizer(
        save_dir=cfg["data"]["visualization_dir"],
        denorm_transform=denorm_transform
    )

    # 2. Find interesting samples to plot (using POSITIVE relevance as the key metric)
    all_scores = []
    for fname in scores_finetuned.keys():
        ft_pos_score = scores_finetuned[fname][1]
        base_pos_score = scores_base[fname][1]
        diff = base_pos_score - ft_pos_score # Positive if base has higher positive fraction
        all_scores.append((fname, ft_pos_score, base_pos_score, diff))

    by_finetuned_pos_asc = sorted(all_scores, key=lambda x: x[1])
    by_base_advantage_desc = sorted(all_scores, key=lambda x: x[3], reverse=True)

    samples_to_plot = {
        "finetuned_worst_pos_relevance": by_finetuned_pos_asc[0][0],
        "finetuned_best_pos_relevance": by_finetuned_pos_asc[-1][0],
        "base_model_biggest_advantage": by_base_advantage_desc[0][0],
        "finetuned_biggest_advantage": by_base_advantage_desc[-1][0],
    }

    print(f"\nSelected samples for plotting: {list(samples_to_plot.values())}")

    # 3. Generate and save plots for the selected samples
    fname_to_idx = {
        os.path.splitext(fname)[0]: i
        for i, fname in enumerate(val_dataset.filenames)
    }

    for reason, filename in samples_to_plot.items():
        print(f"Plotting '{reason}': {filename}")
        
        sample_data = val_dataset[fname_to_idx[filename]]
        image_tensor = sample_data["image"]

        relevance_ft, mask = relevance_mask_dict_finetuned[filename]
        relevance_base, _ = relevance_mask_dict_base[filename]
        relevance_ft = relevance_ft/ torch.abs(relevance_ft).max() if torch.abs(relevance_ft).max() != 0 else relevance_ft
        relevance_base = relevance_base / torch.abs(relevance_base).max() if torch.abs(relevance_base).max() != 0 else relevance_base
        
        visualizer.plot_comparison(
            filename=f"{reason}_{filename}",
            image_tensor=image_tensor,
            mask=mask,
            base_relevance=relevance_base,
            finetuned_relevance=relevance_ft,
            base_scores=scores_base[filename],
            finetuned_scores=scores_finetuned[filename]
        )

    print("\n--- Visualization complete ---")
