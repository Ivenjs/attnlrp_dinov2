from utils import get_db_path, get_hpi_colors, get_mask_transform, load_config
from dataset import GorillaReIDDataset, custom_collate_fn
from lxt.efficient import monkey_patch_zennit
from torchvision import transforms
from basemodel import get_model_wrapper
from knn_helpers import get_knn_db
from eval_helpers import attention_inside_mask, get_query_performance_metrics
from lrp_helpers import compute_knn_proto_margin, compute_knn_proxy_soft, compute_similarity_score
from tqdm import tqdm
from typing import Dict, Tuple, List
from lrp_helpers import get_relevances
from basemodel import TimmWrapper
from torch.utils.data import DataLoader
from visualize import AttentionVisualizer 
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from omegaconf import OmegaConf
import numpy as np 
import pandas as pd
import subprocess
import os
import random
import torch
import argparse 
import wandb
# 1) run sam to get segmentation masks of data split, if not already saved
# 2) create dataset with masks
# 3) run lrp with swept parameters on images in dataset and compute the mask score
# 3a) compare basemodel vs finetuned model on validation (maybe try train to see of results are better?)
# 3b) compare finetuned model on train vs validation (overfitting?)
# 4) save worst performing images and mask their background. How does the knn score change? can I also recompute accuracy with only these few images?


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
    db_video_ids: List[str],
    mask_scores: Dict[str, Tuple], # The output from attention_inside_mask
    batch_size: int,
    device: torch.device,
    decision_metric: str,
    cfg: Dict
) -> pd.DataFrame:
    """
    Performs the causal masking experiment systematically on the entire dataset.

    For each image, it calculates performance metrics before and after masking
    the background, allowing for a quantitative analysis of the background's role.
    """
    print("\n--- Starting Systematic Masking Experiment ---")
    results_list = []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    score_fn = None
    score_fn_kwargs = {}

    if decision_metric == "soft_knn_margin":
        score_fn = compute_knn_proxy_soft
        score_fn_kwargs = {
            "db_embeddings": db_embeddings,
            "db_labels": db_labels,
            "db_filenames": db_filenames,
            "db_video_ids": db_video_ids,
            "distance_metric": cfg["knn"]["distance_metric"],
            "temp": cfg["knn"]["temp"]
        }
    elif decision_metric == "proto_margin":
        score_fn = compute_knn_proto_margin
        score_fn_kwargs = {
            "db_embeddings": db_embeddings,
            "db_labels": db_labels,
            "db_filenames": db_filenames,
            "db_video_ids": db_video_ids,
            "distance_metric": cfg["knn"]["distance_metric"],
            "temp": cfg["knn"]["temp"],
            "topk_neg": cfg["knn"]["topk_neg"]
        }
    elif decision_metric == "similarity":
        score_fn = compute_similarity_score
        score_fn_kwargs = {
            "db_embeddings": db_embeddings,
            "db_labels": db_labels,
            "db_filenames": db_filenames,
            "db_video_ids": db_video_ids,
        }
    else:
        raise ValueError(f"Unsupported evaluation decision metric: '{decision_metric}'")

    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running masking experiment"):
            # --- ROBUST BATCH HANDLING START ---
            
            # Move the entire batch of images to the device at once
            images_orig_batch = batch["image"].to(device)
            masks_batch = batch["mask"] # This is a list of tensors/Nones
            labels_batch = batch["label"]
            filenames_batch = batch["filename"]
            video_ids_batch = batch["video"]

            # --- Process each item within the batch individually ---
            for i in range(len(filenames_batch)):
                # Isolate the data for the i-th sample
                image_orig = images_orig_batch[i].unsqueeze(0) # Keep batch dim -> [1, C, H, W]
                mask_tensor = masks_batch[i]
                label = labels_batch[i]
                filename = filenames_batch[i]
                video_id = video_ids_batch[i]

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
                #change plot colors to hpi
                result_base = score_fn(
                    query_embedding=embedding_orig, query_label=label, query_filename=filename, query_video_id=video_id,
                    **score_fn_kwargs
                )
                if isinstance(result_base, tuple):
                    baseline_proxy_score = result_base[0].item()
                else:
                    baseline_proxy_score = result_base.item()


                # --- 2. Masked Performance ---
                if cfg["eval"]["baseline_value"] == "black":
                    image_masked = image_orig * mask
                elif cfg["eval"]["baseline_value"] == "mean":
                    mean_color = image_orig.mean(dim=[2, 3], keepdim=True) # Shape: [1, C, 1, 1]
                    background_fill = mean_color.expand_as(image_orig)
                    image_masked = image_orig * mask + background_fill * (1 - mask)

                embedding_masked = model_wrapper(image_masked)

                masked_metrics = get_query_performance_metrics(
                    query_embedding=embedding_masked, query_label=label, query_filename=filename,
                    db_embeddings=db_embeddings, db_labels=db_labels, db_filenames=db_filenames,
                    distance_metric=cfg["knn"]["distance_metric"], k=cfg["knn"]["k"]
                )
                result_mask = score_fn(
                    query_embedding=embedding_masked, query_label=label, query_filename=filename, query_video_id=video_id,
                    **score_fn_kwargs
                )
                if isinstance(result_mask, tuple):
                    masked_proxy_score = result_mask[0].item()
                else:
                    masked_proxy_score = result_mask.item()


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
                    'delta_recall_at_k': baseline_metrics['recall_at_k'] - masked_metrics['recall_at_k'],
                    'delta_proxy_score': masked_proxy_score - baseline_proxy_score
                })

    print("--- Masking Experiment Complete ---")
    return pd.DataFrame(results_list)

def analyze_masking_exp(masking_results_df: pd.DataFrame, output_dir: str, colors: Dict[str, float]) -> List[str]:
    # --- Basic Analysis of the Results ---
    print("\n--- Masking Experiment Summary ---")
    print(masking_results_df.describe())
    
    # Calculate overall change in Rank-1 accuracy
    rank1_orig = (masking_results_df['rank_orig'] == 1).mean()
    rank1_masked = (masking_results_df['rank_masked'] == 1).mean()
    print(f"\nOverall Rank-1 Accuracy (Orig):   {rank1_orig:.4f}")
    print(f"Overall Rank-1 Accuracy (Masked): {rank1_masked:.4f}")
    negative_color = colors.get("gray")
    primary_color = colors.get("yellow")
    saved_plot_paths = []
    # --- Correlation Plots ---
    for col in masking_results_df.columns:
        if col.startswith("delta_"):
            plt.figure(figsize=(10, 6))
            plt.scatter(
                masking_results_df['background_attention_total'],
                masking_results_df[col],
                alpha=0.6,
                color=primary_color
            )
            plt.title('Performance Change vs. Background Attention')
            plt.xlabel('Background Attention (1 - AoGR)')
            plt.ylabel(f'Change in {col} (Masked - Original)')
            plt.grid(True)
            correlation_plot_path = os.path.join(output_dir, f"correlation_plot_{col}.png")
            os.makedirs(os.path.dirname(correlation_plot_path), exist_ok=True)
            plt.savefig(correlation_plot_path)
            saved_plot_paths.append(correlation_plot_path)

            # do another plot for positive and negative AoGR
            plt.figure(figsize=(10, 6))
            plt.scatter(
                masking_results_df['background_attention_positive'],
                masking_results_df[col],
                alpha=0.6, label='Positive AoGR',
                color=primary_color
            )
            plt.scatter(
                masking_results_df['background_attention_negative'],
                masking_results_df[col],
                alpha=0.6, label='Negative AoGR', color=negative_color
            )
            plt.title('Performance Change vs. Positive/Negative Background Attention')
            plt.xlabel('Background Attention (1 - AoGR)')
            plt.ylabel(f'Change in {col} (Masked - Original)')
            plt.legend()
            plt.grid(True)
            correlation_plot_path_pos_neg = os.path.join(output_dir, f"correlation_plot_pos_neg_{col}.png")
            os.makedirs(os.path.dirname(correlation_plot_path_pos_neg), exist_ok=True)
            plt.savefig(correlation_plot_path_pos_neg)
            saved_plot_paths.append(correlation_plot_path_pos_neg)

            # Create binned plots for better visibility
            attention_types = {
                'Total': 'background_attention_total',
                'Positive': 'background_attention_positive',
                'Negative': 'background_attention_negative'
            }

            for att_name, att_col in attention_types.items():
                plot_path = os.path.join(output_dir, f"binned_plot_{col}_vs_{att_col}.png")
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                
                create_binned_plot(
                    df=masking_results_df,
                    attention_col=att_col,
                    metric_col=col,
                    output_path=plot_path,
                    plot_color=primary_color,
                    attention_name=att_name
                )
                saved_plot_paths.append(plot_path)
    return saved_plot_paths

def create_binned_plot(
    df: pd.DataFrame, 
    attention_col: str, 
    metric_col: str, 
    output_path: str,
    plot_color: str = "blue",
    attention_name: str = "Attention"
):
    """
    Creates and saves a binned bar plot of a metric vs. an attention score.
    """
    try:
        # Define bins for the attention score (-1 to 1)
        bins = np.linspace(-1.0, 1.0, 11)
        df_copy = df.copy() 
        df_copy['bin_labels'] = pd.cut(df_copy[attention_col], bins=bins, include_lowest=True)


        binned_stats = df_copy.groupby('bin_labels', observed=False)[metric_col].agg(['mean', 'sem'])

        if binned_stats.empty or binned_stats['mean'].isnull().all():
            print(f"Skipping plot: No data for {metric_col} vs {attention_col}")
            return

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        binned_stats.plot(
            kind='bar', y='mean', yerr='sem', ax=ax, capsize=4, 
            legend=False, color=plot_color, edgecolor='black'
        )
        
        # Formatting
        ax.set_title(f'Mean Performance Drop vs. Binned Background {attention_name}')
        ax.set_xlabel(f'Bin of Background {attention_name}')
        ax.set_ylabel(f'Mean Change in {metric_col.replace("delta_", "")}')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()

        # Save and close
        plt.savefig(output_path)
        plt.close(fig)

    except Exception as e:
        print(f"Could not generate binned plot for {metric_col} vs {attention_col}. Error: {e}")

def generate_heatmaps(
    samples_to_plot: Dict[str, str], 
    masking_results_df: pd.DataFrame,
    val_dataset: GorillaReIDDataset, 
    relevance_mask_dict: Dict, 
    model_data_config: Dict, 
    output_dir: str
) -> List[str]:
    print("\n--- Starting heatmap visualization ---")
    saved_heatmap_paths = []
    denorm_transform = get_denormalization_transform(mean=model_data_config['mean'], std=model_data_config['std'])

    visualizer = AttentionVisualizer(
        save_dir=output_dir,
        denorm_transform=denorm_transform
    )
    
    # Create a mapping from filename to its index in the dataset for quick lookup
    fname_to_idx = {
        os.path.splitext(f)[0]: i for i, f in enumerate(val_dataset.filenames)
    }

    for reason, filename in samples_to_plot.items():

        # Get stats for this image from the dataframe
        image_stats = masking_results_df[masking_results_df['filename'] == filename].iloc[0]


        # Get the necessary data for plotting
        if os.path.splitext(filename)[0] not in fname_to_idx:
            print(f"Warning: Filename '{filename}' not found in dataset. Skipping.")
            continue
            
        sample_data = val_dataset[fname_to_idx[os.path.splitext(filename)[0]]]
        image_tensor = sample_data["image"]
        
        relevance, mask = relevance_mask_dict[filename]

        # Normalize relevance map for consistent visualization
        # This makes heatmaps comparable by scaling them to the [-1, 1] range
        relevance = relevance / torch.abs(relevance).max()
        
        save_path = visualizer.plot_heatmap( 
            filename=f"{reason}_{filename}",
            image_tensor=image_tensor,
            mask=mask,
            relevance=relevance,
            stats=image_stats.to_dict() 
        )
        saved_heatmap_paths.append(save_path)
    return saved_heatmap_paths

def select_samples_for_visualization(
    df: pd.DataFrame, 
    n_per_category: int = 2,
    n_random: int = 5
) -> Dict[str, str]:
    """
    Selects a variety of interesting image filenames from the masking results for visualization.

    Args:
        df: The DataFrame containing the results from the masking experiment.
        n_per_category: The number of samples to select for each defined category.
        n_random: The number of random samples to select.

    Returns:
        A dictionary mapping a descriptive reason (for the filename) to a filename.
    """
    print("\n--- Selecting Interesting Samples for Visualization ---")
    
    df = df.sort_values('filename').reset_index(drop=True)
    
    samples_to_plot = {}

    # --- Category 1: "Good Students" (High AoGR, good rank, little change) ---
    # Sort by highest attention on gorilla, then best rank
    good_students = df.sort_values(by=['AoGR_total', 'rank_orig'], ascending=[False, True])
    for i, row in good_students.head(n_per_category).iterrows():
        fname = row['filename']
        key = f"good_student_{i+1}"
        if fname not in samples_to_plot.values():
            samples_to_plot[key] = fname

    # --- Category 2: "Saved by Masking" (Huge performance jump after masking) ---
    # Sort by the largest improvement in proxy score
    saved_by_masking = df.sort_values(by='delta_proxy_score', ascending=False)
    for i, row in saved_by_masking.head(n_per_category).iterrows():
        fname = row['filename']
        key = f"saved_by_masking_{i+1}"
        if fname not in samples_to_plot.values():
            samples_to_plot[key] = fname

    # --- Category 3: "Right for Wrong Reason" (Low AoGR but good rank) ---
    # Filter for Rank 1, then find the ones with the lowest AoGR (most background attention)
    right_wrong_reason = df[df['rank_orig'] == 1].sort_values(by='AoGR_total', ascending=True)
    for i, row in right_wrong_reason.head(n_per_category).iterrows():
        fname = row['filename']
        key = f"right_for_wrong_reason_{i+1}"
        if fname not in samples_to_plot.values():
            samples_to_plot[key] = fname
            
    # --- Category 4: "Hurt by Masking" (Performance got worse) ---
    # Sort by the most negative change in proxy score
    hurt_by_masking = df.sort_values(by='delta_proxy_score', ascending=True)
    for i, row in hurt_by_masking.head(n_per_category).iterrows():
        fname = row['filename']
        key = f"hurt_by_masking_{i+1}"
        if fname not in samples_to_plot.values():
            samples_to_plot[key] = fname

    # --- Category 5: Highest Background Attention ---
    highest_bg = df.sort_values(by='background_attention_total', ascending=False)
    for i, row in highest_bg.head(n_per_category).iterrows():
        fname = row['filename']
        key = f"highest_bg_attn_{i+1}"
        if fname not in samples_to_plot.values():
            samples_to_plot[key] = fname
            
    # --- Category 6: Random Samples ---
    random_samples = df.sample(n=n_random, random_state=27)
    for i, row in random_samples.iterrows():
        fname = row['filename']
        key = f"random_sample_{i+1}"
        if fname not in samples_to_plot.values():
            samples_to_plot[key] = fname

    print(f"Selected {len(samples_to_plot)} unique samples for plotting.")
    return samples_to_plot

def main(cfg: Dict):
    """
    Main function to run the LRP and masking experiment for a single model configuration.
    """
    monkey_patch_zennit(verbose=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    MODE = cfg["lrp"]["mode"]
    print(f"\n--- RUNNING WITH MODE: {MODE} ---")

    DECISION_METRIC = MODE

    is_finetuned = cfg["model"]["finetuned"]
    model_type_str = "finetuned" if is_finetuned else "base"
    print(f"--- Running experiment for: {model_type_str.upper()} MODEL ---")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project="Thesis-Iven", 
        entity="gorillawatch", # Or your entity
        name=f"dinov2_attnlrp_analysis_{model_type_str}_{MODE}",
        config=cfg_dict,
        job_type="analysis"
    )

    # Create a model-specific output directory
    base_output_dir = cfg["data"]["output_base_dir"]
    experiment_output_dir = os.path.join(base_output_dir, model_type_str)
    visualization_dir = os.path.join(experiment_output_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    print(f"All outputs will be saved in: {experiment_output_dir}")


    # --- 2. Load Model and Data ---
    model_wrapper, image_transforms, model_data_config = get_model_wrapper(device=DEVICE, cfg=cfg["model"])

    root_dir = cfg["data"]["dataset_dir"]
    val_dir = os.path.join(root_dir, "validation")
    val_files = [f for f in os.listdir(val_dir) if f.lower().endswith((".jpg", ".png"))]


    mask_transform = get_mask_transform(cfg["model"]["img_size"])

    val_dataset = GorillaReIDDataset(
        image_dir=val_dir,
        filenames=val_files,
        transform=image_transforms,
        base_mask_dir=cfg["data"]["base_mask_dir"],
        mask_transform=mask_transform
    )

    val_dataloader = DataLoader(val_dataset, batch_size=cfg["data"]["batch_size"], num_workers=0, collate_fn=custom_collate_fn, shuffle=False)
    split_name = "validation"

    # --- 3. Compute k-NN Database ---
    db_path_knn = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset=val_dataset,
        split_name=split_name,
        db_dir=cfg["knn"]["db_embeddings_dir"]
    )
    db_embeddings, db_labels, db_filenames, db_video_ids = get_knn_db(
        db_path=db_path_knn,
        dataset=val_dataset,
        model_wrapper=model_wrapper,
        batch_size=cfg["data"]["batch_size"],
        device=DEVICE
    )

    # --- 4. Compute Relevances ---
    db_path_relevances = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset=val_dataset,
        split_name=split_name,
        db_dir=cfg["lrp"]["db_relevances_dir"],
        decision_metric=DECISION_METRIC
    )

    relevances_all = get_relevances(
        db_path=db_path_relevances,
        model_wrapper=model_wrapper,
        dataloader=val_dataloader,
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
        db_video_ids=db_video_ids
    )


    scores = {
        item['filename']: attention_inside_mask(item['relevance'], item['mask'])
        for item in relevances_all
    }

    # --- 5. Run Masking Experiment ---
    masking_results_df = run_masking_experiment(
        model_wrapper=model_wrapper,
        dataset=val_dataset,
        db_embeddings=db_embeddings,
        db_labels=db_labels,
        db_filenames=db_filenames,
        db_video_ids=db_video_ids,
        mask_scores=scores,
        batch_size=cfg["data"]["batch_size"],
        device=DEVICE,
        decision_metric=DECISION_METRIC,
        cfg=cfg
    )

    # Use the dynamic experiment_output_dir for saving results
    results_path = os.path.join(experiment_output_dir, "masking_experiment_results.csv")
    masking_results_df.to_csv(results_path, index=False)
    print(f"\nSaved systematic masking experiment results to: {results_path}")

    wandb.log({
        "masking_experiment_results": wandb.Table(dataframe=masking_results_df)
    })
    # Log key overall metrics to the summary for easy comparison
    rank1_orig = (masking_results_df['rank_orig'] == 1).mean()
    rank1_masked = (masking_results_df['rank_masked'] == 1).mean()
    wandb.summary["rank1_orig"] = rank1_orig
    wandb.summary["rank1_masked"] = rank1_masked
    wandb.summary["mean_delta_proxy_score"] = masking_results_df['delta_proxy_score'].mean()

    hpi_colors = get_hpi_colors(cfg)
    colors = {
        'yellow': hpi_colors["yellow"],
        'gray': hpi_colors["gray"],
    }

    correlation_plot_paths = analyze_masking_exp(masking_results_df=masking_results_df, output_dir=os.path.join(visualization_dir, "plots"), colors=colors)
    corr, p_value = spearmanr(masking_results_df['background_attention_total'], masking_results_df['delta_rank'])
    print(f"\nSpearman Correlation (Background Attention vs. Delta Rank): {corr:.3f} (p-value: {p_value:.3f})")
    wandb.summary["spearman_corr"] = corr
    wandb.summary["spearman_p_value"] = p_value

    samples_to_plot = select_samples_for_visualization(
        df=masking_results_df,
        n_per_category=2, 
        n_random=10
    )
    
    relevance_mask_dict = {
        item['filename']: (item['relevance'], item['mask']) for item in relevances_all
    }
    heatmap_paths = generate_heatmaps(
        samples_to_plot=samples_to_plot,
        masking_results_df=masking_results_df,
        val_dataset=val_dataset, 
        relevance_mask_dict=relevance_mask_dict, 
        model_data_config=model_data_config, 
        output_dir=os.path.join(visualization_dir, "heatmaps")
    )

    wandb.log({
        "analysis_plots/correlations": [
            wandb.Image(p, caption=os.path.basename(p)) for p in correlation_plot_paths
        ],
        "visualizations/heatmaps": [
            wandb.Image(p, caption=os.path.basename(p)) for p in heatmap_paths
        ]
    })


    print(f"\n--- Experiment for {model_type_str.upper()} model complete. ---")
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