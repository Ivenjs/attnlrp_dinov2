import torch
from typing import List
import os
from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from basemodel import TimmWrapper
from eval_helpers_old_backup import faithfulness_eval_proxy_score, faithfulness_eval_acc
from utils import get_hpi_colors, parse_encounter_id


def evaluate_gamma_sweep_proxy_score(
    all_relevance_results: List[Dict],
    evaluation_dataloader: DataLoader,
    model_wrapper: TimmWrapper,
    db_embeddings: torch.Tensor,
    db_labels: List[str],
    db_filenames: List[str],
    db_video_ids: List[str],
    patch_size: int,
    device: torch.device,
    patches_per_step: int = 20,
    baseline_value: str = "mean",
    cross_encounter: bool = True,
    plot_curves: bool = False,
    seed=161
) -> List[Dict]:
    """
    Evaluate faithfulness scores for each image and gamma combination.
    
    Returns:
        - A list of flat dictionaries, each containing results for one metric.
        - A list of curve data points.
    """
    results_list = []
    all_curves_data = []

    relevances_by_file = {}
    for res in all_relevance_results:
        fname = res['filename']
        if fname not in relevances_by_file:
            relevances_by_file[fname] = []
        relevances_by_file[fname].append(res)
    
    for batch in tqdm(evaluation_dataloader, desc="Evaluating Images"):
        input_filename = batch["filename"][0]
        if input_filename not in relevances_by_file:
            continue

        query_label = batch["label"][0]
        input_tensor = batch["image"].to(device)
        video_id = batch["video"][0]

        print(f"\n=== Evaluating Image: {input_filename} ===")
        
        # Iterate through all generated relevance maps for this image
        for relevance_data in relevances_by_file[input_filename]:
            params = relevance_data["params"]
            mode = relevance_data["mode"]
        
            print(f"  Evaluating Mode='{mode}', γ_conv={params['conv_gamma']}, γ_lin={params['lin_gamma']}...")

            eval_kwargs = {
                "query_label": query_label,
                "query_filename": input_filename,
                "query_video_id": video_id,
                "db_embeddings": db_embeddings,
                "db_labels": db_labels,
                "db_filenames": db_filenames,
                "db_video_ids": db_video_ids,
                "distance_metric": params["distance_metric"],
                "proxy_temp": params["proxy_temp"],
                "cross_encounter": cross_encounter,
            }
            if mode == "proto_margin":
                eval_kwargs["topk"] = params["topk"]

            if mode == "similarity":
                eval_kwargs["reference_embedding"] = relevance_data["reference_embedding"].to(device) if relevance_data["reference_embedding"] is not None else None

            srg_results_by_metric = faithfulness_eval_proxy_score(
                relevance_map=relevance_data["relevance"],
                input_tensor=input_tensor,
                model=model_wrapper,
                mode=mode, 
                patch_size=patch_size,
                patches_per_step=patches_per_step,
                input_filename=input_filename,
                baseline_value=baseline_value,
                plot_curves=plot_curves,
                seed=seed,
                **eval_kwargs 
            ) 
            
            for metric_name, srg_results in srg_results_by_metric.items():
                result_row = {
                    "image": input_filename,
                    **params, 
                    "metric_name": metric_name,
                    "faithfulness_score": srg_results["faithfulness_score"],
                    "auc_morf": srg_results["auc_morf"],
                    "auc_lerf": srg_results["auc_lerf"],
                }
                results_list.append(result_row)

                if srg_results["morf_curve"] is not None:
                    all_curve_sets = [
                        ("morf_raw", srg_results["morf_curve"]),
                        ("lerf_raw", srg_results["lerf_curve"]),
                        ("random_raw", srg_results["random_curve"])
                    ]

                    for curve_label, curve in all_curve_sets:
                        for step, score in enumerate(curve):
                            row_dict = {
                                "image": input_filename,
                                "conv_gamma": params["conv_gamma"],
                                "lin_gamma": params["lin_gamma"],
                                "distance_metric": params["distance_metric"],
                                "proxy_temp": params["proxy_temp"],
                                "topk": params["topk"],
                                "metric_name": metric_name,
                                "curve_label": curve_label,
                                "step": step,
                                "score": score
                            }
                            all_curves_data.append(row_dict)
    
    return results_list, all_curves_data



def evaluate_gamma_sweep_acc(
    all_relevance_results: List[Dict], 
    query_dataset: torch.utils.data.Dataset,
    global_query_indices: List[int],
    model: TimmWrapper,
    db_embeddings: torch.Tensor,
    db_labels: List[str],
    db_videos: List[str],
    cfg: Dict[str, Any],
    patch_size: int,
    patches_per_step: int,
    baseline_value: str = "black",
    seed=161
) -> tuple[list[dict], list[dict]]:
    """
    Evaluates downstream faithfulness for multiple sets of relevance maps (e.g., a gamma sweep).
    
    This function groups relevance maps by their generation parameters, runs the batched
    downstream evaluation for each group, and formats the results into flat and long-form
    dataframes for easy analysis and logging.

    Returns:
        - A pandas DataFrame with summary statistics (AUCs, faithfulness) for each param set.
        - A pandas DataFrame with the raw curve data (step, score) for each param set.
    """
    device = db_embeddings.device
    unique_labels = sorted(list(set(db_labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    db_labels_int = torch.tensor([label_to_id[s] for s in db_labels], dtype=torch.long, device=device)

    db_encounters = [parse_encounter_id(v) for v in db_videos]
    unique_encounters = sorted(list(set(db_encounters)))
    encounter_to_id = {enc: i for i, enc in enumerate(unique_encounters)}
    db_encounter_ids_int = torch.tensor([encounter_to_id[s] for s in db_encounters], dtype=torch.long, device=device)

    summary_results_list = []
    all_curves_data_list = []

    relevances_by_params = defaultdict(dict)
    
    for res in all_relevance_results:
        params_key = tuple(sorted(res['params'].items()))
        filename = res['filename'] 
        relevances_by_params[params_key][filename] = res['relevance']

    print(f"Found {len(relevances_by_params)} unique parameter combinations to evaluate.")

    for params_key, relevance_maps_dict in tqdm(relevances_by_params.items(), desc="Evaluating Parameter Sets"):
        params_dict = dict(params_key)
        print(f"\n--- Evaluating parameters: {params_dict} ---")

        srg_results = faithfulness_eval_acc(
            relevance_maps_dict=relevance_maps_dict,
            query_dataset=query_dataset,
            global_query_indices=global_query_indices,
            model=model,
            db_embeddings=db_embeddings,
            db_labels_int=db_labels_int,
            db_encounter_ids_int=db_encounter_ids_int,
            label_to_id=label_to_id,
            encounter_to_id=encounter_to_id,
            cfg=cfg,
            patch_size=patch_size,
            patches_per_step=patches_per_step,
            baseline_value=baseline_value,
            seed=seed,
        )

        summary_row = {
            **params_dict,
            "metric_name": cfg["lrp"]["mode"],
            "faithfulness_score": srg_results["faithfulness_score"],
            "morf_vs_random": srg_results["morf_vs_random"],
            "lerf_vs_random": srg_results["lerf_vs_random"],
            "auc_morf": srg_results["auc_morf"],
            "auc_lerf": srg_results["auc_lerf"],
            "auc_random": srg_results["auc_random"],
        }
        summary_results_list.append(summary_row)

        all_curve_sets = [
            ("morf_raw", srg_results["morf_curve"]),
            ("lerf_raw", srg_results["lerf_curve"]),
            ("random_raw", srg_results["random_curve"])
        ]

        for curve_label, curve in all_curve_sets:
            num_steps = len(curve)
            x_axis = np.linspace(0, 100, num_steps)

            for step_idx, score in enumerate(curve):
                curve_point_row = {
                    **params_dict,
                    "metric_name": cfg["lrp"]["mode"],
                    "curve_label": curve_label,
                    "step": step_idx,
                    "percent_perturbed": x_axis[step_idx],
                    "score": score
                }
                all_curves_data_list.append(curve_point_row)
    
    return summary_results_list, all_curves_data_list

def find_robust_hyperparameters(
    results: List[Dict],
    decision_metric: str,
    sweep_evaluation: str,
) -> Tuple[Dict, pd.DataFrame, Dict]:
    """
    Finds robust hyperparameters based on a specific decision metric, but analyzes all metrics.
    """
    df = pd.DataFrame(results)
    df = df.fillna("NA") 
    
    parameter_groups = df.groupby(['conv_gamma', 'lin_gamma', 'distance_metric', 'proxy_temp','metric_name', 'topk'])    
    analysis_data = []

    for (conv_gamma, lin_gamma, distance_metric, proxy_temp, metric_name, topk), group in parameter_groups:
        scores = group['faithfulness_score'].values 
        num_valid = np.sum(~np.isnan(scores))

        mean_score = np.nanmean(scores)
        std_score = np.nanstd(scores)
        min_score = np.nanmin(scores)
        stability = 1 / (1 + std_score)
        robustness_score = 0
        # Robustness score
        if num_valid == 0:
            stability = 0
            robustness_score = -1 
        else:
            if sweep_evaluation == "proxy":
                robustness_score = (
                    0.3 * min_score +
                    0.3 * stability +
                    0.4 * mean_score
                )
            elif sweep_evaluation == "accuracy":
                robustness_score = mean_score # should be only one value per lerf, morf, random

        analysis_data.append({
            'conv_gamma': conv_gamma,
            'lin_gamma': lin_gamma,
            'distance_metric': distance_metric,
            'proxy_temp': proxy_temp,
            'topk': topk,
            'metric_name': metric_name, 
            
            'raw_mean': mean_score,
            'raw_std': std_score,
            'raw_min': min_score,
            'stability': stability,
            'robustness_score': robustness_score,
            'num_images': len(scores)
        })
    
    analysis_df = pd.DataFrame(analysis_data)

    
    decision_df = analysis_df[analysis_df['metric_name'] == decision_metric].copy()
    if decision_df.empty:
        raise ValueError(f"Decision metric '{decision_metric}' not found in the results.")

    best_idx = decision_df['robustness_score'].idxmax()
    best_row = decision_df.loc[best_idx]

    worst_idx = decision_df['robustness_score'].idxmin()
    worst_row = decision_df.loc[worst_idx]

    best_params_raw = {
        'conv_gamma': best_row['conv_gamma'],
        'lin_gamma': best_row['lin_gamma'],
        'distance_metric': best_row['distance_metric'],
        'proxy_temp': best_row['proxy_temp'],
        'topk': best_row['topk'],
        'metric_name': best_row['metric_name'], 
        'robustness_score': best_row['robustness_score'],
        'mean_score': best_row['raw_mean'],
        'min_score': best_row['raw_min'],
        'std_score': best_row['raw_std'],
    }

    worst_params_raw = {
        'conv_gamma': worst_row['conv_gamma'],
        'lin_gamma': worst_row['lin_gamma'],
        'distance_metric': worst_row['distance_metric'],
        'proxy_temp': worst_row['proxy_temp'],
        'topk': worst_row['topk'],
        'metric_name': worst_row['metric_name'], 
        'robustness_score': worst_row['robustness_score'],
        'mean_score': worst_row['raw_mean'],
        'min_score': worst_row['raw_min'],
        'std_score': worst_row['raw_std'],
    }

    return best_params_raw, analysis_df, worst_params_raw


def visualize_robustness_analysis(
    analysis_df: pd.DataFrame,
    hpi_colors: Dict[str, float],
    save_path: str = "/workspaces/attnlrp_dinov2/src/bachelor_thesis/robustness_analysis/robustness_analysis.png",
) -> List[str]:
    """Creates visualizations for each combination of distance and evaluation metric."""    
    path_root, path_ext = os.path.splitext(save_path)
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []

    plots = [
        ('Mean Faithfulness', 'raw_mean')
    ]

    metric_names = analysis_df['metric_name'].dropna().unique()
    dist_metrics = analysis_df['distance_metric'].dropna().unique() if 'distance_metric' in analysis_df else [None]
    proxy_temps = analysis_df['proxy_temp'].dropna().unique() if 'proxy_temp' in analysis_df else [None]
    topks = analysis_df['topk'].dropna().unique() if 'topk' in analysis_df else [None]

    custom_cmap = sns.blend_palette([hpi_colors.get("yellow"), hpi_colors.get("orange"),hpi_colors.get("red")], as_cmap=True)

    for eval_metric in metric_names:
        for dist_metric in dist_metrics:
            for proxy_temp in proxy_temps:
                for topk in topks:
                    df_subset = analysis_df[
                        (analysis_df['metric_name'] == eval_metric) &
                        (analysis_df['distance_metric'] == dist_metric) &
                        (analysis_df['proxy_temp'] == proxy_temp) &
                        (analysis_df['topk'] == topk)
                    ]
                    if df_subset.empty:
                        continue

                    fig, axes = plt.subplots(1, len(plots), figsize=(7 * len(plots), 6))
                    if len(plots) == 1:
                        axes = [axes]
                        
                    fig.suptitle(
                        f"Robustness Analysis (Eval: {eval_metric}, "
                        f"Distance: {dist_metric}, Proxy Temp: {proxy_temp}, TopK: {topk})",
                        fontsize=20, y=1.05
                    )

                    for ax, (title, value_col) in zip(axes, plots):
                        pivot_df = df_subset.pivot(
                            index='conv_gamma',
                            columns='lin_gamma',
                            values=value_col
                        )
                        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap=custom_cmap, ax=ax)
                        ax.set_title(title)

                    plt.tight_layout(rect=[0, 0, 1, 0.98])
                    metric_save_path = f"{path_root}_{eval_metric}_{dist_metric}_proxy{proxy_temp}_topk{topk}{path_ext}"
                    plt.savefig(metric_save_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    saved_paths.append(metric_save_path)

    return saved_paths

def visualize_holdout_performance(
    analysis_df: pd.DataFrame,
    hpi_colors: Dict[str, float],
    save_path: str = "wandb_plots/holdout_performance.png",
) -> List[str]:
    """
    Creates bar chart visualizations for the sparse holdout set analysis.

    Instead of a heatmap, this plots the performance of each top-K candidate.
    """
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    metric_names = analysis_df['metric_name'].dropna().unique()

    for eval_metric in metric_names:
        df_subset = analysis_df[analysis_df['metric_name'] == eval_metric].copy()
        
        if df_subset.empty:
            continue

        df_subset.sort_values(by='raw_mean', ascending=False, inplace=True)
        
        def create_label(row):
            parts = []
            
            if 'conv_gamma' in row and pd.notna(row['conv_gamma']):
                parts.append(f"conv_gamma: {row['conv_gamma']}")
            if 'lin_gamma' in row and pd.notna(row['lin_gamma']):
                parts.append(f"lin_gamma: {row['lin_gamma']}")

            if 'proxy_temp' in row and pd.notna(row['proxy_temp']):
                try:
                    parts.append(f"temp: {float(row['proxy_temp'])}")
                except (ValueError, TypeError):
                    pass 

            if 'topk' in row and pd.notna(row['topk']):
                try:
                    parts.append(f"top_k: {int(float(row['topk']))}") 
                except (ValueError, TypeError):
                    pass
            return "\n".join(parts)

        df_subset['param_label'] = df_subset.apply(create_label, axis=1)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(max(12, 2 * len(df_subset)), 8))

        bar_color = hpi_colors.get("red", "skyblue")
        
        sns.barplot(
            x='param_label',
            y='raw_mean',
            data=df_subset,
            ax=ax,
            color=bar_color
        )

        ax.set_title(f"Validation Set Performance of Top Candidates of Train Set - {eval_metric}", fontsize=16)
        ax.set_xlabel("Hyperparameter Combination", fontsize=12)
        ax.set_ylabel("Mean Faithfulness Score", fontsize=12)
        ax.tick_params(axis='x', rotation=0, labelsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f')
            
        plt.tight_layout()
        
        metric_save_path = save_path.replace('.png', f'_{eval_metric}.png')
        plt.savefig(metric_save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(metric_save_path)

    return saved_paths

def print_robustness_summary(
    best_params_raw: Dict, 
    analysis_df: pd.DataFrame,
    decision_metric: str
):
    """Prints a summary of the robustness analysis, focusing on the decision metric."""
    print("\n" + "="*60)
    print("ROBUSTNESS ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nDecision made based on METRIC: '{decision_metric.upper()}'")
    print(f"Best Hyperparameters (Robustness on '{decision_metric}'):")
    print(f"  Conv Gamma: {best_params_raw['conv_gamma']}")
    print(f"  Lin Gamma:  {best_params_raw['lin_gamma']}")
    print(f"  Proxy Temp: {best_params_raw['proxy_temp']}")
    print(f"  Top-K:      {best_params_raw['topk']}")
    print(f"  Distance Metric: {best_params_raw['distance_metric']}")
    print(f"  Robustness Score: {best_params_raw['robustness_score']:.4f}")
    
    print(f"\nPerformance Metrics for this combination ('{decision_metric}'):")
    print(f"  Mean Score:       {best_params_raw['mean_score']:.4f}")
    print(f"  Min Score:        {best_params_raw['min_score']:.4f}")
    print(f"  Std Score:        {best_params_raw['std_score']:.4f}")

    # Show top 3 combinations for the decision metric
    print(f"\nTop 3 Combinations for '{decision_metric}':")
    top_3 = analysis_df[analysis_df['metric_name'] == decision_metric].nlargest(3, 'robustness_score')
    for i, (_, row) in enumerate(top_3.iterrows(), 1):
        print(f"  {i}. γ_conv={row['conv_gamma']}, γ_lin={row['lin_gamma']}, dist={row['distance_metric']}, proxy_temp={row['proxy_temp']}, topk={row['topk']}: "
              f"robust={row['robustness_score']:.4f}, mean={row['raw_mean']:.4f}, min={row['raw_min']:.4f}")

def calculate_aggregate_statistics(results: List[Dict]) -> Dict:
    """
    Calculate various aggregate statistics from the results for both raw and normalized scores.
    
    Args:
        results: List of evaluation results from evaluate_multi_image_gamma_sweep
    
    Returns:
        Dictionary with aggregate statistics for both score types
    """
    df = pd.DataFrame(results)
    
    # Overall statistics for raw scores
    raw_stats = {
        'raw_mean': df['faithfulness_score'].mean(),
        'raw_median': df['faithfulness_score'].median(),
        'raw_std': df['faithfulness_score'].std(),
        'raw_min': df['faithfulness_score'].min(),
        'raw_max': df['faithfulness_score'].max(),
        'raw_q25': df['faithfulness_score'].quantile(0.25),
        'raw_q75': df['faithfulness_score'].quantile(0.75)
    }
    
    overall_stats = {**raw_stats}

    parameter_stats_raw = df.groupby(['conv_gamma', 'lin_gamma', 'distance_metric', 'proxy_temp', 'topk'])['faithfulness_score'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    parameter_stats_raw.columns = [f'raw_{col}' if col not in ['conv_gamma', 'lin_gamma', 'distance_metric', 'proxy_temp', 'topk'] else col
                              for col in parameter_stats_raw.columns]

    parameter_stats = pd.merge(parameter_stats_raw, on=['conv_gamma', 'lin_gamma', 'distance_metric', 'proxy_temp', 'topk'])

    parameter_stats = pd.merge(parameter_stats, on=['conv_gamma', 'lin_gamma', 'distance_metric', 'proxy_temp', 'topk'])

    best_by_raw_mean = parameter_stats.loc[parameter_stats['raw_mean'].idxmax()]
    best_by_raw_min = parameter_stats.loc[parameter_stats['raw_min'].idxmax()]
    
    image_stats_raw = df.groupby(['image'])['faithfulness_score'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    image_stats_raw.columns = [f'raw_{col}' if col != 'image' else col 
                              for col in image_stats_raw.columns]

    image_stats = pd.merge(image_stats_raw, on=['image'])

    return {
        'overall': overall_stats,
        'by_parameter': parameter_stats,
        'by_image': image_stats,
        'best_by_raw_mean': best_by_raw_mean,
        'best_by_raw_min': best_by_raw_min
    }

def log_sweep(
    cfg: Dict,
    final_decision: str,
    approved_params: Dict,
    worst_params: Dict,
    tune_performance: Dict,
    holdout_performance: Dict,
    generalization_drop_percent: float,
    analysis_df_tune: pd.DataFrame,
    analysis_df_holdout: pd.DataFrame,
    tune_results_list: List[Dict],
    holdout_results_list: List[Dict],
    tune_curves_list: List,
    holdout_curves_list: List,
    run
):
    """Logs the complete story of a nested validation experiment to W&B, handling multiple metrics."""
    print("\n--- Logging Nested Validation Experiment to Weights & Biases ---")
    hpi_colors = get_hpi_colors(cfg)


    wandb.summary["final_decision"] = final_decision
    wandb.summary["decision_metric"] = cfg["lrp"]["mode"]
    wandb.summary["generalization_drop_percent"] = generalization_drop_percent
    
    wandb.summary["approved_conv_gamma"] = approved_params['conv_gamma']
    wandb.summary["approved_lin_gamma"] = approved_params['lin_gamma']
    wandb.summary["approved_distance_metric"] = approved_params['distance_metric']
    wandb.summary["approved_proxy_temp"] = approved_params['proxy_temp']
    wandb.summary["approved_topk"] = approved_params['topk']

    wandb.summary[f"tune_set_mean_faithfulness ({cfg['lrp']['mode']})"] = tune_performance['mean_faithfulness']
    wandb.summary[f"holdout_set_mean_faithfulness ({cfg['lrp']['mode']})"] = holdout_performance['mean_faithfulness']

    mode = cfg["sweep"]["sweep_evaluation"]

    log_payload = {}

    log_payload.update({
        "tune_set/raw_results_per_image": wandb.Table(dataframe=pd.DataFrame(tune_results_list)),
        "holdout_set/raw_results_per_image": wandb.Table(dataframe=pd.DataFrame(holdout_results_list)),
        "tune_set/parameter_analysis": wandb.Table(dataframe=analysis_df_tune),
        "holdout_set/parameter_analysis": wandb.Table(dataframe=analysis_df_holdout)
    })


    print("\nGenerating heatmaps for tune set analysis...")
    tune_paths = visualize_robustness_analysis(
        analysis_df_tune, 
        hpi_colors=hpi_colors, 
        save_path="wandb_plots/tune_analysis.png"
    )

    print("Generating bar charts for holdout set performance...")
    holdout_paths = visualize_holdout_performance(
        analysis_df_holdout, 
        hpi_colors=hpi_colors, 
        save_path="wandb_plots/holdout_performance.png"
    )

    log_payload.update({
        "plots/tune_set_analysis_heatmaps": [wandb.Image(p, caption=os.path.basename(p)) for p in tune_paths],
        "plots/holdout_set_performance_barcharts": [wandb.Image(p, caption=os.path.basename(p)) for p in holdout_paths]
    })

    print("\n--- Logging Faithfulness Curves for Approved Parameters (All Metrics) ---")

    needed_cols = ["conv_gamma", "lin_gamma", "proxy_temp", "distance_metric", "topk"]

    tune_curves_df = pd.DataFrame(tune_curves_list)
    holdout_curves_df = pd.DataFrame(holdout_curves_list)

    if mode == "proxy_score":
        try:
            img_size = cfg['model']['img_size']
            patch_size = cfg['model']['patch_size']
            patches_per_step = cfg['eval']['patches_per_step']
            total_patches = (img_size // patch_size) ** 2
            print(f"Calculating fractions for proxy score: total_patches={total_patches}, patches_per_step={patches_per_step}")
            
            def add_fraction_col(df):
                df['fraction_perturbed'] = (df['step'] * patches_per_step) / total_patches
                df['fraction_perturbed'] = df['fraction_perturbed'].clip(upper=1.0) # Cap at 100%
                return df
                
        except KeyError as e:
            print(f"Error: Missing key {e} in config needed for fraction calculation. Cannot proceed with curve logging.")
            return
    elif mode == "accuracy":
        def add_fraction_col(df):
            if 'percent_perturbed' in df.columns:
                df['fraction_perturbed'] = df['percent_perturbed'] / 100.0
            elif 'fraction_perturbed' in df.columns:
                pass # Already exists
            else:
                 raise KeyError("For 'accuracy' mode, expected 'percent_perturbed' or 'fraction_perturbed' column in curve data.")
            return df

    tune_curves_df = add_fraction_col(tune_curves_df)
    holdout_curves_df = add_fraction_col(holdout_curves_df)
    
    needed_cols = ["conv_gamma", "lin_gamma", "proxy_temp", "distance_metric", "topk"]
    for col in needed_cols:
        if col not in tune_curves_df.columns: tune_curves_df[col] = np.nan
        if col not in holdout_curves_df.columns: holdout_curves_df[col] = np.nan
        if col not in approved_params: approved_params[col] = np.nan
        if col not in worst_params: worst_params[col] = np.nan
    
    approved_tune_df = tune_curves_df[_build_filter(tune_curves_df, approved_params)].copy()
    approved_holdout_df = holdout_curves_df[_build_filter(holdout_curves_df, approved_params)].copy()
    approved_tune_df['split'] = 'tune'
    approved_holdout_df['split'] = 'holdout'
    approved_curves_df = pd.concat([approved_tune_df, approved_holdout_df], ignore_index=True)

    worst_tune_df = tune_curves_df[_build_filter(tune_curves_df, worst_params)].copy()
    worst_holdout_df = holdout_curves_df[_build_filter(holdout_curves_df, worst_params)].copy()
    worst_tune_df['split'] = 'tune'
    worst_holdout_df['split'] = 'holdout'
    worst_curves_df = pd.concat([worst_tune_df, worst_holdout_df], ignore_index=True)

    if not approved_curves_df.empty:
        log_payload["approved_params/curve_data"] = wandb.Table(dataframe=approved_curves_df)
        print(f"Logged {len(approved_curves_df)} rows of curve data for approved parameters.")
    if not worst_curves_df.empty:
        log_payload["worst_params/curve_data"] = wandb.Table(dataframe=worst_curves_df)
        print(f"Logged {len(worst_curves_df)} rows of curve data for worst parameters.")

    plot_args = {"hpi_colors": hpi_colors, "mode": mode}
    if mode == "accuracy":
        plot_args["plot_fractions"] = cfg.get("faithfulness", {}).get("fractions_to_record")

    if not approved_curves_df.empty:
        for eval_metric in approved_curves_df['metric_name'].unique():
            metric_df = approved_curves_df[approved_curves_df['metric_name'] == eval_metric]
            log_key_tune = f"plots/approved_curves_tune_{eval_metric}"
            log_payload[log_key_tune] = plot_and_log_faithfulness_curves(
                df=metric_df[metric_df['split'] == 'tune'],
                title_prefix=f"Approved Params {eval_metric} - Train Set",
                db_set="Train Set",
                log_key=log_key_tune, **plot_args
            )
            log_key_holdout = f"plots/approved_curves_holdout_{eval_metric}"
            log_payload[log_key_holdout] = plot_and_log_faithfulness_curves(
                df=metric_df[metric_df['split'] == 'holdout'],
                title_prefix=f"Approved Params {eval_metric} - Validation Set",
                db_set="Train + Validation Set",
                log_key=log_key_holdout, **plot_args
            )

    if not worst_curves_df.empty:
        for eval_metric in worst_curves_df['metric_name'].unique():
            metric_df = worst_curves_df[worst_curves_df['metric_name'] == eval_metric]
            log_key_tune = f"plots/worst_curves_tune_{eval_metric}"
            log_payload[log_key_tune] = plot_and_log_faithfulness_curves(
                df=metric_df[metric_df['split'] == 'tune'],
                title_prefix=f"Worst Params {eval_metric} - Train Set",
                db_set="Train Set",
                log_key=log_key_tune, **plot_args
            )
            log_key_holdout = f"plots/worst_curves_holdout_{eval_metric}"
            log_payload[log_key_holdout] = plot_and_log_faithfulness_curves(
                df=metric_df[metric_df['split'] == 'holdout'],
                title_prefix=f"Worst Params {eval_metric} - Validation Set",
                db_set="Train + Validation Set",
                log_key=log_key_holdout, **plot_args
            )

    if not approved_curves_df.empty:
        print("Generating interactive plot for approved parameters...")
        plot_df = approved_curves_df.copy()
        
        if mode == 'accuracy':
            x_col = 'percent_perturbed' if 'percent_perturbed' in plot_df.columns else 'fraction_perturbed'
        else:
            x_col = 'step'

        if x_col != 'step':
            plot_df.rename(columns={x_col: 'step'}, inplace=True)

        plot_df['series'] = plot_df.apply(
            lambda row: f"approved/{row['split']}/{row['curve_label'].replace('_raw','')}", axis=1
        )
        
        plot_table = wandb.Table(dataframe=plot_df[['series', 'step', 'score']])
        
        wandb_plot = wandb.plot.line(
            plot_table, 
            x="fraction_perturbed",
            y="score", 
            stroke="series",                
            title="Interactive Faithfulness Curves (Approved Params)"
        )
        log_payload["plots/interactive_approved_curves"] = wandb_plot

    final_log_payload = {k: v for k, v in log_payload.items() if v is not None}
    
    if final_log_payload:
        print("\n--- Logging all artifacts to W&B in a single step ---")
        wandb.log(final_log_payload)
    print("--- Finished logging to W&B ---")

    print("--- Finished logging to W&B ---")

def _build_filter(df: pd.DataFrame, params: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in params.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if col not in df.columns:
            continue
        fill_default = -1 if df[col].dtype.kind in "if" else "NA"
        mask &= (df[col].fillna(fill_default) == val)
    return mask

def plot_and_log_faithfulness_curves(
    df: pd.DataFrame,
    title_prefix: str,
    db_set: str,
    log_key: str,
    hpi_colors: Dict[str, float],
    mode: str,
    plot_fractions: Optional[List[float]] = None,
    save_dir: str = "wandb_plots"
) -> Optional[wandb.Image]:
    """
    Plots faithfulness curves with mode-specific styling and logs to W&B.

    Handles both pre-aggregated data (like accuracy) and per-image data
    (like proxy scores) that needs aggregation.
    """
    if df.empty:
        print(f"Skipping plot for {log_key} as the DataFrame is empty.")
        return None
    
    if 'fraction_perturbed' not in df.columns:
        print(f"Warning: 'fraction_perturbed' column not found for {log_key}. Skipping plot.")
        return None

    print(f"Plotting '{mode}' curve for {log_key} with {len(df)} data points...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {
        'morf_raw': hpi_colors.get("red", "red"),
        'lerf_raw': hpi_colors.get("yellow", "blue"),
        'random_raw': hpi_colors.get("gray", "gray"),
    }

    if mode == "accuracy":
        ax.set_title(f'{title_prefix}:\nImpact of Patch Perturbation on k-NN Re-Id Accuracy', fontsize=14)
        ax.set_ylabel(f"Balanced Cross-Encounter k-NN@5 Accuracy on {db_set}", fontsize=10)
        ax.set_ylim(bottom=0)
    elif mode == "proxy_score":
        ax.set_title(f'{title_prefix}:\nImpact of Patch Perturbation on Proxy Score', fontsize=14)
        ax.set_ylabel("Proxy Score", fontsize=12)

    x_col = 'fraction_perturbed'
    is_pre_aggregated = df.groupby(['curve_label', x_col]).size().max() == 1
    
    for label, group in df.groupby('curve_label'):
        legend_label = label.replace('_raw', '').upper()
        if is_pre_aggregated:
            group = group.sort_values(x_col)
            ax.plot(group[x_col], group['score'], label=legend_label, color=colors.get(label, 'black'))
        else:
            agg_data = group.groupby(x_col)['score'].agg(['mean', 'std']).reset_index().sort_values(x_col)
            ax.plot(agg_data[x_col], agg_data['mean'], label=f"Mean {legend_label}", color=colors.get(label, 'black'))
            ax.fill_between(
                agg_data[x_col],
                agg_data['mean'] - agg_data['std'].fillna(0),
                agg_data['mean'] + agg_data['std'].fillna(0),
                alpha=0.2, color=colors.get(label, 'gray')
            )
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{log_key.replace('/', '_')}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150)
    print(f"Saved curve plot to: {save_path}")
    
    image_to_log = wandb.Image(save_path)
    plt.close(fig)
    return image_to_log

