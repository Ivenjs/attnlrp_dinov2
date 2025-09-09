import torch
from typing import List
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from basemodel import TimmWrapper
from eval_helpers import srg_eval
from utils import get_hpi_colors


def evaluate_gamma_sweep(
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
    baseline_value: str = "black",
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

            srg_results_by_metric = srg_eval(
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



def find_robust_hyperparameters(
    results: List[Dict],
    decision_metric: str,
) -> Tuple[Dict, pd.DataFrame, Dict]:
    """
    Finds robust hyperparameters based on a specific decision metric, but analyzes all metrics.
    """
    df = pd.DataFrame(results)
    df = df.fillna("NA") # For the None's
    
    # Group by gamma parameters
    parameter_groups = df.groupby(['conv_gamma', 'lin_gamma', 'distance_metric', 'proxy_temp','metric_name', 'topk'])    
    analysis_data = []

    for (conv_gamma, lin_gamma, distance_metric, proxy_temp, metric_name, topk), group in parameter_groups:
        scores = group['faithfulness_score'].values #can potentially include NaN's
        num_valid = np.sum(~np.isnan(scores))
        num_invalid = np.sum(np.isnan(scores))

        mean_score = np.nanmean(scores)
        std_score = np.nanstd(scores)
        min_score = np.nanmin(scores)
        stability = 1 / (1 + std_score)

        # Robustness score
        if num_valid == 0:
            stability = 0
            robustness_score = -1 
        else:
            stability = 1 / (1 + std_score)
            robustness_score = (
                0.3 * min_score +
                0.3 * stability +
                0.4 * mean_score
            )
        
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

    
    # Select the best parameters based on the specified decision_metric
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
        'metric_name': best_row['metric_name'], # The metric used for this decision
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
        'metric_name': worst_row['metric_name'], # The metric used for this decision
        'robustness_score': worst_row['robustness_score'],
        'mean_score': worst_row['raw_mean'],
        'min_score': worst_row['raw_min'],
        'std_score': worst_row['raw_std'],
    }

    return best_params_raw, analysis_df, worst_params_raw


def visualize_robustness_analysis(
    analysis_df: pd.DataFrame,
    hpi_colors: Dict[str, float],
    save_path: str = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/robustness_analysis/robustness_analysis.png",
) -> List[str]:
    """Creates visualizations for each combination of distance and evaluation metric."""    
    path_root, path_ext = os.path.splitext(save_path)
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []

    plots = [
        ('Mean Faithfulness', 'raw_mean'),
        ('Minimum Faithfulness (Worst-Case)', 'raw_min'),
        ('Overall Robustness Score', 'robustness_score')
    ]

    # nur valide Werte nehmen (kein None / NaN)
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

                    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
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
        print(f"  {i}. γ_conv={row['conv_gamma']}, γ_lin={row['lin_gamma']}, dist={row['distance_metric']}, proxy_temp={row['proxy_temp']}: "
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
    
    # Combine all overall stats
    overall_stats = {**raw_stats}
    
    # Statistics by gamma combination for both score types
    parameter_stats_raw = df.groupby(['conv_gamma', 'lin_gamma', 'distance_metric', 'proxy_temp'])['faithfulness_score'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    parameter_stats_raw.columns = [f'raw_{col}' if col not in ['conv_gamma', 'lin_gamma', 'distance_metric', 'proxy_temp'] else col
                              for col in parameter_stats_raw.columns]
    
    # Merge gamma statistics
    parameter_stats = pd.merge(parameter_stats_raw, on=['conv_gamma', 'lin_gamma', 'distance_metric', 'proxy_temp'])

    parameter_stats = pd.merge(parameter_stats, on=['conv_gamma', 'lin_gamma', 'distance_metric', 'proxy_temp'])

    # Find best gamma combinations by different metrics
    best_by_raw_mean = parameter_stats.loc[parameter_stats['raw_mean'].idxmax()]
    best_by_raw_min = parameter_stats.loc[parameter_stats['raw_min'].idxmax()]
    
    # Statistics by image (aggregated across gamma combinations)
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

def log_nested_validation_to_wandb(
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

    # --- 2. Log Detailed Data as Tables ---
    # The 'metric_name' column will now be present in these tables
    log_payload = {}

    log_payload.update({
        "tune_set/raw_results_per_image": wandb.Table(dataframe=pd.DataFrame(tune_results_list)),
        "holdout_set/raw_results_per_image": wandb.Table(dataframe=pd.DataFrame(holdout_results_list)),
        "tune_set/parameter_analysis": wandb.Table(dataframe=analysis_df_tune),
        "holdout_set/parameter_analysis": wandb.Table(dataframe=analysis_df_holdout)
    })

    # --- 3. Log Visualizations ---
    # visualize_robustness_analysis now creates plots for all metrics
    tune_paths = visualize_robustness_analysis(analysis_df_tune, hpi_colors=hpi_colors, save_path="wandb_plots/tune_analysis.png")
    holdout_paths = visualize_robustness_analysis(analysis_df_holdout, hpi_colors=hpi_colors, save_path="wandb_plots/holdout_analysis.png")

    log_payload.update({
        "plots/tune_set_analysis_heatmaps": [wandb.Image(p, caption=os.path.basename(p)) for p in tune_paths],
        "plots/holdout_set_analysis_heatmaps": [wandb.Image(p, caption=os.path.basename(p)) for p in holdout_paths]
    })

    # --- 4. Log Curves ---
    print("\n--- Logging Faithfulness Curves for Approved Parameters (All Metrics) ---")

    needed_cols = ["conv_gamma", "lin_gamma", "proxy_temp", "distance_metric", "topk"]

    tune_curves_df = pd.DataFrame(tune_curves_list)
    holdout_curves_df = pd.DataFrame(holdout_curves_list)
    print(f"Total tune curves: {len(tune_curves_df)}, holdout curves: {len(holdout_curves_df)}")
    for col in needed_cols:
        if col not in tune_curves_df.columns:
            tune_curves_df[col] = np.nan
        if col not in holdout_curves_df.columns:
            holdout_curves_df[col] = np.nan

        if col not in approved_params:
            approved_params[col] = np.nan
        if col not in worst_params:
            worst_params[col] = np.nan
    
    # --- Approved ---
    approved_tune_curves_df = tune_curves_df[_build_filter(tune_curves_df, approved_params)].copy()
    approved_holdout_curves_df = holdout_curves_df[_build_filter(holdout_curves_df, approved_params)].copy()

    print(f"Found {len(approved_tune_curves_df)} tune curves and {len(approved_holdout_curves_df)} holdout curves for approved parameters.")

    approved_tune_curves_df['split'] = 'tune'
    approved_holdout_curves_df['split'] = 'holdout'
    combined_curves_df = pd.concat([approved_tune_curves_df, approved_holdout_curves_df])

    for eval_metric in combined_curves_df['metric_name'].unique():
        metric_df = combined_curves_df[combined_curves_df['metric_name'] == eval_metric]

        log_key_tune = f"plots/mean_curves_tune_{eval_metric}"
        tune_plot = plot_and_log_mean_curve(
            df=metric_df[metric_df['split'] == 'tune'],
            title=f"Mean Curves on Tune Set ({eval_metric})",
            log_key=log_key_tune,
            hpi_colors=hpi_colors
        )
        if tune_plot: log_payload[log_key_tune] = tune_plot

        log_key_holdout = f"plots/mean_curves_holdout_{eval_metric}"
        holdout_plot = plot_and_log_mean_curve(
            df=metric_df[metric_df['split'] == 'holdout'],
            title=f"Mean Curves on Holdout Set ({eval_metric})",
            log_key=log_key_holdout,
            hpi_colors=hpi_colors
        )
        if holdout_plot: log_payload[log_key_holdout] = holdout_plot


    # --- Worst ---
    worst_tune_df = tune_curves_df[_build_filter(tune_curves_df, worst_params)].copy()
    worst_holdout_df = holdout_curves_df[_build_filter(holdout_curves_df, worst_params)].copy()

    worst_tune_df['split'] = 'tune'
    worst_holdout_df['split'] = 'holdout'
    combined_worst_df = pd.concat([worst_tune_df, worst_holdout_df])

    for eval_metric in combined_worst_df['metric_name'].unique():
        metric_df = combined_worst_df[combined_worst_df['metric_name'] == eval_metric]

        log_key_worst_tune = f"plots/worst_curves_tune_{eval_metric}"
        worst_tune_plot = plot_and_log_mean_curve(
            df=metric_df[metric_df['split'] == 'tune'],
            title=f"Worst Params - Tune Set ({eval_metric})",
            log_key=log_key_worst_tune,
            hpi_colors=hpi_colors
        )
        if worst_tune_plot: log_payload[log_key_worst_tune] = worst_tune_plot

        log_key_worst_holdout = f"plots/worst_curves_holdout_{eval_metric}"
        worst_holdout_plot = plot_and_log_mean_curve(
            df=metric_df[metric_df['split'] == 'holdout'],
            title=f"Worst Params - Holdout Set ({eval_metric})",
            log_key=log_key_worst_holdout,
            hpi_colors=hpi_colors
        )
        if worst_holdout_plot: log_payload[log_key_worst_holdout] = worst_holdout_plot

        if not combined_curves_df.empty:
            combined_curves_df['series'] = combined_curves_df.apply(
                lambda row: f"{row['split']}/{row['metric_name']}/{row['curve_label']}", axis=1
            )
            plot_table = wandb.Table(dataframe=combined_curves_df[['series', 'step', 'score']])
            wandb_plot = wandb.plot.line(
                plot_table, x="step", y="score", stroke="series",                
                title="Faithfulness Curves (Tune vs. Holdout, All Metrics)"
            )
            log_payload["plots/aggregate_faithfulness_curves"] = wandb_plot

    # --- 5. MAKE THE SINGLE, CONSOLIDATED LOG CALL ---
    if log_payload:
        print("\n--- Logging all artifacts to W&B in a single step ---")
        wandb.log(log_payload)
    print("--- Finished logging to W&B ---")

def _build_filter(df: pd.DataFrame, params: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in params.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            # Skip this param if it's None/NaN → don't filter on it
            continue
        if col not in df.columns:
            continue
        # choose default depending on dtype
        fill_default = -1 if df[col].dtype.kind in "if" else "NA"
        mask &= (df[col].fillna(fill_default) == val)
    return mask

def plot_and_log_mean_curve(
    df: pd.DataFrame,
    title: str,
    log_key: str,
    hpi_colors: Dict[str, float],
    save_dir: str = "wandb_plots"
) -> wandb.Image:
    """
    Calculates and plots mean curves with std deviation bands from a dataframe.
    Logs the resulting plot to Weights & Biases as a static image.
    """
    print(f"Plotting mean curve for {log_key} with {len(df)} data points...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        'morf_raw': hpi_colors["red"],
        'lerf_raw': hpi_colors["yellow"],
        'random_raw': hpi_colors["gray"],
    }
    
    for label, group in df.groupby('curve_label'):
        agg_data = group.groupby('step')['score'].agg(['mean', 'std']).reset_index()
        agg_data = agg_data.sort_values('step')

        x = agg_data['step']
        mean_score = agg_data['mean']
        std_score = agg_data['std'].fillna(0)

        ax.plot(x, mean_score, label=f"Mean {label}", color=colors.get(label, 'black'))

        ax.fill_between(
            x,
            mean_score - std_score,
            mean_score + std_score,
            alpha=0.2,
            color=colors.get(label, 'gray'),
            label=f'Std Dev {label}'
        )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Fraction of Patches Perturbed (Step)", fontsize=12)
    ax.set_ylabel("Mean Score", fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Dynamische y-Achsen-Grenzen
    y_min = df['score'].min()
    y_max = df['score'].max()
    padding = (y_max - y_min) * 0.05 
    ax.set_ylim(y_min - padding, y_max + padding)

    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{log_key.replace('/', '_')}.png"
    save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path, dpi=150)
    print(f"Saved mean curve plot to: {save_path}")

    image_to_log = wandb.Image(save_path)
    
    plt.close(fig)
    return image_to_log
