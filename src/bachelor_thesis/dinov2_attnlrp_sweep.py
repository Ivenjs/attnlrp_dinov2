import itertools
import torch
import torch.nn as nn
from typing import List
import yaml
import os
from PIL import Image
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm


from lrp_helpers import compute_simple_attnlrp_pass_batched, compute_knn_attnlrp_pass_batched, LRPConservationChecker
from basemodel import TimmWrapper
from eval_helpers import srg_knn


from dino_patcher import DINOPatcher

CONV_GAMMAS = [0.1, 0.25, 1.0]
LIN_GAMMAS = [0.0, 0.05, 0.1, 0.25]

def run_gamma_sweep(
    model_wrapper: TimmWrapper, 
    dataloader: DataLoader,  
    device: torch.device,
    mode: str = "simple",
    db_embeddings: torch.Tensor = None,  
    db_filenames: List[str] = None,  
    k_neighbors: int = 5,
    distance_metrics: List[str] = ["euclidean"],
    conv_gamma_values: List[float] = CONV_GAMMAS,
    lin_gamma_values: List[float] = LIN_GAMMAS,
    verbose: bool = False
) -> Tuple[Dict[int, Dict[Tuple[float, float], torch.Tensor]], Dict[int, Dict[Tuple[float, float], Any]]]:
    #TODO: use batching?

    """
    Runs a sweep over gamma parameters for multiple inputs, managing patches efficiently.
    """
    all_relevances = defaultdict(dict)
    all_violations = defaultdict(dict)

    print("--- Starting Gamma Sweep ---")
    print("Patching model for LRP and Conservation Checking for the duration of the sweep...")

    with DINOPatcher(model_wrapper), LRPConservationChecker(model_wrapper) as checker:
        
        param_combinations = list(itertools.product(conv_gamma_values, lin_gamma_values, 
                                                     distance_metrics))
        # Loop over gamma combinations for this input
        for i, (conv_gamma, lin_gamma, distance_metric) in enumerate(param_combinations):
            print(f"\n=== Processing Param Combination {i+1}/{len(param_combinations)}: "
                  f"conv_γ={conv_gamma}, lin_γ={lin_gamma}, dist={distance_metric} ===")

            for input_batch, filename_batch in tqdm(dataloader, desc="Processing batches"):
                input_batch = input_batch.to(device)
            
                # Call the inner-loop function
                if mode == "simple":
                    relevance_batch, violation_batch = compute_simple_attnlrp_pass_batched(
                        conv_gamma=conv_gamma,
                        lin_gamma=lin_gamma,
                        model_wrapper=model_wrapper,
                        input_batch=input_batch,  # Add batch dim back
                        checker=checker,
                        verbose=verbose  
                    )
                elif mode == "knn":
                    assert db_embeddings is not None, "db_embeddings must be provided for 'knn' mode."
                    assert db_filenames is not None, "db_filenames must be provided for 'knn' mode."

                    relevance_batch, violation_batch = compute_knn_attnlrp_pass_batched(
                        conv_gamma=conv_gamma,
                        lin_gamma=lin_gamma,
                        model_wrapper=model_wrapper,
                        input_batch=input_batch,  # Add batch dim for single input
                        checker=checker,
                        verbose=verbose,
                        db_embeddings=db_embeddings,  
                        db_filenames=db_filenames,  
                        input_filename_batch=filename_batch,
                        distance_metric=distance_metric,
                        k_neighbors=k_neighbors  
                    )
            
                key = (conv_gamma, lin_gamma, distance_metric)
                for j, filename in enumerate(filename_batch):
                    all_relevances[filename][key] = relevance_batch[j].detach().cpu()
                    all_violations[filename][key] = violation_batch[j]

    print("\n--- Gamma Sweep Complete ---")
    print("Model has been restored to its original state.")

    return dict(all_relevances), dict(all_violations)

def evaluate_gamma_sweep(
    relevances_by_parameters: Dict[str, Dict[Tuple[float, float, str], torch.Tensor]], 
    violations_by_parameters: Dict[str, Dict[Tuple[float, float, str], Any]],
    evaluation_dataloader: DataLoader,
    model_wrapper,
    db_embeddings: torch.Tensor,
    db_filenames: List[str],
    patch_size: int,
    device: torch.device,
    k_neighbors: int = 5,
    plot_curves: bool = False
) -> List[Dict]:
    """
    Evaluate faithfulness scores for each image and gamma combination.
    
    Returns:
        List of dictionaries with structure:
        [
            {
                "input_idx": 0,
                "conv_gamma": 0.1,
                "lin_gamma": 0.1,
                "faithfulness_score": 0.85,
                "violations": {...}
            },
            ...
        ]
    """
    results = []
    all_curves_data = []

    # Get a sample filename to determine the parameter combinations to test
    sample_filename = next(iter(relevances_by_parameters.keys()))
    parameters_combinations = list(relevances_by_parameters[sample_filename].keys())
    
    # The dataloader will elegantly handle loading one image at a time
    for i, (input_tensor, filename_batch) in enumerate(tqdm(evaluation_dataloader, desc="Evaluating Images")):
        input_filename = filename_batch[0] # Unpack batch of size 1
        # The input_tensor is already a batch of 1: [1, C, H, W]
        
        print(f"\n=== Evaluating Image {i + 1}/{len(evaluation_dataloader.dataset)}: {input_filename} ===")
        
        for parameters in parameters_combinations:
            conv_gamma, lin_gamma, distance_metric = parameters

            if parameters not in relevances_by_parameters[input_filename]:
                print(f"  Skipping params {parameters} for {input_filename} (not found).")
                continue

            relevance_map = relevances_by_parameters[input_filename][parameters]
            violations = violations_by_parameters[input_filename][parameters]

            print(f"  Evaluating γ_conv={conv_gamma}, γ_lin={lin_gamma}, dist={distance_metric}")
            
            srg_results = srg_knn(
                relevance_map=relevance_map,
                input_tensor=input_tensor.to(device), 
                model=model_wrapper,
                patch_size=patch_size,
                db_embeddings=db_embeddings,
                db_filenames=db_filenames,
                input_filename=input_filename,
                distance_metric=distance_metric,
                k_neighbors=k_neighbors,
                plot_curves=plot_curves
            )
            
            results.append({
                "image": input_filename,
                "conv_gamma": conv_gamma,
                "lin_gamma": lin_gamma,
                "distance_metric": distance_metric,
                "faithfulness_score": srg_results["faithfulness_score"],
                "normalized_faithfulness_score": srg_results["normalized_faithfulness_score"],
                "violations": violations,
                "auc_morf": srg_results["auc_morf"],
                "auc_lerf": srg_results["auc_lerf"],
            })

            all_curve_sets = [
                ("morf_raw", srg_results["morf_curve"]),
                ("lerf_raw", srg_results["lerf_curve"]),
                ("morf_norm", srg_results["morf_curve_norm"]),
                ("lerf_norm", srg_results["lerf_curve_norm"])
            ]

            for curve_label, curve in all_curve_sets:
                for step, score in enumerate(curve):
                    all_curves_data.append([
                        input_filename,
                        conv_gamma,
                        lin_gamma,
                        distance_metric,
                        curve_label,  # This now contains "morf_raw", "lerf_norm", etc.
                        step, 
                        score 
                    ])
    
    return results, all_curves_data



def find_robust_hyperparameters(
    results: List[Dict],
    robustness_percentile: float = 0.9,
    min_score_threshold: float = 0.0
) -> Tuple[Dict, Dict, pd.DataFrame]:
    """
    Find hyperparameters that are robust across images for both raw and normalized scores.
    
    Args:
        results: List of evaluation results from evaluate_multi_image_gamma_sweep
        robustness_percentile: What percentile of cases should be "good" (0.9 = 90%)
        min_score_threshold: Minimum acceptable faithfulness score
    
    Returns:
        best_params_raw: Dictionary with best hyperparameters for raw scores
        best_params_normalized: Dictionary with best hyperparameters for normalized scores
        analysis_df: DataFrame with detailed analysis for both score types
    """
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Group by gamma parameters
    parameter_groups = df.groupby(['conv_gamma', 'lin_gamma', 'distance_metric'])
    
    analysis_data = []

    for (conv_gamma, lin_gamma, distance_metric), group in parameter_groups:
        # Raw score analysis
        raw_scores = group['faithfulness_score'].values
        raw_mean = np.mean(raw_scores)
        raw_std = np.std(raw_scores)
        raw_min = np.min(raw_scores)
        raw_percentile = np.percentile(raw_scores, robustness_percentile * 100)
        raw_good_scores = np.sum(raw_scores >= min_score_threshold)
        raw_robustness_ratio = raw_good_scores / len(raw_scores)
        raw_stability = 1 / (1 + raw_std)
        
        # Raw robustness score
        raw_robustness_score = (
            0.4 * raw_min +                # Worst case shouldn't be terrible
            0.3 * raw_stability +          # Low variance is good
            0.2 * raw_mean +               # Good average performance
            0.1 * raw_robustness_ratio     # High success rate
        )
        
        # Normalized score analysis
        norm_scores = group['normalized_faithfulness_score'].values
        norm_mean = np.mean(norm_scores)
        norm_std = np.std(norm_scores)
        norm_min = np.min(norm_scores)
        norm_percentile = np.percentile(norm_scores, robustness_percentile * 100)
        norm_good_scores = np.sum(norm_scores >= min_score_threshold)
        norm_robustness_ratio = norm_good_scores / len(norm_scores)
        norm_stability = 1 / (1 + norm_std)
        
        # Normalized robustness score
        norm_robustness_score = (
            0.4 * norm_min +               # Worst case shouldn't be terrible
            0.3 * norm_stability +         # Low variance is good
            0.2 * norm_mean +              # Good average performance
            0.1 * norm_robustness_ratio    # High success rate
        )
        
        # Score consistency (correlation between raw and normalized)
        score_correlation = np.corrcoef(raw_scores, norm_scores)[0, 1] if len(raw_scores) > 1 else 1.0
        
        analysis_data.append({
            'conv_gamma': conv_gamma,
            'lin_gamma': lin_gamma,
            'distance_metric': distance_metric,
            
            # Raw score metrics
            'raw_mean': raw_mean,
            'raw_std': raw_std,
            'raw_min': raw_min,
            'raw_percentile': raw_percentile,
            'raw_robustness_ratio': raw_robustness_ratio,
            'raw_stability': raw_stability,
            'raw_robustness_score': raw_robustness_score,
            
            # Normalized score metrics
            'norm_mean': norm_mean,
            'norm_std': norm_std,
            'norm_min': norm_min,
            'norm_percentile': norm_percentile,
            'norm_robustness_ratio': norm_robustness_ratio,
            'norm_stability': norm_stability,
            'norm_robustness_score': norm_robustness_score,
            
            # Meta metrics
            'score_correlation': score_correlation,
            'num_images': len(raw_scores)
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    
    # Find best parameters for raw scores
    best_raw_idx = analysis_df['raw_robustness_score'].idxmax()
    best_params_raw = {
        'conv_gamma': analysis_df.loc[best_raw_idx, 'conv_gamma'],
        'lin_gamma': analysis_df.loc[best_raw_idx, 'lin_gamma'],
        'distance_metric': analysis_df.loc[best_raw_idx, 'distance_metric'],
        'robustness_score': analysis_df.loc[best_raw_idx, 'raw_robustness_score'],
        'mean_score': analysis_df.loc[best_raw_idx, 'raw_mean'],
        'min_score': analysis_df.loc[best_raw_idx, 'raw_min'],
        'std_score': analysis_df.loc[best_raw_idx, 'raw_std'],
        'stability': analysis_df.loc[best_raw_idx, 'raw_stability'],
        'robustness_ratio': analysis_df.loc[best_raw_idx, 'raw_robustness_ratio']
    }
    
    # Find best parameters for normalized scores
    best_norm_idx = analysis_df['norm_robustness_score'].idxmax()
    best_params_normalized = {
        'conv_gamma': analysis_df.loc[best_norm_idx, 'conv_gamma'],
        'lin_gamma': analysis_df.loc[best_norm_idx, 'lin_gamma'],
        'distance_metric': analysis_df.loc[best_norm_idx, 'distance_metric'],
        'robustness_score': analysis_df.loc[best_norm_idx, 'norm_robustness_score'],
        'mean_score': analysis_df.loc[best_norm_idx, 'norm_mean'],
        'min_score': analysis_df.loc[best_norm_idx, 'norm_min'],
        'std_score': analysis_df.loc[best_norm_idx, 'norm_std'],
        'stability': analysis_df.loc[best_norm_idx, 'norm_stability'],
        'robustness_ratio': analysis_df.loc[best_norm_idx, 'norm_robustness_ratio']
    }
    
    return best_params_raw, best_params_normalized, analysis_df


def visualize_robustness_analysis(
    analysis_df: pd.DataFrame,
    save_path: str = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/robustness_analysis/robustness_analysis.png"
):
    """
    Create visualizations for the robustness analysis, generating a separate figure
    for each distance metric found in the 'distance_metric' column.

    Args:
        analysis_df (pd.DataFrame): DataFrame containing the analysis results. 
                                    Must include 'conv_gamma', 'lin_gamma', 'distance_metric',
                                    and the score columns ('raw_mean', 'norm_mean', etc.).
        save_path (str): The base path for saving the plots. The distance metric name 
                         will be appended to this path.
    """
    # 1. Get the unique distance metrics from the DataFrame
    distance_metrics = analysis_df['distance_metric'].unique()
    
    # Get the base path and extension for saving files
    path_root, path_ext = os.path.splitext(save_path)

    # 2. Loop over each distance metric
    for metric in distance_metrics:
        print(f"--- Generating plots for distance metric: {metric} ---")
        
        # 3. Filter the DataFrame for the current metric
        df_subset = analysis_df[analysis_df['distance_metric'] == metric].copy()

        # Create a new figure for this metric's plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # Add a clear, overarching title for the figure
        fig.suptitle(f'Robustness Analysis (Distance Metric: {metric.title()})', fontsize=20, y=1.02)
        
        # --- Plotting logic (same as before, but uses df_subset) ---
        
        # 1. Raw mean scores
        pivot_raw_mean = df_subset.pivot(index='conv_gamma', columns='lin_gamma', values='raw_mean')
        sns.heatmap(pivot_raw_mean, annot=True, fmt='.3f', cmap='viridis', ax=axes[0,0])
        axes[0,0].set_title('Raw Mean Faithfulness Scores')
        
        # 2. Normalized mean scores
        pivot_norm_mean = df_subset.pivot(index='conv_gamma', columns='lin_gamma', values='norm_mean')
        sns.heatmap(pivot_norm_mean, annot=True, fmt='.3f', cmap='viridis', ax=axes[0,1])
        axes[0,1].set_title('Normalized Mean Faithfulness Scores')
        
        # 3. Raw robustness scores
        pivot_raw_robust = df_subset.pivot(index='conv_gamma', columns='lin_gamma', values='raw_robustness_score')
        sns.heatmap(pivot_raw_robust, annot=True, fmt='.3f', cmap='viridis', ax=axes[1,0])
        axes[1,0].set_title('Raw Robustness Scores')
        
        # 4. Normalized robustness scores
        pivot_norm_robust = df_subset.pivot(index='conv_gamma', columns='lin_gamma', values='norm_robustness_score')
        sns.heatmap(pivot_norm_robust, annot=True, fmt='.3f', cmap='viridis', ax=axes[1,1])
        axes[1,1].set_title('Normalized Robustness Scores')
        
        # 5. Raw minimum scores (worst-case robustness)
        pivot_raw_min = df_subset.pivot(index='conv_gamma', columns='lin_gamma', values='raw_min')
        sns.heatmap(pivot_raw_min, annot=True, fmt='.3f', cmap='viridis', ax=axes[2,0])
        axes[2,0].set_title('Raw Minimum Scores (Worst-Case)')
        
        # 6. Normalized minimum scores (worst-case robustness)
        pivot_norm_min = df_subset.pivot(index='conv_gamma', columns='lin_gamma', values='norm_min')
        sns.heatmap(pivot_norm_min, annot=True, fmt='.3f', cmap='viridis', ax=axes[2,1])
        axes[2,1].set_title('Normalized Minimum Scores (Worst-Case)')
        
        # Adjust layout to prevent titles from overlapping
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to make space for suptitle
        
        # 4. Create a unique save path for the current metric's plot
        metric_save_path = f"{path_root}_{metric}{path_ext}"
        
        # Save and show the plot for the current metric
        print(f"Saving plot to: {metric_save_path}")
        plt.savefig(metric_save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Close the current figure to free up memory before the next loop iteration
        plt.close(fig)


def print_robustness_summary(
    best_params_raw: Dict, 
    best_params_normalized: Dict, 
    analysis_df: pd.DataFrame, 
    results: List[Dict] = None
):
    """
    Print a summary of the robustness analysis for both raw and normalized scores.
    """
    print("\n" + "="*60)
    print("ROBUSTNESS ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n{'='*25} RAW SCORES {'='*25}")
    print(f"Best Hyperparameters (Raw Score Robustness):")
    print(f"  Conv Gamma: {best_params_raw['conv_gamma']}")
    print(f"  Lin Gamma:  {best_params_raw['lin_gamma']}")
    print(f"  Distance Metric: {best_params_raw['distance_metric']}")
    print(f"  Robustness Score: {best_params_raw['robustness_score']:.4f}")
    
    print(f"\nRaw Score Performance Metrics:")
    print(f"  Mean Score:       {best_params_raw['mean_score']:.4f}")
    print(f"  Min Score:        {best_params_raw['min_score']:.4f}")
    print(f"  Std Score:        {best_params_raw['std_score']:.4f}")
    print(f"  Stability:        {best_params_raw['stability']:.4f}")
    print(f"  Robustness Ratio: {best_params_raw['robustness_ratio']:.4f}")
    
    print(f"\n{'='*22} NORMALIZED SCORES {'='*22}")
    print(f"Best Hyperparameters (Normalized Score Robustness):")
    print(f"  Conv Gamma: {best_params_normalized['conv_gamma']}")
    print(f"  Lin Gamma:  {best_params_normalized['lin_gamma']}")
    print(f"  Distance Metric: {best_params_normalized['distance_metric']}")
    print(f"  Robustness Score: {best_params_normalized['robustness_score']:.4f}")
    
    print(f"\nNormalized Score Performance Metrics:")
    print(f"  Mean Score:       {best_params_normalized['mean_score']:.4f}")
    print(f"  Min Score:        {best_params_normalized['min_score']:.4f}")
    print(f"  Std Score:        {best_params_normalized['std_score']:.4f}")
    print(f"  Stability:        {best_params_normalized['stability']:.4f}")
    print(f"  Robustness Ratio: {best_params_normalized['robustness_ratio']:.4f}")
    
    # Compare if they suggest the same hyperparameters
    same_params = (best_params_raw['conv_gamma'] == best_params_normalized['conv_gamma'] and 
                   best_params_raw['lin_gamma'] == best_params_normalized['lin_gamma'] and
                     best_params_raw['distance_metric'] == best_params_normalized['distance_metric'])
    
    print(f"\n{'='*25} COMPARISON {'='*25}")
    print(f"Same optimal hyperparameters: {'Yes' if same_params else 'No'}")
    
    if not same_params:
        print(f"Raw optimal: γ_conv={best_params_raw['conv_gamma']}, γ_lin={best_params_raw['lin_gamma']}, "
              f"distance_metric={best_params_raw['distance_metric']}")
        print(f"Normalized optimal: γ_conv={best_params_normalized['conv_gamma']}, γ_lin={best_params_normalized['lin_gamma']}, "
              f"distance_metric={best_params_normalized['distance_metric']}")
    
    # Show top 3 combinations for each score type
    print(f"\nTop 3 Raw Score Combinations:")
    top_3_raw = analysis_df.nlargest(3, 'raw_robustness_score')
    for i, (_, row) in enumerate(top_3_raw.iterrows(), 1):
        print(f"  {i}. γ_conv={row['conv_gamma']}, γ_lin={row['lin_gamma']}, distance_metric={row['distance_metric']}: "
              f"robust={row['raw_robustness_score']:.4f}, "
              f"mean={row['raw_mean']:.4f}, "
              f"min={row['raw_min']:.4f}")
    
    print(f"\nTop 3 Normalized Score Combinations:")
    top_3_norm = analysis_df.nlargest(3, 'norm_robustness_score')
    for i, (_, row) in enumerate(top_3_norm.iterrows(), 1):
        print(f"  {i}. γ_conv={row['conv_gamma']}, γ_lin={row['lin_gamma']}, distance_metric={row['distance_metric']}: "
              f"robust={row['norm_robustness_score']:.4f}, "
              f"mean={row['norm_mean']:.4f}, "
              f"min={row['norm_min']:.4f}")
    
    # Show most stable combinations for each score type
    print(f"\nMost Stable Raw Score Combinations:")
    stable_3_raw = analysis_df.nlargest(3, 'raw_stability')
    for i, (_, row) in enumerate(stable_3_raw.iterrows(), 1):
        print(f"  {i}. γ_conv={row['conv_gamma']}, γ_lin={row['lin_gamma']}, distance_metric={row['distance_metric']}: "
              f"stability={row['raw_stability']:.4f}, "
              f"std={row['raw_std']:.4f}")
    
    print(f"\nMost Stable Normalized Score Combinations:")
    stable_3_norm = analysis_df.nlargest(3, 'norm_stability')
    for i, (_, row) in enumerate(stable_3_norm.iterrows(), 1):
        print(f"  {i}. γ_conv={row['conv_gamma']}, γ_lin={row['lin_gamma']}, distance_metric={row['distance_metric']}: "
              f"stability={row['norm_stability']:.4f}, "
              f"std={row['norm_std']:.4f}")
    
    # Add aggregate statistics if results are provided
    if results is not None:
        print(f"\n" + "="*60)
        print("AGGREGATE STATISTICS")
        print("="*60)
        
        stats = calculate_aggregate_statistics(results)
        overall = stats['overall']
        
        print(f"\nOverall Raw Score Performance:")
        print(f"  Mean:     {overall['raw_mean']:.4f}")
        print(f"  Median:   {overall['raw_median']:.4f}")
        print(f"  Std Dev:  {overall['raw_std']:.4f}")
        print(f"  Min:      {overall['raw_min']:.4f}")
        print(f"  Max:      {overall['raw_max']:.4f}")
        
        print(f"\nOverall Normalized Score Performance:")
        print(f"  Mean:     {overall['norm_mean']:.4f}")
        print(f"  Median:   {overall['norm_median']:.4f}")
        print(f"  Std Dev:  {overall['norm_std']:.4f}")
        print(f"  Min:      {overall['norm_min']:.4f}")
        print(f"  Max:      {overall['norm_max']:.4f}")
        
        print(f"\nScore Correlation Analysis:")
        print(f"  Raw-Normalized Correlation: {overall['score_correlation']:.4f}")
        print(f"  Mean Absolute Difference:   {overall['mean_abs_diff']:.4f}")
        
        return stats

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
    
    # Overall statistics for normalized scores
    norm_stats = {
        'norm_mean': df['normalized_faithfulness_score'].mean(),
        'norm_median': df['normalized_faithfulness_score'].median(),
        'norm_std': df['normalized_faithfulness_score'].std(),
        'norm_min': df['normalized_faithfulness_score'].min(),
        'norm_max': df['normalized_faithfulness_score'].max(),
        'norm_q25': df['normalized_faithfulness_score'].quantile(0.25),
        'norm_q75': df['normalized_faithfulness_score'].quantile(0.75)
    }
    
    # Correlation and consistency metrics
    correlation_stats = {
        'score_correlation': df['faithfulness_score'].corr(df['normalized_faithfulness_score']),
        'mean_abs_diff': (df['faithfulness_score'] - df['normalized_faithfulness_score']).abs().mean(),
        'median_abs_diff': (df['faithfulness_score'] - df['normalized_faithfulness_score']).abs().median()
    }
    
    # Combine all overall stats
    overall_stats = {**raw_stats, **norm_stats, **correlation_stats}
    
    # Statistics by gamma combination for both score types
    parameter_stats_raw = df.groupby(['conv_gamma', 'lin_gamma', 'distance_metric'])['faithfulness_score'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    parameter_stats_raw.columns = [f'raw_{col}' if col not in ['conv_gamma', 'lin_gamma', 'distance_metric'] else col
                              for col in parameter_stats_raw.columns]

    parameter_stats_norm = df.groupby(['conv_gamma', 'lin_gamma', 'distance_metric'])['normalized_faithfulness_score'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    parameter_stats_norm.columns = [f'norm_{col}' if col not in ['conv_gamma', 'lin_gamma', 'distance_metric'] else col
                               for col in parameter_stats_norm.columns]
    
    # Merge gamma statistics
    parameter_stats = pd.merge(parameter_stats_raw, parameter_stats_norm, on=['conv_gamma', 'lin_gamma', 'distance_metric'])

    # Add correlation by gamma combination
    parameter_correlations = df.groupby(['conv_gamma', 'lin_gamma', 'distance_metric']).apply(
        lambda x: x['faithfulness_score'].corr(x['normalized_faithfulness_score'])
    ).reset_index(name='score_correlation')

    parameter_stats = pd.merge(parameter_stats, parameter_correlations, on=['conv_gamma', 'lin_gamma', 'distance_metric'])

    # Find best gamma combinations by different metrics
    best_by_raw_mean = parameter_stats.loc[parameter_stats['raw_mean'].idxmax()]
    best_by_norm_mean = parameter_stats.loc[parameter_stats['norm_mean'].idxmax()]
    best_by_raw_min = parameter_stats.loc[parameter_stats['raw_min'].idxmax()]
    best_by_norm_min = parameter_stats.loc[parameter_stats['norm_min'].idxmax()]

    valid_corrs = parameter_stats['score_correlation'].dropna()
    if not valid_corrs.empty:
        best_by_correlation = parameter_stats.loc[valid_corrs.idxmax()]
    else:
        best_by_correlation = None  # oder irgendein Default
        print("No valid correlation scores found – every group had ≤1 sample.")
    
    # Statistics by image (aggregated across gamma combinations)
    image_stats_raw = df.groupby(['image'])['faithfulness_score'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    image_stats_raw.columns = [f'raw_{col}' if col != 'image' else col 
                              for col in image_stats_raw.columns]
    
    image_stats_norm = df.groupby(['image'])['normalized_faithfulness_score'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    image_stats_norm.columns = [f'norm_{col}' if col != 'image' else col 
                               for col in image_stats_norm.columns]

    image_stats = pd.merge(image_stats_raw, image_stats_norm, on=['image'])

    return {
        'overall': overall_stats,
        'by_parameter': parameter_stats,
        'by_image': image_stats,
        'best_by_raw_mean': best_by_raw_mean,
        'best_by_norm_mean': best_by_norm_mean,
        'best_by_raw_min': best_by_raw_min,
        'best_by_norm_min': best_by_norm_min,
        'best_by_correlation': best_by_correlation
    }

def log_sweep_to_wandb(
    results: List[Dict],
    analysis_df: pd.DataFrame,
    all_curves_data: List,
    best_params_raw: Dict,
    best_params_normalized: Dict,
    aggregate_stats: Dict
):
    """
    Logs all results from a gamma sweep to W&B, including pre-configured plots.
    """


    print("\n--- Logging results to Weights & Biases ---")

    # --- 1. Log Summary Metrics ---
    wandb.summary["best_raw_conv_gamma"] = best_params_raw['conv_gamma']
    wandb.summary["best_raw_lin_gamma"] = best_params_raw['lin_gamma']
    wandb.summary["best_raw_distance_metric"] = best_params_raw['distance_metric']
    wandb.summary["best_raw_robustness_score"] = best_params_raw['robustness_score']
    wandb.summary["best_raw_mean_score"] = best_params_raw['mean_score']
    wandb.summary["best_raw_min_score"] = best_params_raw['min_score']
    wandb.summary["best_raw_std_score"] = best_params_raw['std_score']
    wandb.summary["best_raw_stability"] = best_params_raw['stability']
    wandb.summary["best_raw_robustness_ratio"] = best_params_raw['robustness_ratio']
    wandb.summary["best_norm_conv_gamma"] = best_params_normalized['conv_gamma']
    wandb.summary["best_norm_lin_gamma"] = best_params_normalized['lin_gamma']
    wandb.summary["best_norm_distance_metric"] = best_params_normalized['distance_metric']
    wandb.summary["best_norm_robustness_score"] = best_params_normalized['robustness_score']
    wandb.summary["best_norm_mean_score"] = best_params_normalized['mean_score']
    wandb.summary["best_norm_min_score"] = best_params_normalized['min_score']
    wandb.summary["best_norm_std_score"] = best_params_normalized['std_score']
    wandb.summary["best_norm_stability"] = best_params_normalized['stability']
    wandb.summary["best_norm_robustness_ratio"] = best_params_normalized['robustness_ratio']
    
    # --- 2. Create the Data Tables ---

    # === FIX IS HERE ===
    # First, create a raw DataFrame from your list of dictionaries
    raw_df = pd.DataFrame(results)

    # Now, handle the nested 'violations' column
    # This will create a new DataFrame from the 'violations' dictionaries
    violations_flat_df = pd.json_normalize(raw_df['violations'])
    # It's good practice to add a prefix to the new columns
    violations_flat_df = violations_flat_df.add_prefix('violations.')

    # Drop the original nested column and join the new flattened columns
    # axis=1 means we are concatenating columns side-by-side
    detailed_results_df = pd.concat(
        [raw_df.drop(columns=['violations']), violations_flat_df], 
        axis=1
    )
    # detailed_results_df now has a flat structure, with NaN for any missing values,
    # which wandb.Table can handle perfectly.
    
    # Now create the wandb.Table from the cleaned DataFrame
    detailed_results_table = wandb.Table(dataframe=detailed_results_df)
    # ====================

    analysis_table = wandb.Table(dataframe=analysis_df)
    
    curves_df = pd.DataFrame(
        all_curves_data,
        columns=["image", "conv_gamma", "lin_gamma", "distance_metric", "curve_label", "step", "score"]
    )
    faithfulness_curves_table = wandb.Table(dataframe=curves_df)

    raw_curves_df = curves_df[curves_df['curve_label'].str.contains('_raw')]
    norm_curves_df = curves_df[curves_df['curve_label'].str.contains('_norm')]
    
    # --- LOG RAW CURVES ---
    for i, ((image, conv_g, lin_g, dist_m), group) in enumerate(raw_curves_df.groupby(['image', 'conv_gamma', 'lin_gamma', 'distance_metric'])):
        plot_key = f"raw_curves_per_run/plot_{i}"
        
        # 2. The title can remain long and descriptive
        title = f"Img: {image}, γ_c={conv_g}, γ_l={lin_g}, dist={dist_m}"
        
        # The rest of your code is perfect
        table_for_plot = wandb.Table(dataframe=group)
        wandb.log({
            plot_key: wandb.plot.line(
                table_for_plot,
                x="step",
                y="score",
                stroke="curve_label", 
                title=title
            )
        })

    # --- LOG NORMALIZED CURVES ---
    for i, ((image, conv_g, lin_g, dist_m), group) in enumerate(norm_curves_df.groupby(['image', 'conv_gamma', 'lin_gamma', 'distance_metric'])):
        plot_key = f"norm_curves_per_run/plot_{i}"

        # 2. The title can remain long and descriptive
        title = f"Img: {image}, γ_c={conv_g}, γ_l={lin_g}, dist={dist_m}"
        
        # The rest of your code is perfect
        table_for_plot = wandb.Table(dataframe=group)
        wandb.log({
            plot_key: wandb.plot.line(
                table_for_plot,
                x="step",
                y="score",
                stroke="curve_label", 
                title=title
            )
    })
    

    # --- 4. Log the Heatmaps and Other Tables ---
    analysis_plot_path = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/robustness_analysis/robustness_analysis.png"
    if os.path.exists(analysis_plot_path):
        wandb.log({"summary_plots/robustness_heatmaps": wandb.Image(analysis_plot_path)})
    
    wandb.log({
        "data_tables/detailed_results": detailed_results_table,
        "data_tables/robustness_analysis_by_parameter": analysis_table,
        "data_tables/faithfulness_curves_raw_data": faithfulness_curves_table
    })

    # --- 5. Log Artifacts ---
    print("Creating and logging artifact...")
    artifact = wandb.Artifact('robustness-sweep-results', type='analysis-results')
    artifact.add(detailed_results_table, "detailed_results")
    artifact.add(analysis_table, "robustness_analysis_by_parameter")
    artifact.add(faithfulness_curves_table, "faithfulness_curves_raw_data")
    wandb.log_artifact(artifact)

    print("--- Finished logging to W&B ---")
