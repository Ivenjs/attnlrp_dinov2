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

from lrp_helpers import compute_simple_attnlrp_pass, compute_knn_attnlrp_pass, LRPConservationChecker
from basemodel import TimmWrapper
from eval_helpers import srg_knn


from dino_patcher import DINOPatcher

CONV_GAMMAS = [0.1, 0.25, 1.0]
LIN_GAMMAS = [0.0, 0.05, 0.1, 0.25]

def run_gamma_sweep(
    model_wrapper: TimmWrapper, 
    input_tensors: torch.Tensor,  # Now a batched tensor [batch_size, C, H, W]
    mode: str = "simple",
    db_embeddings: torch.Tensor = None,  
    db_labels: List[str] = None,  
    ground_truth_labels: List[str] = None,
    k_neighbors: int = 5,
    conv_gamma_values: List[float] = CONV_GAMMAS,
    lin_gamma_values: List[float] = LIN_GAMMAS,
    verbose: bool = False
) -> Tuple[Dict[int, Dict[Tuple[float, float], torch.Tensor]], Dict[int, Dict[Tuple[float, float], Any]]]:
    #TODO: use batching?

    """
    Runs a sweep over gamma parameters for multiple inputs, managing patches efficiently.
    """
    all_relevances = {}
    all_violations = {}

    print("--- Starting Gamma Sweep ---")
    print("Patching model for LRP and Conservation Checking for the duration of the sweep...")

    with DINOPatcher(model_wrapper), LRPConservationChecker(model_wrapper) as checker:
        
        param_combinations = list(itertools.product(conv_gamma_values, lin_gamma_values))
        
        # Loop over each input in the batch
        for input_idx in range(input_tensors.shape[0]):  # Changed this line
            print(f"\n=== Processing Input {input_idx + 1}/{input_tensors.shape[0]} ===")
            
            # Extract single input from batch
            input_tensor = input_tensors[input_idx]  # Shape: [C, H, W]
            
            # Get the corresponding ground truth label
            ground_truth_label = ground_truth_labels[input_idx] if ground_truth_labels else None
            
            # Initialize storage for this input
            all_relevances[input_idx] = {}
            all_violations[input_idx] = {}
            
            # Loop over gamma combinations for this input
            for i, (conv_gamma, lin_gamma) in enumerate(param_combinations):
                print(f"--- Running Pass {i+1}/{len(param_combinations)} for Input {input_idx + 1} ---")
                
                # Call the inner-loop function
                if mode == "simple":
                    relevance, violations = compute_simple_attnlrp_pass(
                        conv_gamma=conv_gamma,
                        lin_gamma=lin_gamma,
                        model_wrapper=model_wrapper,
                        input_tensor=input_tensor.unsqueeze(0),  # Add batch dim back
                        checker=checker,
                        verbose=verbose  
                    )
                elif mode == "knn":
                    assert db_embeddings is not None, "db_embeddings must be provided for 'knn' mode."
                    assert db_labels is not None, "db_labels must be provided for 'knn' mode."
                    assert ground_truth_label is not None, "ground_truth_label must be provided for 'knn' mode."

                    relevance, violations = compute_knn_attnlrp_pass(
                        conv_gamma=conv_gamma,
                        lin_gamma=lin_gamma,
                        model_wrapper=model_wrapper,
                        input_tensor=input_tensor.unsqueeze(0),  # Add batch dim for single input
                        checker=checker,
                        verbose=verbose,
                        db_embeddings=db_embeddings,  
                        db_labels=db_labels,  
                        ground_truth_label=ground_truth_label,  
                        k_neighbors=k_neighbors  
                    )
                
                # Store the results for this input and gamma combination
                key = (conv_gamma, lin_gamma)
                all_relevances[input_idx][key] = relevance.detach().cpu()
                all_violations[input_idx][key] = violations

    print("\n--- Gamma Sweep Complete ---")
    print("Model has been restored to its original state.")
    
    return all_relevances, all_violations

def evaluate_gamma_sweep(
    relevances_by_gamma: Dict[int, Dict[Tuple[float, float], torch.Tensor]], 
    violations_by_gamma: Dict[int, Dict[Tuple[float, float], Any]],
    input_tensors: torch.Tensor,
    ground_truth_labels: List[str],
    model_wrapper,
    db_embeddings: torch.Tensor,
    db_labels: List[str],
    patch_size: int,
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

    # Get all gamma combinations from the first image
    gamma_combinations = list(relevances_by_gamma[0].keys())
    
    for input_idx in range(input_tensors.shape[0]):
        print(f"\n=== Evaluating Image {input_idx + 1}/{input_tensors.shape[0]} ===")
        
        input_tensor = input_tensors[input_idx].unsqueeze(0)  # Add batch dim
        ground_truth_label = ground_truth_labels[input_idx]
        
        for gammas in gamma_combinations:
            conv_gamma, lin_gamma = gammas
            
            # Get relevance map for this image and gamma combination
            relevance_map = relevances_by_gamma[input_idx][gammas]
            violations = violations_by_gamma[input_idx][gammas]
            
            print(f"  Evaluating γ_conv={conv_gamma}, γ_lin={lin_gamma}")
            
            # Compute faithfulness score
            srg_results = srg_knn(
                relevance_map=relevance_map,
                input_tensor=input_tensor,
                model=model_wrapper,
                patch_size=patch_size,
                db_embeddings=db_embeddings,
                db_labels=db_labels,
                ground_truth_label=ground_truth_label,
                k_neighbors=k_neighbors,
                plot_curves=plot_curves
            )
            
            results.append({
                "input_idx": input_idx,
                "image_label": ground_truth_label,
                "conv_gamma": conv_gamma,
                "lin_gamma": lin_gamma,
                "faithfulness_score": srg_results["faithfulness_score"],
                "normalized_faithfulness_score": srg_results["normalized_faithfulness_score"],
                "violations": violations,
                "auc_morf": srg_results["auc_morf"],
                "auc_lerf": srg_results["auc_lerf"],
            })

            for curve_type, curve in [("morf", srg_results["morf_curve"]), ("lerf", srg_results["lerf_curve"])]:
                for step, score in enumerate(curve):
                    all_curves_data.append([
                        input_idx,
                        ground_truth_label,
                        conv_gamma,
                        lin_gamma,
                        curve_type,
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
    gamma_groups = df.groupby(['conv_gamma', 'lin_gamma'])
    
    analysis_data = []
    
    for (conv_gamma, lin_gamma), group in gamma_groups:
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
    results: List[Dict],
    save_path: str = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/robustness_analysis/robustness_analysis.png"
):
    """
    Create visualizations for the robustness analysis showing both raw and normalized scores.
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. Raw mean scores
    pivot_raw_mean = analysis_df.pivot(index='conv_gamma', columns='lin_gamma', values='raw_mean')
    sns.heatmap(pivot_raw_mean, annot=True, fmt='.3f', cmap='viridis', ax=axes[0,0])
    axes[0,0].set_title('Raw Mean Faithfulness Scores')
    
    # 2. Normalized mean scores
    pivot_norm_mean = analysis_df.pivot(index='conv_gamma', columns='lin_gamma', values='norm_mean')
    sns.heatmap(pivot_norm_mean, annot=True, fmt='.3f', cmap='viridis', ax=axes[0,1])
    axes[0,1].set_title('Normalized Mean Faithfulness Scores')
    
    # 3. Raw robustness scores
    pivot_raw_robust = analysis_df.pivot(index='conv_gamma', columns='lin_gamma', values='raw_robustness_score')
    sns.heatmap(pivot_raw_robust, annot=True, fmt='.3f', cmap='viridis', ax=axes[1,0])
    axes[1,0].set_title('Raw Robustness Scores')
    
    # 4. Normalized robustness scores
    pivot_norm_robust = analysis_df.pivot(index='conv_gamma', columns='lin_gamma', values='norm_robustness_score')
    sns.heatmap(pivot_norm_robust, annot=True, fmt='.3f', cmap='viridis', ax=axes[1,1])
    axes[1,1].set_title('Normalized Robustness Scores')
    
    # 5. Raw minimum scores (worst-case robustness)
    pivot_raw_min = analysis_df.pivot(index='conv_gamma', columns='lin_gamma', values='raw_min')
    sns.heatmap(pivot_raw_min, annot=True, fmt='.3f', cmap='viridis', ax=axes[2,0])
    axes[2,0].set_title('Raw Minimum Scores (Worst-Case)')
    
    # 6. Normalized minimum scores (worst-case robustness)
    pivot_norm_min = analysis_df.pivot(index='conv_gamma', columns='lin_gamma', values='norm_min')
    sns.heatmap(pivot_norm_min, annot=True, fmt='.3f', cmap='viridis', ax=axes[2,1])
    axes[2,1].set_title('Normalized Minimum Scores (Worst-Case)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


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
    print(f"  Robustness Score: {best_params_normalized['robustness_score']:.4f}")
    
    print(f"\nNormalized Score Performance Metrics:")
    print(f"  Mean Score:       {best_params_normalized['mean_score']:.4f}")
    print(f"  Min Score:        {best_params_normalized['min_score']:.4f}")
    print(f"  Std Score:        {best_params_normalized['std_score']:.4f}")
    print(f"  Stability:        {best_params_normalized['stability']:.4f}")
    print(f"  Robustness Ratio: {best_params_normalized['robustness_ratio']:.4f}")
    
    # Compare if they suggest the same hyperparameters
    same_params = (best_params_raw['conv_gamma'] == best_params_normalized['conv_gamma'] and 
                   best_params_raw['lin_gamma'] == best_params_normalized['lin_gamma'])
    
    print(f"\n{'='*25} COMPARISON {'='*25}")
    print(f"Same optimal hyperparameters: {'Yes' if same_params else 'No'}")
    
    if not same_params:
        print(f"Raw optimal:        γ_conv={best_params_raw['conv_gamma']}, γ_lin={best_params_raw['lin_gamma']}")
        print(f"Normalized optimal: γ_conv={best_params_normalized['conv_gamma']}, γ_lin={best_params_normalized['lin_gamma']}")
    
    # Show top 3 combinations for each score type
    print(f"\nTop 3 Raw Score Combinations:")
    top_3_raw = analysis_df.nlargest(3, 'raw_robustness_score')
    for i, (_, row) in enumerate(top_3_raw.iterrows(), 1):
        print(f"  {i}. γ_conv={row['conv_gamma']}, γ_lin={row['lin_gamma']}: "
              f"robust={row['raw_robustness_score']:.4f}, "
              f"mean={row['raw_mean']:.4f}, "
              f"min={row['raw_min']:.4f}")
    
    print(f"\nTop 3 Normalized Score Combinations:")
    top_3_norm = analysis_df.nlargest(3, 'norm_robustness_score')
    for i, (_, row) in enumerate(top_3_norm.iterrows(), 1):
        print(f"  {i}. γ_conv={row['conv_gamma']}, γ_lin={row['lin_gamma']}: "
              f"robust={row['norm_robustness_score']:.4f}, "
              f"mean={row['norm_mean']:.4f}, "
              f"min={row['norm_min']:.4f}")
    
    # Show most stable combinations for each score type
    print(f"\nMost Stable Raw Score Combinations:")
    stable_3_raw = analysis_df.nlargest(3, 'raw_stability')
    for i, (_, row) in enumerate(stable_3_raw.iterrows(), 1):
        print(f"  {i}. γ_conv={row['conv_gamma']}, γ_lin={row['lin_gamma']}: "
              f"stability={row['raw_stability']:.4f}, "
              f"std={row['raw_std']:.4f}")
    
    print(f"\nMost Stable Normalized Score Combinations:")
    stable_3_norm = analysis_df.nlargest(3, 'norm_stability')
    for i, (_, row) in enumerate(stable_3_norm.iterrows(), 1):
        print(f"  {i}. γ_conv={row['conv_gamma']}, γ_lin={row['lin_gamma']}: "
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
    gamma_stats_raw = df.groupby(['conv_gamma', 'lin_gamma'])['faithfulness_score'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    gamma_stats_raw.columns = [f'raw_{col}' if col not in ['conv_gamma', 'lin_gamma'] else col 
                              for col in gamma_stats_raw.columns]
    
    gamma_stats_norm = df.groupby(['conv_gamma', 'lin_gamma'])['normalized_faithfulness_score'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    gamma_stats_norm.columns = [f'norm_{col}' if col not in ['conv_gamma', 'lin_gamma'] else col 
                               for col in gamma_stats_norm.columns]
    
    # Merge gamma statistics
    gamma_stats = pd.merge(gamma_stats_raw, gamma_stats_norm, on=['conv_gamma', 'lin_gamma'])
    
    # Add correlation by gamma combination
    gamma_correlations = df.groupby(['conv_gamma', 'lin_gamma']).apply(
        lambda x: x['faithfulness_score'].corr(x['normalized_faithfulness_score'])
    ).reset_index(name='score_correlation')
    
    gamma_stats = pd.merge(gamma_stats, gamma_correlations, on=['conv_gamma', 'lin_gamma'])
    
    # Find best gamma combinations by different metrics
    best_by_raw_mean = gamma_stats.loc[gamma_stats['raw_mean'].idxmax()]
    best_by_norm_mean = gamma_stats.loc[gamma_stats['norm_mean'].idxmax()]
    best_by_raw_min = gamma_stats.loc[gamma_stats['raw_min'].idxmax()]
    best_by_norm_min = gamma_stats.loc[gamma_stats['norm_min'].idxmax()]
    
    valid_corrs = gamma_stats['score_correlation'].dropna()
    if not valid_corrs.empty:
        best_by_correlation = gamma_stats.loc[valid_corrs.idxmax()]
    else:
        best_by_correlation = None  # oder irgendein Default
        print("No valid correlation scores found – every group had ≤1 sample.")
    
    # Statistics by image (aggregated across gamma combinations)
    image_stats_raw = df.groupby(['input_idx', 'image_label'])['faithfulness_score'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    image_stats_raw.columns = [f'raw_{col}' if col not in ['input_idx', 'image_label'] else col 
                              for col in image_stats_raw.columns]
    
    image_stats_norm = df.groupby(['input_idx', 'image_label'])['normalized_faithfulness_score'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    image_stats_norm.columns = [f'norm_{col}' if col not in ['input_idx', 'image_label'] else col 
                               for col in image_stats_norm.columns]
    
    image_stats = pd.merge(image_stats_raw, image_stats_norm, on=['input_idx', 'image_label'])
    
    return {
        'overall': overall_stats,
        'by_gamma': gamma_stats,
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
    aggregate_stats: Dict,
    config: Dict
):
    """
    Logs all results from a gamma sweep to W&B, including pre-configured plots.
    """
    print("\n--- Logging results to Weights & Biases ---")

    # --- 1. Log Summary Metrics ---
    # This part remains the same.
    wandb.summary["best_raw_conv_gamma"] = best_params_raw['conv_gamma']
    # ... etc for other best params ...
    wandb.summary["best_norm_robustness_score"] = best_params_normalized['robustness_score']
    
    
    # --- 2. Create the Data Tables ---
    # This is also the same as before.
    # in case violations are too much: filtered_df = df.loc[:, ~df.columns.str.startswith("violations.")]
    detailed_results_table = wandb.Table(dataframe=pd.DataFrame(results))
    analysis_table = wandb.Table(dataframe=analysis_df)
    
    curves_df = pd.DataFrame(
        all_curves_data,
        columns=["input_idx", "image_label", "conv_gamma", "lin_gamma", "curve_type", "step", "score"]
    )
    faithfulness_curves_table = wandb.Table(dataframe=curves_df)

    # --- 3. Log the Pre-configured Line Plot for Faithfulness Curves ---
    # THIS IS THE KEY CHANGE.
    # We create a plot object from our table.
    # We will create one plot per image to avoid clutter.
    """num_images = curves_df['input_idx'].nunique()
    for i in range(num_images):
        # Filter the table for the current image
        image_label = curves_df[curves_df['input_idx'] == i]['image_label'].iloc[0]
        image_table = wandb.Table(
            dataframe=curves_df[curves_df['input_idx'] == i]
        )

        # Create a unique title for the plot for this image
        plot_title = f"Faithfulness Curves for Image {i} ({image_label})"
        
        # Log the plot object
        # The key here will be the section name in the W&B dashboard.
        # The value is the plot object itself.
        wandb.log({
            f"plots/faithfulness_curves_img_{i}": wandb.plot.line_series(
                xs=curves_df['step'].unique(), # The x-values (perturbation steps)
                ys=curves_df.pivot_table(index='step', columns=['conv_gamma', 'lin_gamma', 'curve_type'], values='score').values, # The y-values for each series
                keys=curves_df.pivot_table(index='step', columns=['conv_gamma', 'lin_gamma', 'curve_type'], values='score').columns.map(str), # The legend labels for each line
                title=plot_title,
                xname="Perturbation Step"
            )
        })"""

    # The line_series plot is powerful but can be complex if you have many series.
    # An alternative is logging one plot per gamma combination. Let's do that as well, as it's often cleaner.
    for (idx, label, conv_g, lin_g), group in curves_df.groupby(['input_idx', 'image_label', 'conv_gamma', 'lin_gamma']):
        plot_key = f"curves_per_run/img_{idx}_{label}_conv_{conv_g}_lin_{lin_g}"
        table_for_plot = wandb.Table(dataframe=group)
        wandb.log({
            plot_key: wandb.plot.line(
                table_for_plot,
                x="step",
                y="score",
                stroke="curve_type", # This creates separate lines for 'morf' and 'lerf'
                title=f"Curves for {label} (γ_conv={conv_g}, γ_lin={lin_g})"
            )
        })


    # --- 4. Log the Heatmaps and Other Tables ---
    analysis_plot_path = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/robustness_analysis/robustness_analysis.png"
    if os.path.exists(analysis_plot_path):
        wandb.log({"summary_plots/robustness_heatmaps": wandb.Image(analysis_plot_path)})
    
    wandb.log({
        "data_tables/detailed_results": detailed_results_table,
        "data_tables/robustness_analysis_by_gamma": analysis_table,
        "data_tables/faithfulness_curves_raw_data": faithfulness_curves_table
    })

    # --- 5. Log Artifacts ---
    # This part remains the same.
    print("Creating and logging artifact...")
    artifact = wandb.Artifact('robustness-sweep-results', type='analysis-results')
    artifact.add(detailed_results_table, "detailed_results")
    artifact.add(analysis_table, "robustness_analysis_by_gamma")
    artifact.add(faithfulness_curves_table, "faithfulness_curves_raw_data")
    wandb.log_artifact(artifact)

    print("--- Finished logging to W&B ---")