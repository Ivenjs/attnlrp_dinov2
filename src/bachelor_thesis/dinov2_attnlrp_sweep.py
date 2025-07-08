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
            faithfulness_score = srg_knn(
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
                "faithfulness_score": faithfulness_score,
                "violations": violations
            })
    
    return results


def find_robust_hyperparameters(
    results: List[Dict],
    robustness_percentile: float = 0.9,
    min_score_threshold: float = 0.0
) -> Tuple[Dict, pd.DataFrame]:
    """
    Find hyperparameters that are robust across images using Pareto/Robustness approach.
    
    Args:
        results: List of evaluation results from evaluate_multi_image_gamma_sweep
        robustness_percentile: What percentile of cases should be "good" (0.9 = 90%)
        min_score_threshold: Minimum acceptable faithfulness score
    
    Returns:
        best_params: Dictionary with best hyperparameters
        analysis_df: DataFrame with detailed analysis
    """
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Group by gamma parameters
    gamma_groups = df.groupby(['conv_gamma', 'lin_gamma'])
    
    analysis_data = []
    
    for (conv_gamma, lin_gamma), group in gamma_groups:
        scores = group['faithfulness_score'].values
        
        # Robustness metrics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        percentile_score = np.percentile(scores, robustness_percentile * 100)
        
        # Count how many images have "good" scores
        good_scores = np.sum(scores >= min_score_threshold)
        robustness_ratio = good_scores / len(scores)
        
        # Stability metric: lower is better (less variance)
        stability = 1 / (1 + std_score)  # Inverse of std, normalized
        
        # Combined robustness score
        # Prioritize: minimum score, then stability, then mean
        robustness_score = (
            0.4 * min_score +           # Worst case shouldn't be terrible
            0.3 * stability +           # Low variance is good
            0.2 * mean_score +          # Good average performance
            0.1 * robustness_ratio      # High success rate
        )
        
        analysis_data.append({
            'conv_gamma': conv_gamma,
            'lin_gamma': lin_gamma,
            'mean_score': mean_score,
            'std_score': std_score,
            'min_score': min_score,
            'percentile_score': percentile_score,
            'robustness_ratio': robustness_ratio,
            'stability': stability,
            'robustness_score': robustness_score,
            'num_images': len(scores)
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    
    # Find best parameters based on robustness score
    best_idx = analysis_df['robustness_score'].idxmax()
    best_params = {
        'conv_gamma': analysis_df.loc[best_idx, 'conv_gamma'],
        'lin_gamma': analysis_df.loc[best_idx, 'lin_gamma'],
        'robustness_score': analysis_df.loc[best_idx, 'robustness_score'],
        'mean_score': analysis_df.loc[best_idx, 'mean_score'],
        'min_score': analysis_df.loc[best_idx, 'min_score'],
        'std_score': analysis_df.loc[best_idx, 'std_score']
    }
    
    return best_params, analysis_df


def visualize_robustness_analysis(
    analysis_df: pd.DataFrame,
    results: List[Dict],
    save_path: str = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/robustness_analysis.png"
):
    """
    Create visualizations for the robustness analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Heatmap of mean scores
    pivot_mean = analysis_df.pivot(index='conv_gamma', columns='lin_gamma', values='mean_score')
    sns.heatmap(pivot_mean, annot=True, fmt='.3f', cmap='viridis', ax=axes[0,0])
    axes[0,0].set_title('Mean Faithfulness Scores')
    
    # 2. Heatmap of minimum scores (robustness)
    pivot_min = analysis_df.pivot(index='conv_gamma', columns='lin_gamma', values='min_score')
    sns.heatmap(pivot_min, annot=True, fmt='.3f', cmap='viridis', ax=axes[0,1])
    axes[0,1].set_title('Minimum Faithfulness Scores (Robustness)')
    
    # 3. Heatmap of standard deviation (stability)
    pivot_std = analysis_df.pivot(index='conv_gamma', columns='lin_gamma', values='std_score')
    sns.heatmap(pivot_std, annot=True, fmt='.3f', cmap='viridis_r', ax=axes[1,0])  # Reverse colormap
    axes[1,0].set_title('Standard Deviation (Lower = More Stable)')
    
    # 4. Heatmap of robustness score
    pivot_robust = analysis_df.pivot(index='conv_gamma', columns='lin_gamma', values='robustness_score')
    sns.heatmap(pivot_robust, annot=True, fmt='.3f', cmap='viridis', ax=axes[1,1])
    axes[1,1].set_title('Combined Robustness Score')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_robustness_summary(best_params: Dict, analysis_df: pd.DataFrame, results: List[Dict] = None):
    """
    Print a summary of the robustness analysis.
    """
    print("\n" + "="*50)
    print("ROBUSTNESS ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"\nBest Hyperparameters (Robustness Approach):")
    print(f"  Conv Gamma: {best_params['conv_gamma']}")
    print(f"  Lin Gamma:  {best_params['lin_gamma']}")
    print(f"  Robustness Score: {best_params['robustness_score']:.4f}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Mean Score:   {best_params['mean_score']:.4f}")
    print(f"  Min Score:    {best_params['min_score']:.4f}")
    print(f"  Std Score:    {best_params['std_score']:.4f}")
    
    # Show top 3 most robust combinations
    print(f"\nTop 3 Most Robust Combinations:")
    top_3 = analysis_df.nlargest(3, 'robustness_score')
    for i, (_, row) in enumerate(top_3.iterrows(), 1):
        print(f"  {i}. γ_conv={row['conv_gamma']}, γ_lin={row['lin_gamma']}: "
              f"score={row['robustness_score']:.4f}, min={row['min_score']:.4f}")
    
    # Show most stable (lowest std) combinations
    print(f"\nMost Stable Combinations (lowest std):")
    stable_3 = analysis_df.nsmallest(3, 'std_score')
    for i, (_, row) in enumerate(stable_3.iterrows(), 1):
        print(f"  {i}. γ_conv={row['conv_gamma']}, γ_lin={row['lin_gamma']}: "
              f"std={row['std_score']:.4f}, mean={row['mean_score']:.4f}")
    
    # Add aggregate statistics if results are provided
    if results is not None:
        print(f"\n" + "="*50)
        print("AGGREGATE STATISTICS")
        print("="*50)
        
        stats = calculate_aggregate_statistics(results)
        overall = stats['overall']
        
        print(f"\nOverall Performance (all images, all gamma combinations):")
        print(f"  Mean:     {overall['overall_mean']:.4f}")
        print(f"  Median:   {overall['overall_median']:.4f}")
        print(f"  Std Dev:  {overall['overall_std']:.4f}")
        print(f"  Min:      {overall['overall_min']:.4f}")
        print(f"  Max:      {overall['overall_max']:.4f}")
        print(f"  Q25:      {overall['overall_q25']:.4f}")
        print(f"  Q75:      {overall['overall_q75']:.4f}")
        
        print(f"\nBest Gamma Combinations by Different Metrics:")
        print(f"  Best Mean:   γ_conv={stats['best_by_mean']['conv_gamma']}, "
              f"γ_lin={stats['best_by_mean']['lin_gamma']}, "
              f"score={stats['best_by_mean']['mean']:.4f}")
        print(f"  Best Median: γ_conv={stats['best_by_median']['conv_gamma']}, "
              f"γ_lin={stats['best_by_median']['lin_gamma']}, "
              f"score={stats['best_by_median']['median']:.4f}")
        print(f"  Best Min:    γ_conv={stats['best_by_min']['conv_gamma']}, "
              f"γ_lin={stats['best_by_min']['lin_gamma']}, "
              f"score={stats['best_by_min']['min']:.4f}")
        
        return stats