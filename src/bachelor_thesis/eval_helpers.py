import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from basemodel import TimmWrapper
from typing import Tuple, List, Dict
from knn_helpers import compute_knn_proxy_score, compute_distances, compute_knn_proxy_soft


PATCH_SIZE = 14  # Size of the patches to average over

def calculate_auc(curve: torch.Tensor) -> float:
    """Calculates the Area Under the Curve using the mean, as in the paper."""
    # The paper defines the area as (1/N) * sum(f_j(x_k)).
    # This is equivalent to the mean of the curve points.
    return torch.mean(curve).item()


def _run_knn_perturbation(
    model: torch.nn.Module, # UN-PATCHED model
    input_tensor: torch.Tensor,
    patch_order: torch.Tensor,
    perturbation_type: str,
    patch_size: int,
    # k-NN specific args
    query_label: str,         
    query_filename: str,      
    db_embeddings: torch.Tensor,
    db_labels: list,
    db_filenames: list,
    distance_metric: str,
    proxy_temp: float,
    patches_per_step: int = 1,
    evaluation_metric: str = "soft_knn_margin",
    baseline_value: str = "black",
) -> torch.Tensor:
    """
    Runs a perturbation experiment, tracking the k-NN proxy score at each step.

    Args:
        ... (other args) ...
        patches_per_step (int): The number of patches to add/remove between each
                                evaluation. Set to 1 for maximum granularity.
                                A larger value speeds up the process.
    """
    if not evaluation_metric == "soft_knn_margin":
        raise ValueError(f"Unsupported evaluation metric: {evaluation_metric}")

    model.eval()

    if baseline_value.lower() == "black":
            baseline_fill = torch.zeros_like(input_tensor)
    elif baseline_value.lower() == "mean":
        mean_color = input_tensor.mean(dim=[2, 3], keepdim=True)
        baseline_fill = mean_color.expand_as(input_tensor)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_value}")

    # Calculate the initial, unperturbed k-NN proxy score
    with torch.no_grad():
        initial_embedding = model(input_tensor)
        initial_score = compute_knn_proxy_soft(
            initial_embedding, query_label, query_filename, db_embeddings,
            db_labels, db_filenames, distance_metric, temp=proxy_temp
        )

    num_patches = len(patch_order)
    h, w = input_tensor.shape[-2:]
    num_patches_w = w // patch_size


    output_scores = [initial_score.item()]

    if perturbation_type == 'deletion':
        perturbed_tensor = input_tensor.clone()
    else: # insertion
        perturbed_tensor = baseline_fill.clone()


    patches_processed_so_far = 0
    
    pbar = tqdm(total=num_patches, desc=f"{perturbation_type.capitalize()} Eval (Granular)")

    while patches_processed_so_far < num_patches:
        start_idx = patches_processed_so_far
        end_idx = min(start_idx + patches_per_step, num_patches)
        patches_to_process = patch_order[start_idx:end_idx]

        for patch_idx in patches_to_process:
            row = (patch_idx // num_patches_w) * patch_size
            col = (patch_idx % num_patches_w) * patch_size

            if perturbation_type == 'deletion':
                perturbed_tensor[..., row:row+patch_size, col:col+patch_size] = \
                    baseline_fill[..., row:row+patch_size, col:col+patch_size]
            else: # insertion
                original_patch = input_tensor[..., row:row+patch_size, col:col+patch_size]
                perturbed_tensor[..., row:row+patch_size, col:col+patch_size] = original_patch

        # After perturbing the chunk, run the model and get the score
        with torch.no_grad():
            current_embedding = model(perturbed_tensor)
            score = compute_knn_proxy_soft(
                current_embedding, query_label, query_filename, db_embeddings,
                db_labels, db_filenames, distance_metric, temp=proxy_temp
            )
            output_scores.append(score.item())

        # Update progress
        num_in_chunk = end_idx - start_idx
        patches_processed_so_far += num_in_chunk
        pbar.update(num_in_chunk)
    
    pbar.close()

    # Convert the list of scores to a tensor for calculations
    return torch.tensor(output_scores, device=input_tensor.device)


def srg_knn(
    relevance_map: torch.Tensor,
    input_tensor: torch.Tensor,
    model: torch.nn.Module, # UN-PATCHED model
    patch_size: int,
    # k-NN specific args
    query_label: str,         
    query_filename: str,      
    db_embeddings: torch.Tensor,
    db_labels: list,
    db_filenames: list,
    distance_metric: str,
    proxy_temp: float,
    patches_per_step: int,
    baseline_value: str = "black",
    plot_curves: bool = False,
    evaluation_metrics: List[str] = ["soft_knn_margin"]
) -> Dict:
    """
    Calculates the ∆A_F (SRG-like) score for a k-NN explanation.
    A higher score is better.
    """
    # relevance should be shape (1,1,H,W)
    patch_relevance = F.avg_pool2d(relevance_map, kernel_size=patch_size, stride=patch_size)
    patch_relevance_flat = patch_relevance.flatten()

    lerf_order = torch.argsort(patch_relevance_flat, descending=False)
    morf_order = torch.argsort(patch_relevance_flat, descending=True)

    perturb_args = {
        "patch_size": patch_size,
        "query_label": query_label,
        "query_filename": query_filename,
        "db_embeddings": db_embeddings,
        "db_labels": db_labels,
        "db_filenames": db_filenames,
        "distance_metric": distance_metric,
        "proxy_temp": proxy_temp,
        "patches_per_step": patches_per_step,
        "baseline_value": baseline_value
    }

    

    results = {}

    for metric_name in evaluation_metrics:
        perturb_args["evaluation_metric"] = metric_name

        morf_curve = _run_knn_perturbation(model, input_tensor, morf_order, 'deletion', **perturb_args)
        lerf_curve = _run_knn_perturbation(model, input_tensor, lerf_order, 'deletion', **perturb_args)

        random_curve = _run_knn_perturbation(
            model, input_tensor, torch.randperm(len(morf_order)), 'deletion', **perturb_args
        )

        auc_morf = calculate_auc(morf_curve)
        auc_lerf = calculate_auc(lerf_curve)
        auc_random = calculate_auc(random_curve)

        faithfulness_score = auc_lerf - auc_morf
        morf_vs_random = auc_random - auc_morf
        lerf_vs_random = auc_lerf - auc_random

        print(f"Area under LeRF curve: {auc_lerf:.4f}")
        print(f"Area under MoRF curve: {auc_morf:.4f}")
        print(f"Area under Random curve: {auc_random:.4f}")
        print(f"-------------------------------------------")
        print(f"Faithfulness Score (A_LeRF - A_MoRF): {faithfulness_score:.4f}")
        print(f"MoRF Improvement vs. Random (A_Rand - A_MoRF): {morf_vs_random:.4f}")

        results[metric_name] = {
            'faithfulness_score': faithfulness_score,
            'morf_vs_random': morf_vs_random,
            'lerf_vs_random': lerf_vs_random,
            'auc_morf': auc_morf,
            'auc_lerf': auc_lerf,
            'auc_random': auc_random,
            'morf_curve': morf_curve.cpu().numpy(),
            'lerf_curve': lerf_curve.cpu().numpy(),
            'random_curve': random_curve.cpu().numpy()
        }
    

    

    if plot_curves:
        num_metrics = len(evaluation_metrics)
        fig, axs = plt.subplots(1, num_metrics, figsize=(9 * num_metrics, 7), sharey=False)
        if num_metrics == 1:
            axs = [axs] 

        for i, metric_name in enumerate(evaluation_metrics):
            if metric_name not in results or results[metric_name]['morf_curve'] is None:
                continue # Skip plotting for this metric if data is invalid
            metric_results = results[metric_name]
            num_eval_steps = len(metric_results['morf_curve'])
            x_axis = np.linspace(0, 100, num_eval_steps)
            
            axs[i].plot(x_axis, metric_results['morf_curve'], label=f"MoRF (AUC={metric_results['auc_morf']:.3f})", color='red')
            axs[i].plot(x_axis, metric_results['lerf_curve'], label=f"LeRF (AUC={metric_results['auc_lerf']:.3f})", color='blue')
            axs[i].plot(x_axis, metric_results['random_curve'], label=f"Random (AUC={metric_results['auc_random']:.3f})", color='gray', linestyle='--')
            axs[i].set_title(f"Perturbation Curves ({metric_name.replace('_', ' ').title()})")
            axs[i].set_xlabel('% of Patches Removed')
            axs[i].set_ylabel(f"{metric_name.replace('_', ' ').title()} Score")
            axs[i].legend()
            axs[i].grid(True, linestyle='--', alpha=0.6)
            if 'recall' in metric_name:
                 axs[i].set_ylim([-0.05, 1.05])

        plt.suptitle(f'Faithfulness Evaluation for {query_filename}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Consider making the save path an argument if needed
        plt.savefig("/workspaces/bachelor_thesis_code/src/bachelor_thesis/curves/multi_metric_curves.png")
        plt.close(fig)

    return results

def attention_inside_mask(relevance: torch.Tensor, mask: np.ndarray) -> Tuple[float, float, float]:
    """
    Args:
        mask (np.ndarray): Boolean or binary array of shape (1 ,H, W).
        relevance (torch.Tensor): Tensor of shape (1, 1, H, W) with relevance scores.

    Returns:
        Tuple[float, float, float]:
            (total_fraction, positive_fraction, negative_fraction)
    """
    target_size = relevance.shape[-2:]  # e.g., (518, 518)
    mask_tensor = torch.from_numpy(mask).to(relevance.device)

    if mask_tensor.shape[-2:] != target_size:
        # F.interpolate needs a 4D input (N, C, H, W), so we add a temporary batch dimension.
        # We must also convert to float for the interpolation operation.
        # 'nearest' mode is crucial for masks to avoid creating non-binary values.
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0).float(),
            size=target_size,
            mode='nearest'
        ).squeeze(0) # Remove the temporary batch dimension.

    # From this point on, mask_tensor is guaranteed to have the same H, W as relevance.

    relevance_squeezed = relevance.squeeze(0).squeeze(0)  # from (1, 1, H, W) -> (H, W)
    boolean_mask = mask_tensor.squeeze(0).bool() # from (1, H, W) -> (H, W)

    relevance_inside = relevance_squeezed[boolean_mask]

    total_abs_relevance = relevance_squeezed.abs().sum()
    inside_abs_relevance = relevance_inside.abs().sum()

    positive_relevance = relevance_squeezed.clamp(min=0)
    negative_relevance = relevance_squeezed.clamp(max=0).abs()

    pos_inside = positive_relevance[boolean_mask].sum()
    pos_total = positive_relevance.sum()

    neg_inside = negative_relevance[boolean_mask].sum()
    neg_total = negative_relevance.sum()

    # Calculate fractions, guarding against division by zero
    total_frac = (inside_abs_relevance / total_abs_relevance).item() if total_abs_relevance > 0 else 0.0
    pos_frac = (pos_inside / pos_total).item() if pos_total > 0 else 0.0
    neg_frac = (neg_inside / neg_total).item() if neg_total > 0 else 0.0

    return total_frac, pos_frac, neg_frac


