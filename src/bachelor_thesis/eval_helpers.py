import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from basemodel import TimmWrapper
from typing import Tuple, List
from knn_helpers import compute_knn_proxy_score, compute_evaluation_score


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
    k_neighbors: int,
    patches_per_step: int = 1,
    baseline_value: float = 0.0,
) -> torch.Tensor:
    """
    Runs a perturbation experiment, tracking the k-NN proxy score at each step.

    Args:
        ... (other args) ...
        patches_per_step (int): The number of patches to add/remove between each
                                evaluation. Set to 1 for maximum granularity.
                                A larger value speeds up the process.
    """
    model.eval()
    # Calculate the initial, unperturbed k-NN proxy score
    with torch.no_grad():
        initial_embedding = model(input_tensor)
        
        # This function will now be responsible for finding the fixed neighbors
        _, friends_indices, _ = compute_knn_proxy_score(
            initial_embedding, query_label, query_filename, db_embeddings, 
            db_labels, db_filenames, distance_metric, k_neighbors,
            return_indices=True # Add this flag to your score function
        )

        # If there are no friends in the initial k-NN, the evaluation is meaningless.
        if not friends_indices:
            logging.warning(f"No friends found for {query_filename} in its initial k-NN set. "
                            "Perturbation curve will be flat at 0.")
            num_steps = (len(patch_order) // patches_per_step) + 1
            return torch.zeros(num_steps, device=input_tensor.device)

        # Step 2: Calculate the initial score for the curve using the *evaluation* score.
        initial_score = compute_evaluation_score(
            initial_embedding, db_embeddings, friends_indices, distance_metric
        )

    num_patches = len(patch_order)
    h, w = input_tensor.shape[-2:]
    num_patches_w = w // patch_size


    output_scores = [initial_score]

    if perturbation_type == 'deletion':
        perturbed_tensor = input_tensor.clone()
    else: # insertion
        perturbed_tensor = torch.full_like(input_tensor, baseline_value)


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
                perturbed_tensor[..., row:row+patch_size, col:col+patch_size] = baseline_value
            else: # insertion
                original_patch = input_tensor[..., row:row+patch_size, col:col+patch_size]
                perturbed_tensor[..., row:row+patch_size, col:col+patch_size] = original_patch

        # After perturbing the chunk, run the model and get the score
        with torch.no_grad():
            current_embedding = model(perturbed_tensor)
            score = compute_evaluation_score(
                current_embedding, db_embeddings, friends_indices, distance_metric
            )
            output_scores.append(score.item())

        # Update progress
        num_in_chunk = end_idx - start_idx
        patches_processed_so_far += num_in_chunk
        pbar.update(num_in_chunk)
    
    pbar.close()

    # Convert the list of scores to a tensor for calculations
    return torch.tensor(output_scores, device=input_tensor.device)

def run_random_baseline_perturbation(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_patches: int,
    # Pass all the other args needed by _run_knn_perturbation
    **kwargs 
) -> torch.Tensor:
    """
    Runs a perturbation experiment using a random patch order.
    """
    print("Running Random Baseline Perturbation...")
    # Create a random permutation of patch indices
    random_order = torch.randperm(num_patches, device=input_tensor.device)
    
    # We can run a "deletion" experiment with this random order.
    # The result represents the expected performance drop from random occlusions.
    random_curve = _run_knn_perturbation(
        model=model,
        input_tensor=input_tensor,
        patch_order=random_order,
        perturbation_type='deletion',
        **kwargs
    )
    return random_curve

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
    k_neighbors: int,
    patches_per_step: int,
    plot_curves: bool = False,
) -> float:
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
        "k_neighbors": k_neighbors,
        "patches_per_step": patches_per_step,
    }

    morf_curve = _run_knn_perturbation(
        model, input_tensor, morf_order, 'deletion', patch_size, 
        query_label, query_filename, db_embeddings, db_labels, db_filenames, distance_metric, k_neighbors, patches_per_step=patches_per_step
    )
    lerf_curve = _run_knn_perturbation(
        model, input_tensor, lerf_order, 'deletion', patch_size,
        query_label, query_filename, db_embeddings, db_labels, db_filenames, distance_metric, k_neighbors, patches_per_step=patches_per_step
    )

    random_curve = run_random_baseline_perturbation(model, input_tensor, len(lerf_order), **perturb_args)

    auc_morf = calculate_auc(morf_curve)
    auc_lerf = calculate_auc(lerf_curve)
    auc_random = calculate_auc(random_curve)

    delta_a_f = auc_lerf - auc_morf
    
    # A positive score means your attribution is better at finding important patches than chance.
    morf_vs_random = auc_random - auc_morf
    
    # A positive score means attribution is better at finding UNimportant patches than chance.
    lerf_vs_random = auc_lerf - auc_random

    print(f"\n--- Faithfulness Scores ---")
    print(f"Area under LeRF curve: {auc_lerf:.4f}")
    print(f"Area under MoRF curve: {auc_morf:.4f}")
    print(f"Area under Random curve: {auc_random:.4f}")
    print(f"-------------------------------------------")
    print(f"Final Score (A_LeRF - A_MoRF): {delta_a_f:.4f}")
    print(f"MoRF Improvement vs. Random (A_Rand - A_MoRF): {morf_vs_random:.4f} (Higher is better)")
    print(f"LeRF Improvement vs. Random (A_LeRF - A_Rand): {lerf_vs_random:.4f} (Higher is better)")

    if plot_curves:
        num_eval_steps = len(morf_curve)
        x_axis = np.linspace(0, 100, num_eval_steps) # Percentage of patches removed

        plt.figure(figsize=(8, 6))
        plt.plot(x_axis, morf_curve.cpu().numpy(), label=f'MoRF (AUC={auc_morf:.3f})', color='red')
        plt.plot(x_axis, lerf_curve.cpu().numpy(), label=f'LeRF (AUC={auc_lerf:.3f})', color='blue')
        plt.plot(x_axis, random_curve.cpu().numpy(), label=f'Random (AUC={auc_random:.3f})', color='gray', linestyle='--')
        
        plt.title('Perturbation Curves vs. Random Baseline')
        plt.xlabel('% of Patches Removed')
        plt.ylabel('k-NN Similarity Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("/workspaces/bachelor_thesis_code/src/bachelor_thesis/curves/comparison_curves.png")

    return {
        "faithfulness_score": delta_a_f,
        "morf_vs_random": morf_vs_random,
        "lerf_vs_random": lerf_vs_random,
        "auc_morf": auc_morf,
        "auc_lerf": auc_lerf,
        "auc_random": auc_random,
        "morf_curve": morf_curve.cpu().numpy(),
        "lerf_curve": lerf_curve.cpu().numpy(),
        "random_curve": random_curve.cpu().numpy(),
    }

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