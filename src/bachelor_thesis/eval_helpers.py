import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from basemodel import TimmWrapper
from typing import Tuple, List, Dict
from knn_helpers import compute_knn_proxy_score, compute_distances


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
    evaluation_metric: str = "similarity" # "similarity" or "recall_at_k"
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

        if evaluation_metric == 'similarity':
            initial_score = compute_evaluation_score(
                initial_embedding, db_embeddings, friends_indices, distance_metric
            )
        elif evaluation_metric == 'recall_at_k':
            # The initial recall should be 1.0 by definition, because this is about the initial k from the query
            initial_score = torch.tensor(1.0)
        else:
            raise ValueError(f"Unknown evaluation_metric: {evaluation_metric}")

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
            if evaluation_metric == 'similarity':
                score = compute_evaluation_score(
                    current_embedding, db_embeddings, friends_indices, distance_metric
                )
                output_scores.append(score.item())
            elif evaluation_metric == 'recall_at_k':
                score = compute_evaluation_score_recall_at_k(
                    current_embedding=current_embedding,
                    db_embeddings=db_embeddings,
                    original_friends_indices=set(friends_indices),
                    k=k_neighbors,
                    distance_metric=distance_metric,
                    query_filename=query_filename,
                    db_filenames=db_filenames,
                )
                output_scores.append(score)
            else:
                raise ValueError(f"Unknown evaluation_metric: {evaluation_metric}")

        # Update progress
        num_in_chunk = end_idx - start_idx
        patches_processed_so_far += num_in_chunk
        pbar.update(num_in_chunk)
    
    pbar.close()

    # Convert the list of scores to a tensor for calculations
    return torch.tensor(output_scores, device=input_tensor.device)

def _run_random_baseline_perturbation(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_patches: int,
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
    evaluation_metrics = ["similarity", "recall_at_k"]
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
        "k_neighbors": k_neighbors,
        "patches_per_step": patches_per_step,
    }

    results = {}

    for metric_name in evaluation_metrics:
        perturb_args["evaluation_metric"] = metric_name

        morf_curve = _run_knn_perturbation(model, input_tensor, morf_order, 'deletion', **perturb_args)
        lerf_curve = _run_knn_perturbation(model, input_tensor, lerf_order, 'deletion', **perturb_args)
        random_curve = _run_random_baseline_perturbation(model, input_tensor, len(morf_order), **perturb_args)

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


def compute_evaluation_score(
    current_embedding: torch.Tensor,
    db_embeddings: torch.Tensor,
    friends_indices: list,
    distance_metric: str,
) -> torch.Tensor:
    """
    Computes an intuitive, similarity-like score for evaluation purposes.
    This score is designed to be high when the query is close to its friends.
    The idea is the same as for the compute_proxy_score function
    - For cosine distance, it converts the [0, 2] distance back to a [-1, 1] similarity.
    - For Euclidean distance, it maps the [0, 2] distance to a [1, 0] similarity.

    Args:
        current_embedding (torch.Tensor): The embedding of the (perturbed) query image.
        db_embeddings (torch.Tensor): The database embeddings.
        friends_indices (list): The list of indices for the *fixed* set of friends.
        distance_metric (str): The metric used, 'cosine' or 'euclidean'.

    Returns:
        torch.Tensor: A single scalar score. Larger is better.
    """
    if not friends_indices:
        # If there are no friends, the score is undefined. Return 0.
        return torch.tensor(0.0, device=current_embedding.device)

    all_distances = compute_distances(current_embedding, db_embeddings, distance_metric)
    dist_to_friends = all_distances[friends_indices]

    if distance_metric == "cosine":
        # Cosine distance = 1 - similarity.
        # So, similarity = 1 - distance.
        # This will be in the range [-1, 1].
        similarity_score = 1 - dist_to_friends.mean()
    elif distance_metric == "euclidean":
        # Euclidean distance on the unit hypersphere is in [0, 2].
        # We can map this to a [1, 0] similarity range to make it intuitive.
        # score = 1 - (dist / 2).
        # When dist=0, score=1. When dist=2, score=0.
        similarity_score = 1 - (dist_to_friends.mean() / 2.0)
    else:
        raise ValueError(f"Unknown metric for evaluation: {distance_metric}")

    return similarity_score

def compute_evaluation_score_recall_at_k(
    current_embedding: torch.Tensor,
    db_embeddings: torch.Tensor,
    original_friends_indices: set,
    k: int,
    distance_metric: str,
    query_filename: str, 
    db_filenames: list,
) -> float:
    """
    Computes a Recall@k-based evaluation score.
    This score measures what fraction of the original friends are still in the top-k.
    """

    num_db_samples = len(db_filenames)
    
    is_query_in_db = query_filename in db_filenames
    
    num_available_neighbors = num_db_samples - 1 if is_query_in_db else num_db_samples
    
    # Ensure k is not larger than the number of available neighbors.
    # If there are no available neighbors, effective_k will be 0.
    effective_k = min(k, num_available_neighbors)

    # If there are no possible neighbors to find, we can't compute a score.
    # Return a neutral score (0) or handle as an error. NORMALLY, THIS SHOULD NOT HAPPEN.
    if effective_k <= 0:
        # Returning a neutral score is often a safe default.
        logging.warning(f"No available neighbors for query '{query_filename}'. Returning neutral score. THIS IS VERY UNUSUAL!")
        return torch.tensor(0.0, device=query_embedding.device, requires_grad=True)

    num_original_friends = len(original_friends_indices)
    if num_original_friends == 0:
        return 0.0

    with torch.no_grad():
        distances = compute_distances(current_embedding, db_embeddings, distance_metric)
        
        try:
            query_idx = db_filenames.index(query_filename)
            distances[query_idx] = torch.inf
        except ValueError:
            pass 

        # Find the new top-k
        new_top_k_indices = torch.topk(distances, effective_k, largest=False).indices
        new_top_k_set = set(new_top_k_indices.cpu().numpy())

        # Count how many of the original friends are in the new top-k set
        retained_friends_count = len(original_friends_indices.intersection(new_top_k_set))
        
        # Calculate recall
        recall = retained_friends_count / num_original_friends
        
    return recall