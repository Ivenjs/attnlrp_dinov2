import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from basemodel import TimmWrapper
from knn_helpers import compute_knn_proxy_score


PATCH_SIZE = 14  # Size of the patches to average over

# Assume these are defined elsewhere
# from my_project import TimmWrapper, LRPPatcher

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
    db_embeddings: torch.Tensor,
    db_filenames: list,
    input_filename: str,
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
        initial_score = compute_knn_proxy_score(
            initial_embedding, input_filename, db_embeddings, db_filenames, distance_metric, k_neighbors
        )

    num_patches = len(patch_order)
    h, w = input_tensor.shape[-2:]
    num_patches_w = w // patch_size

    # Use a list to store scores, as the length depends on patches_per_step.
    # It's more flexible than a pre-allocated tensor.
    output_scores = [initial_score]

    if perturbation_type == 'deletion':
        perturbed_tensor = input_tensor.clone()
    else: # insertion
        perturbed_tensor = torch.full_like(input_tensor, baseline_value)

    # We iterate through the patches in steps of `patches_per_step`.
    # This loop is much more direct than the previous one.
    patches_processed_so_far = 0
    
    # The progress bar now tracks the number of patches processed.
    pbar = tqdm(total=num_patches, desc=f"{perturbation_type.capitalize()} Eval (Granular)")

    while patches_processed_so_far < num_patches:
        # Determine the next chunk of patches to process
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
            score = compute_knn_proxy_score(
                current_embedding, input_filename, db_embeddings, db_filenames, distance_metric, k_neighbors
            )
            output_scores.append(score)

        # Update progress
        num_in_chunk = end_idx - start_idx
        patches_processed_so_far += num_in_chunk
        pbar.update(num_in_chunk)
    
    pbar.close()

    # Convert the list of scores to a tensor for calculations
    return torch.tensor(output_scores, device=input_tensor.device)

def normalize_curve(curve: torch.Tensor) -> torch.Tensor:
    """Normalize the curve by its starting value to ensure comparability."""
    start_val = curve[0].item()
    if abs(start_val) < 1e-6:  # avoid div-by-zero
        return curve
    return curve / start_val

def srg_knn(
    relevance_map: torch.Tensor,
    input_tensor: torch.Tensor,
    model: torch.nn.Module, # UN-PATCHED model
    patch_size: int,
    # k-NN specific args
    db_embeddings: torch.Tensor,
    db_filenames: list,
    input_filename: str,
    distance_metric: str,
    k_neighbors: int,
    patches_per_step: int,
    plot_curves: bool = False,
) -> float:
    """
    Calculates the ∆A_F (SRG-like) score for a k-NN explanation.
    A higher score is better.
    """
    # Ensure the relevance map is a 4D tensor [N, C, H, W] for consistency.
    # The new batched pipeline stores it as 3D [C, H, W].
    if relevance_map.dim() == 3:
        relevance_map = relevance_map.unsqueeze(0)  # Add a batch dimension -> [1, C, H, W]
        
    relevance_per_pixel = relevance_map.sum(dim=1, keepdim=True) # removed abs here because doesnt really make sense for parameter sweep that control negative vs positive relevance
    patch_relevance = F.avg_pool2d(relevance_per_pixel, kernel_size=patch_size, stride=patch_size)
    patch_relevance_flat = patch_relevance.flatten()

    lerf_order = torch.argsort(patch_relevance_flat, descending=False)
    morf_order = torch.argsort(patch_relevance_flat, descending=True)

    morf_curve = _run_knn_perturbation(
        model, input_tensor, morf_order, 'deletion', patch_size, 
        db_embeddings, db_filenames, input_filename, distance_metric, k_neighbors, patches_per_step=patches_per_step
    )
    lerf_curve = _run_knn_perturbation(
        model, input_tensor, lerf_order, 'deletion', patch_size,
        db_embeddings, db_filenames, input_filename, distance_metric, k_neighbors, patches_per_step=patches_per_step    
    )

    morf_curve_norm = normalize_curve(morf_curve)
    lerf_curve_norm = normalize_curve(lerf_curve)

    auc_morf_norm = calculate_auc(morf_curve_norm)
    auc_lerf_norm = calculate_auc(lerf_curve_norm)
    
    delta_a_f_norm = auc_lerf_norm - auc_morf_norm

    auc_morf = calculate_auc(morf_curve)
    auc_lerf = calculate_auc(lerf_curve)
    
    delta_a_f = auc_lerf - auc_morf
    
    print(f"\n--- Faithfulness Score (∆A_F) ---")
    print(f"Area under LeRF curve: {auc_lerf:.4f}")
    print(f"Area under MoRF curve: {auc_morf:.4f}")
    print(f"Final Score (A_LeRF - A_MoRF): {delta_a_f:.4f}")

    print(f"Normalized Final Score (A_LeRF - A_MoRF): {delta_a_f_norm:.4f}")

    if plot_curves:
        num_eval_steps = len(morf_curve)
        x_axis = range(num_eval_steps)

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].plot(x_axis, morf_curve.cpu().numpy(), label='MoRF Deletion', color='red')
        axs[0, 0].set_title(f'MoRF Curve (Raw) – AUC={auc_morf:.3f}')
        axs[0, 0].set_xlabel(f'Evaluation Step (Stepsize: {patches_per_step} patches)')
        axs[0, 0].set_ylabel('Model Score')
        axs[0, 0].legend()

        axs[0, 1].plot(x_axis, lerf_curve.cpu().numpy(), label='LeRF Deletion', color='blue')
        axs[0, 1].set_title(f'LeRF Curve (Raw) – AUC={auc_lerf:.3f}')
        axs[0, 1].set_xlabel(f'Evaluation Step (Stepsize: {patches_per_step} patches)')
        axs[0, 1].set_ylabel('Model Score')
        axs[0, 1].legend()

        axs[1, 0].plot(x_axis, morf_curve_norm.cpu().numpy(), label='MoRF Deletion (Norm.)', color='orange')
        axs[1, 0].set_title(f'MoRF Curve (Noramlized) – AUC={auc_morf_norm:.3f}')
        axs[1, 0].set_xlabel(f'Evaluation Step (Stepsize: {patches_per_step} patches)')
        axs[1, 0].set_ylabel('Normalized Score')
        axs[1, 0].legend()

        axs[1, 1].plot(x_axis, lerf_curve_norm.cpu().numpy(), label='LeRF Deletion (Norm.)', color='green')
        axs[1, 1].set_title(f'LeRF Curve (Noramlized) – AUC={auc_lerf_norm:.3f}')
        axs[1, 1].set_xlabel(f'Evaluation Step (Stepsize: {patches_per_step} patches)')
        axs[1, 1].set_ylabel('Normalized Score')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.savefig("/workspaces/bachelor_thesis_code/src/bachelor_thesis/curves/deletion_metric_curves_split.png")


    return {
        "faithfulness_score": delta_a_f,
        "normalized_faithfulness_score": delta_a_f_norm,
        "morf_curve": morf_curve.cpu().numpy(),
        "lerf_curve": lerf_curve.cpu().numpy(),
        "morf_curve_norm": morf_curve_norm.cpu().numpy(),
        "lerf_curve_norm": lerf_curve_norm.cpu().numpy(),
        "auc_morf": auc_morf,
        "auc_lerf": auc_lerf,
    }

def _run_perturbation(
    model: TimmWrapper, # Expects the UN-PATCHED model
    input_tensor: torch.Tensor,
    patch_order: torch.Tensor,
    perturbation_type: str, # Either 'deletion' (MoRF/Flipping) or 'insertion' (LeRF with reversed logic)
    target_class_idx: int,
    patch_size: int,
    baseline_value: float = 0.0
) -> torch.Tensor:
    """
    Runs a single perturbation experiment (e.g., MoRF Deletion).
    
    Returns a curve of the model's output score at each perturbation step.
    """
    # Ensure model is in eval mode and we don't compute gradients here
    model.eval()
    
    # Get the original, unperturbed score as the first point of the curve
    with torch.no_grad():
        initial_score = model(input_tensor)[0, target_class_idx]

    num_patches = len(patch_order)
    h, w = input_tensor.shape[-2:]
    #num_patches_h = h // patch_size
    num_patches_w = w // patch_size

    # The curve will have N+1 points: 0% perturbed to 100% perturbed
    output_curve = torch.zeros(num_patches + 1, device=input_tensor.device)
    output_curve[0] = initial_score

    # Create the tensor to be perturbed.
    # For deletion, start with the original image and remove patches.
    # For insertion, start with a baseline and add patches.
    if perturbation_type == 'deletion':
        perturbed_tensor = input_tensor.clone()
    elif perturbation_type == 'insertion':
        perturbed_tensor = torch.full_like(input_tensor, baseline_value)
    else:
        raise ValueError("perturbation_type must be 'deletion' or 'insertion'")

    for i in tqdm(range(num_patches), desc=f"{perturbation_type.capitalize()}"):
        # Get the index of the patch to flip
        patch_idx_to_flip = patch_order[i]
        
        # Convert the 1D flat patch index to 2D grid coordinates
        row = (patch_idx_to_flip // num_patches_w) * patch_size
        col = (patch_idx_to_flip % num_patches_w) * patch_size
        
        # Apply the perturbation
        if perturbation_type == 'deletion':
            # Flip the patch to the baseline value
            perturbed_tensor[..., row:row+patch_size, col:col+patch_size] = baseline_value
        elif perturbation_type == 'insertion':
            # Copy the original patch from the input tensor
            original_patch = input_tensor[..., row:row+patch_size, col:col+patch_size]
            perturbed_tensor[..., row:row+patch_size, col:col+patch_size] = original_patch

        # Get the model's output for the perturbed input
        with torch.no_grad():
            score = model(perturbed_tensor)[0, target_class_idx]
            output_curve[i + 1] = score
            
    return output_curve


def srg(
    relevance_map: torch.Tensor,
    input_tensor: torch.Tensor,
    model: torch.nn.Module, # Expects the UN-PATCHED model
    target_class_idx: int, # should be the ground truth class index
    patch_size: int
) -> float:
    """
    Calculates faithfulness metrics based on the paper (MoRF/LeRF Deletion/Flipping).

    Args:
        relevance_map: The (B, C, H, W) relevance map from LRP.
        input_tensor: The original (B, C, H, W) input tensor.
        model: The original, UN-PATCHED model to evaluate against.
        target_class_idx: The index of the class/logit that was explained.
        patch_size: The size of the patches (e.g., 14 or 16 for ViTs).

    Returns:
        The Delta A_F score, where a higher value is better.
    """
    # 1. Aggregate relevance per patch
    # Taking abs() is common to treat both positive and negative relevance as important.
    relevance_per_pixel = relevance_map.sum(dim=1, keepdim=True) # removed abs here because doesnt really make sense for parameter sweep that control negative vs positive relevance
    patch_relevance = F.avg_pool2d(relevance_per_pixel, kernel_size=patch_size, stride=patch_size)
    patch_relevance_flat = patch_relevance.flatten()

    # 2. Get the sorting order for patches
    # Ascending order for LeRF (Least Relevant First)
    lerf_order = torch.argsort(patch_relevance_flat, descending=False)
    # Descending order for MoRF (Most Relevant First)
    morf_order = torch.argsort(patch_relevance_flat, descending=True)

    # 3. Run perturbation experiments
    # The paper uses "Flipping" (Deletion) for both MoRF and LeRF.
    print("--- Calculating MoRF (Most Relevant First) Deletion Curve ---")
    morf_curve = _run_perturbation(model, input_tensor, morf_order, 'deletion', target_class_idx, patch_size)

    print("\n--- Calculating LeRF (Least Relevant First) Deletion Curve ---")
    lerf_curve = _run_perturbation(model, input_tensor, lerf_order, 'deletion', target_class_idx, patch_size)
    
    # 4. Calculate the Area Under the Curve (AUC) for each
    auc_morf = calculate_auc(morf_curve)
    auc_lerf = calculate_auc(lerf_curve)
    
    # 5. Compute the final metric ∆A_F
    # "a higher score signifies a more faithful explainer"
    # A_F_LeRF should be high (score degrades slowly)
    # A_F_MoRF should be low (score degrades quickly)
    delta_a_f = auc_lerf - auc_morf

    print("\n--- Faithfulness Scores ---")
    print(f"Area under LeRF curve: {auc_lerf:.4f}")
    print(f"Area under MoRF curve: {auc_morf:.4f}")
    print(f"Final Score (∆A_F = A_LeRF - A_MoRF): {delta_a_f:.4f}")

    # You can also plot the curves for visual inspection
    # import matplotlib.pyplot as plt
    # plt.plot(morf_curve.cpu().numpy(), label='MoRF Deletion')
    # plt.plot(lerf_curve.cpu().numpy(), label='LeRF Deletion')
    # plt.legend()
    # plt.xlabel('Number of Patches Perturbed')
    # plt.ylabel('Model Score')
    # plt.title('Deletion Metric Curves')
    # plt.show()

    return delta_a_f





def attention_inside_mask(mask: np.ndarray, relevance: torch.Tensor) -> float:
    """
    Args:
        mask (np.ndarray): Boolean or binary array of shape (H, W).
        relevance (torch.Tensor): Tensor of shape (H, W) or (1, H, W) with relevance scores.

    Returns:
        Tuple[float, float, float]:
            (total_fraction, positive_fraction, negative_fraction)
    """
    if relevance.dim() == 4:
        relevance = relevance.squeeze(0)

    # Sanity check
    assert relevance.shape[1:] == mask.shape, "Mask and relevance must have the same shape"

    mask_tensor = torch.from_numpy(mask).to(relevance.device).bool()
    mask_tensor = mask_tensor.unsqueeze(0).expand(relevance.shape)  # expand to match(3, H, W)
    relevance_inside = relevance[mask_tensor]
    # relevance_outside = relevance[~mask_tensor]

    # Compute total relevance values
    total_abs_relevance = relevance.abs().sum()
    inside_abs_relevance = relevance_inside.abs().sum()

    # Compute positive and negative separately
    positive_relevance = relevance.clamp(min=0)
    negative_relevance = relevance.clamp(max=0).abs()

    pos_inside = positive_relevance[mask_tensor].sum()
    pos_total = positive_relevance.sum()

    neg_inside = negative_relevance[mask_tensor].sum()
    neg_total = negative_relevance.sum()

    total_frac = (inside_abs_relevance / total_abs_relevance).item() if total_abs_relevance > 0 else 0.0
    pos_frac = (pos_inside / pos_total).item() if pos_total > 0 else 0.0
    neg_frac = (neg_inside / neg_total).item() if neg_total > 0 else 0.0

    return total_frac, pos_frac, neg_frac
