import torch.nn.functional as F
import torch
from basemodel import TimmWrapper
import numpy as np

PATCH_SIZE = 14  # Size of the patches to average over

def srg(relevance: torch.Tensor, input_tensor: torch.Tensor, model: TimmWrapper) -> float:
    relevance_per_pixel = torch.abs(relevance).sum(dim=1, keepdim=True)
    
    patch_relevance = F.avg_pool2d(
        relevance_per_pixel,
        kernel_size=PATCH_SIZE,
        stride=PATCH_SIZE
    )
    
    patch_relevance_flat = patch_relevance.flatten()
    
    # Sort the patch relevances to get the indices for flipping
    # `torch.argsort` gives you the indices that would sort the tensor.
    # By default, it's in ascending order (Least Informative First).
    sorted_indices = torch.argsort(patch_relevance_flat)

    lif_order = sorted_indices

    # For MIF, reverse the order.
    mif_order = torch.flip(sorted_indices, dims=[0])
    
    lif_curve = _perturbation_exp(model, input_tensor, lif_order)
    mif_curve = _perturbation_exp(model, input_tensor, mif_order)
    
    initial_score = lif_curve[0]
    lif_curve_norm = lif_curve / initial_score
    mif_curve_norm = mif_curve / initial_score

    # Calculate the difference at each perturbation step
    # We ignore the first point (0% perturbation) as the difference is always zero.
    difference_curve = lif_curve_norm[1:] - mif_curve_norm[1:]

    # The SRG is the sum (or mean) of these differences. The mean is better
    # as it makes the metric independent of the number of steps.
    srg_score = torch.mean(difference_curve).item()

    print(f"Symmetric Relevance Gain (SRG): {srg_score:.4f}")
    return srg_score
    
def _perturbation_exp(model: TimmWrapper, input_tensor: torch.Tensor, flipping_order: torch.Tensor) -> torch.Tensor:
    """
    Runs the pixel-flipping experiment for a given order.

    Returns:
        A list of model outputs (e.g., the max embedding value) at each step.
    """
    num_patches = len(flipping_order)
    # Define how many steps to take. You don't need to flip one-by-one.
    # Flipping in chunks (e.g., 5% of patches at a time) is much faster.
    num_steps = 20
    step_size = num_patches // num_steps

    perturbed_tensor = input_tensor.clone()
    
    # Baseline: random noise
    baseline = torch.rand_like(input_tensor)

    # Store the model's confidence at each step
    outputs = []
    
    # Get the initial model output on the original image
    with torch.no_grad():
        initial_output = model(input_tensor)
        # We need a single scalar to represent "confidence".
        # Let's use the max value of the embedding, like in our LRP backward pass.
        # TODO: also use knn classifier to get the confidence
        initial_confidence = torch.max(initial_output).item()
    outputs.append(initial_confidence)

    # Loop through the steps
    for i in range(num_steps):
        # Get the indices of the patches to flip in this step
        start_idx = i * step_size
        end_idx = (i + 1) * step_size
        patches_to_flip = flipping_order[start_idx:end_idx]

        # "Flip" the patches by replacing them with the baseline
        for patch_idx in patches_to_flip:
            # Convert the 1D patch index to 2D grid coordinates
            row = (patch_idx // (input_tensor.shape[3] // PATCH_SIZE)).item()
            col = (patch_idx % (input_tensor.shape[3] // PATCH_SIZE)).item()

            # Get the pixel coordinates
            r_start, c_start = row * PATCH_SIZE, col * PATCH_SIZE
            r_end, c_end = r_start + PATCH_SIZE, c_start + PATCH_SIZE

            # Replace the patch area in the perturbed tensor with the baseline
            perturbed_tensor[:, :, r_start:r_end, c_start:c_end] = baseline[:, :, r_start:r_end, c_start:c_end]
            
        # Get the new model output on the perturbed image
        with torch.no_grad():
            current_output = model(perturbed_tensor)
            current_confidence = torch.max(current_output).item()
        outputs.append(current_confidence)
        
    return torch.tensor(outputs)

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