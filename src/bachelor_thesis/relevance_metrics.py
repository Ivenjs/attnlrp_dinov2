import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, entropy

def _to_numpy_flat(tensor: torch.Tensor):
    """Helper to convert a tensor to a flattened numpy array."""
    if not isinstance(tensor, torch.Tensor):
        return np.array([])
    return tensor.detach().cpu().numpy().flatten()

def sparsity_metric(relevance: torch.Tensor, threshold=1e-6) -> float:
    """Calculates the proportion of non-zero elements."""
    if torch.all(relevance == 0):
        return 0.0
    rel = relevance.abs().flatten()
    active = (rel > threshold).float().sum()
    return (active / rel.numel()).item()

def topk_concentration(relevance: torch.Tensor, k_fraction=0.05) -> float:
    """Calculates the proportion of total relevance mass in the top k% of patches."""
    if torch.all(relevance == 0):
        return 0.0
    rel = relevance.abs().flatten()
    total_mass = rel.sum()
    if total_mass == 0:
        return 0.0
    
    k = max(1, int(rel.numel() * k_fraction))
    topk_mass = torch.topk(rel, k).values.sum()
    
    return (topk_mass / total_mass).item()

def relevance_entropy(relevance: torch.Tensor) -> float:
    """Calculates the entropy of the normalized relevance distribution."""
    if torch.all(relevance == 0):
        return np.nan
    rel = relevance.abs().flatten()
    total = rel.sum()
    if total == 0:
        return np.nan
    p = rel / total
    # Add epsilon for numerical stability in log
    return -(p * torch.log(p + 1e-9)).sum().item()

def gini_coefficient(relevance: torch.Tensor) -> float:
    """Calculates the Gini coefficient, a measure of inequality/concentration."""
    if torch.all(relevance == 0):
        return 0.0 # Perfect equality
    rel_np = np.sort(_to_numpy_flat(relevance.abs()))
    n = len(rel_np)
    if n == 0 or rel_np[-1] == 0:
        return 0.0
    
    cum_rel = np.cumsum(rel_np)
    # The Gini coefficient is the area between the Lorenz curve and the line of equality
    gini = (n + 1 - 2 * (np.sum(cum_rel) / cum_rel[-1])) / n
    return gini

def auroc_ap(relevance: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """Calculates AUROC and Average Precision."""
    y_true = _to_numpy_flat(mask).astype(int)
    scores = _to_numpy_flat(relevance)

    # If mask is all ones or all zeros, these metrics are undefined.
    if np.unique(y_true).size == 1:
        return np.nan, np.nan
    
    try:
        auroc = roc_auc_score(y_true, scores)
        ap = average_precision_score(y_true, scores)
    except ValueError: # Can happen if scores are constant
        return np.nan, np.nan
        
    return auroc, ap

def mean_inside_outside(relevance: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """Calculates mean relevance inside and outside the mask."""
    mask_bool = mask.bool()
    
    inside_relevance = relevance[mask_bool]
    outside_relevance = relevance[~mask_bool]
    
    mean_in = inside_relevance.mean().item() if inside_relevance.numel() > 0 else np.nan
    mean_out = outside_relevance.mean().item() if outside_relevance.numel() > 0 else np.nan
    
    return mean_in, mean_out

def compute_all_relevance_metrics(relevance: torch.Tensor, mask: torch.Tensor) -> dict:
    """
    Computes all defined metrics for a given relevance map and mask.
    Handles None mask and all-zero relevance maps gracefully.
    """
    if mask is None:
        # If no mask, we can only compute mask-agnostic metrics
        return {
            'sparsity': sparsity_metric(relevance),
            'topk_concentration_5pct': topk_concentration(relevance, k_fraction=0.05),
            'entropy': relevance_entropy(relevance),
            'gini': gini_coefficient(relevance),
            'auroc': np.nan,
            'ap': np.nan,
            'mean_in': np.nan,
            'mean_out': np.nan,
        }
        
    # Mask-dependent metrics
    auroc, ap = auroc_ap(relevance, mask)
    mean_in, mean_out = mean_inside_outside(relevance, mask)
    
    # AoGR is simply the fraction of total relevance inside the mask
    relevance_abs = relevance.abs()
    total_relevance_mass = relevance_abs.sum()
    if total_relevance_mass > 0:
        aogr = (relevance_abs[mask.bool()].sum() / total_relevance_mass).item()
    else:
        aogr = np.nan # Undefined if total relevance is zero

    return {
        'sparsity': sparsity_metric(relevance),
        'topk_concentration_5pct': topk_concentration(relevance, k_fraction=0.05),
        'entropy': relevance_entropy(relevance),
        'gini': gini_coefficient(relevance),
        'auroc': auroc,
        'ap': ap,
        'mean_in': mean_in,
        'mean_out': mean_out,
        'aogr': aogr, # Your original AoGR metric is here!
        'background_attention': 1.0 - aogr if not np.isnan(aogr) else np.nan
    }