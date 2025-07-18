import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

import zennit.rules as z_rules
from zennit.composites import LayerMapComposite

import datetime
import os
from zennit.image import imgify

from knn_helpers import compute_knn_proxy_score

class LRPConservationChecker:
    """
    A context manager to check for LRP relevance conservation in a PyTorch model.

    This checker always attaches hooks and calculates relevance sums when active.
    The `check()` method returns a dictionary of any violations found.
    A `verbose` flag controls whether `check()` also prints a detailed report
    to the console.

    The performance overhead of the hooks is generally small compared to the
    backward pass itself.

    Args:
        model (nn.Module): The PyTorch model to inspect.
    """
    # TODO: This only works if you disable all bias terms in all linear layers and also 
    # replace the softmax with nn.Identity just for the sake of testing conservation
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.results: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    def _create_hook(self, name: str):
        """Creates a backward hook for a specific module."""
        def hook(module: nn.Module, grad_input: tuple, grad_output: tuple):
            rin, rout = None, None
            if grad_output and grad_output[0] is not None:
                rin = grad_output[0].sum().item()
            if grad_input and grad_input[0] is not None:
                rout = grad_input[0].sum().item()
            elif rin is not None:
                # For the very first layer, grad_input might be None.
                rout = rin
            self.results[name] = (rin, rout)
        return hook

    def __enter__(self):
        """Attach hooks to all leaf modules."""
        self.results.clear()
        self.handles.clear()
        for name, module in self.model.named_modules():
            if not list(module.children()): # Hook leaf modules
                full_name = f"{name} ({module.__class__.__name__})"
                handle = module.register_full_backward_hook(self._create_hook(full_name))
                self.handles.append(handle)
        return self

    def check(self, verbose: bool = True) -> Dict[str, float]:
        """
        Calculates violations and optionally prints a detailed report.

        Args:
            verbose (bool): If True, prints a detailed report of relevance
                            conservation for each layer. Defaults to True.

        Returns:
            Dict[str, float]: A dictionary of violations, where keys are module
                              names and values are the relevance differences.
                              This is returned regardless of the verbose setting.
        """
        violations = {}
        # Sort results by name for consistent output order
        sorted_results = sorted(self.results.items())

        for name, (rin, rout) in sorted_results:
            if rin is None or rout is None:
                continue
            
            if not torch.isclose(torch.tensor(rin), torch.tensor(rout), atol=1e-5):
                diff = rin - rout
                violations[name] = diff
        
        if verbose:
            print("\n--- LRP Conservation Check ---")
            if not sorted_results:
                print("No relevance data was captured.")
            else:
                for name, (rin, rout) in sorted_results:
                    if rin is None or rout is None:
                        continue
                    
                    diff = rin - rout
                    status = "OK" if name not in violations else "VIOLATION"
                    print(
                        f"{status} - Layer: {name:<45} | "
                        f"R_in: {rin:>15.6f}, R_out: {rout:>15.6f}, Diff: {diff:>15.6f}"
                    )
            
            print("-" * 100)
            if not violations:
                print("All checked layers are conservative.")
            else:
                print(f"Found {len(violations)} conservation violation(s).")
            print("-" * 100 + "\n")
            
        return violations

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove all attached hooks."""
        for handle in self.handles:
            handle.remove()
            
def compute_simple_attnlrp_pass(
    conv_gamma: float, 
    lin_gamma: float, 
    model_wrapper: nn.Module, 
    input_tensor: torch.Tensor,
    checker: LRPConservationChecker, 
    verbose: bool = False
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes a single LRP forward/backward pass for a given set of gamma rules.

    ASSUMES that the model is already patched by DINOPatcher, that zennit has been patched and that this
    function is called within an active LRPConservationChecker context.
    """
    input_tensor.grad = None
    
    #TODO maybe integrate this into the dinopatcher when optimal parameters have been found
    zennit_comp = LayerMapComposite(
        [
            (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
            (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
        ]
    )
    
    try:
        zennit_comp.register(model_wrapper)

        output = model_wrapper(input_tensor.requires_grad_())
        most_active_feature_idx = torch.argmax(output, dim=1).item()
        
        if verbose:
            print(f"Explaining feature {most_active_feature_idx} for Gammas (Conv: {conv_gamma}, Lin: {lin_gamma})")

        output[0, most_active_feature_idx].backward()

        relevance = input_tensor * input_tensor.grad
        
        violations = checker.check(verbose=verbose)

    finally:
        zennit_comp.remove()
    
    return relevance, violations

def compute_knn_attnlrp_pass(
    conv_gamma: float, 
    lin_gamma: float, 
    model_wrapper: nn.Module, 
    input_tensor: torch.Tensor,
    checker: LRPConservationChecker,
    # parameters required for the k-NN score
    db_embeddings: torch.Tensor,
    db_filenames: list,
    input_filename: str,
    distance_metric: str = "euclidean",
    k_neighbors: int = 5,
    verbose: bool = False
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes a single LRP pass explaining a k-NN classification decision.

    This function calculates a differentiable proxy score based on the k-NN
    outcome and backpropagates from it to generate the relevance map.
    """
    # Reset gradients for this specific pass
    input_tensor.grad = None

    # Zennit rules MUST be set and removed for each pass
    zennit_comp = LayerMapComposite(
        [
            (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
            (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
        ]
    )
    
    try:
        zennit_comp.register(model_wrapper)

        query_embedding = model_wrapper(input_tensor.requires_grad_())

        knn_score = compute_knn_proxy_score(
            query_embedding=query_embedding,
            query_filename=input_filename,
            db_embeddings=db_embeddings,
            db_filenames=db_filenames,
            distance_metric=distance_metric,
            k=k_neighbors
        )
        
        if verbose:
            print(f"Explaining k-NN proxy score: {knn_score.item():.4f} for Gammas (Conv: {conv_gamma}, Lin: {lin_gamma})")

        knn_score.backward()

        relevance = input_tensor * input_tensor.grad
        
        violations = checker.check(verbose=verbose)

    finally:
        zennit_comp.remove()
    
    return relevance, violations


def visualize_relevances(
    relevances: List[torch.Tensor], 
    mode: str,
    image_name: str,
    output_dir: str = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/heatmaps",
    dim: Tuple[int, int] = (3,5)
) -> torch.Tensor:
    heatmaps = [] 
    for parameters, relevance in relevances.items():
        heatmap = relevance.sum(1)
        denom = abs(heatmap).max()
        
        #TODO label the heatmap with the gammas
        
        heatmap = heatmap / denom

        heatmaps.append(heatmap[0].detach().cpu().numpy())
    
    save_path = f"{output_dir}/{mode}/{image_name}/"
    os.makedirs(save_path, exist_ok=True)
    current_dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    imgify(heatmaps, vmin=-1, vmax=1, grid=dim).save(os.path.join(save_path, f"dinov2_heatmap_{current_dt}.png"))