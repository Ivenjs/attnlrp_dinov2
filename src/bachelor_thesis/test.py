import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

import zennit.rules as z_rules
from zennit.composites import LayerMapComposite

from lxt.efficient import monkey_patch_zennit
from dino_patcher import DINOPatcher
from lrp_helpers import visualize_relevances

from basemodel import load_finetuned_timm_wrapper

from PIL import Image
import itertools

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


if __name__ == "__main__":
    monkey_patch_zennit(verbose=True) 

    CHECKPOINT_PATH = ("/workspaces/bachelor_thesis_code/giantbodybest74ens82.pth")
    IMG_SIZE = 518
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BACKBONE = "vit_giant_patch14_dinov2.lvd142m"
    EMBEDDING_DIM = 256  
    SAVE_HEATMAPS = True  
    PATCH_SIZE = 14  
    model_dtype = torch.float32  

    if DEVICE == "cuda" and torch.cude.is_bf16_supported():
        model_dtype = torch.bfloat16

    CONV_GAMMAS = [0.1, 0.25, 1.0]
    LIN_GAMMAS = [0.0, 0.05, 0.1, 0.25]

    model_wrapper, transforms = load_finetuned_timm_wrapper(
        checkpoint_path=CHECKPOINT_PATH,
        backbone_name=BACKBONE,
        embedding_size=EMBEDDING_DIM,
        image_size=IMG_SIZE,
        device=DEVICE,
        model_dtype=model_dtype,
    )

    # 3. Prepare your input image
    image = Image.open("/workspaces/bachelor_thesis_code/src/bachelor_thesis/image2.png").convert("RGB")
    input_tensor = transforms(image).unsqueeze(0).to(DEVICE)

    all_relevances = {}
    all_violations = {}

    print("Patching model for LRP and Conservation Checking for the duration of the sweep...")

    with DINOPatcher(model_wrapper, attention_mode="cp_lrp"), LRPConservationChecker(model_wrapper) as checker:
        
        param_combinations = list(itertools.product(CONV_GAMMAS, LIN_GAMMAS))
        
        for i, (conv_gamma, lin_gamma) in enumerate(param_combinations):
            print(f"\n--- Running Pass {i+1}/{len(param_combinations)} ---")
            
            relevance, violations = compute_simple_attnlrp_pass(
                conv_gamma=conv_gamma,
                lin_gamma=lin_gamma,
                model_wrapper=model_wrapper,
                input_tensor=input_tensor,
                checker=checker,
                verbose=False  
            )
            
            # Store the results
            key = (conv_gamma, lin_gamma)
            all_relevances[key] = relevance.detach().cpu()
            all_violations[key] = violations

    print("\n--- Gamma Sweep Complete ---")
    print("Model has been restored to its original state.")
    
    visualize_relevances(all_relevances)
    
