import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import torch

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
    
    def __init__(self, model: nn.Module, input_layer_name: Optional[str] = None):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.results: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        # The name of the first layer that processes the input tensor
        self.input_layer_name = input_layer_name

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

    def check(self, target_logit_value: float, verbose: bool = True) -> Dict[str, float]:
        """
        Calculates violations and optionally prints a detailed report.

        Args:
            target_logit_value (float): The scalar value of the output logit
                                    that was backpropagated. The total
                                    relevance should conserve to this value.
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
            print("--- Overall Conservation Summary ---")
            print(f"Target Logit Value:        {target_logit_value:>15.6f}")

            total_input_relevance = None
            input_layer_full_name = "N/A"

            if self.input_layer_name:
                # Find the full name of the input layer from the results
                for name in self.results.keys():
                    if name.startswith(self.input_layer_name):
                        input_layer_full_name = name
                        _, rout = self.results[name]
                        total_input_relevance = rout
                        break
            
            if total_input_relevance is not None:
                print(f"Total Input Relevance:       {total_input_relevance:>15.6f} (from '{input_layer_full_name}')")
                overall_diff = target_logit_value - total_input_relevance
                overall_status = "OK" if torch.isclose(torch.tensor(target_logit_value), torch.tensor(total_input_relevance), atol=1e-4) else "VIOLATION"
                print(f"Overall Difference:          {overall_diff:>15.6f} ({overall_status})")
            else:
                print(f"Could not find input layer '{self.input_layer_name}' in captured results.")
                print("Using alphabetically first layer as a fallback (likely incorrect):")
                if sorted_results:
                    first_layer_name, (_, first_rout) = sorted_results[0]
                    if first_rout is not None:
                         print(f"Fallback Input Relevance:    {first_rout:>15.6f} (from '{first_layer_name}')")

            print("-" * 100 + "\n")
            
        return violations

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove all attached hooks."""
        for handle in self.handles:
            handle.remove()

class BiasManager:
    """
    A context manager to temporarily disable and restore bias terms
    in a PyTorch model. This is crucial for LRP conservation checks.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_biases = {}

    def __enter__(self):
        self.original_biases.clear()
        for name, module in self.model.named_modules():
            # Check for linear and layernorm layers which commonly have biases
            if hasattr(module, 'bias') and module.bias is not None:
                # Store the original bias and its name
                self.original_biases[name] = module.bias
                module.bias = None # Disable it
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore all original biases
        for name, module in self.model.named_modules():
            if name in self.original_biases:
                # We stored the tensor itself, so we can re-assign it
                # We need to wrap it in nn.Parameter to restore its gradient properties
                module.bias = nn.Parameter(self.original_biases[name])
        self.original_biases.clear()