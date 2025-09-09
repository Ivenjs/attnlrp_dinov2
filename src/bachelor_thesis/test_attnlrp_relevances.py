import torch
import unittest
from checkers import LRPConservationChecker, BiasManager
from dino_patcher import DINOPatcher
from basemodel import get_model_wrapper
from typing import Tuple, Dict
import torch.nn as nn

def compute_conservation_pass(
    model_wrapper: nn.Module, 
    input_tensor: torch.Tensor,
    checker: LRPConservationChecker, 
    verbose: bool = False
) -> Dict[str, float]:
    """
    Computes a single LRP backward pass to check for relevance conservation.
    - `BiasManager` is active.
    - `DINOPatcher` is active.
    - `LRPConservationChecker` is active.
    - NO zennit rules are applied, as we rely on the patches and the inherent
      conservation of bias-free linear layers.
    """

    input_tensor.grad = None
    input_tensor = input_tensor.to(torch.bfloat16)
    
    output = model_wrapper(input_tensor.requires_grad_())
    most_active_feature_idx = torch.argmax(output, dim=1).item()
    
    # Select any feature to explain
    target_feature = output[0, most_active_feature_idx]
    target_value = target_feature.item()
    
    if verbose:
        print(f"Explaining feature with value: {target_value:.4f}")

    target_feature.backward()
    
    # Pass the exact value we backpropagated from to the checker
    violations = checker.check(target_logit_value=target_value, verbose=verbose)
    
    return violations

class TestLRPConservation(unittest.TestCase):

    def test_dinov2_patches_conserve_relevance(self, cfg):
        """
        Verifies that the DINOPatcher implementation correctly conserves relevance.
        This test is run in an idealized environment:
        1. All biases are temporarily disabled.
        2. A single output embedding is backpropagated.
        3. The conservative CP-LRP attention patch is used.
        """
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        VERBOSE = True  
        torch.manual_seed(cfg["seed"])  

        model_wrapper, _, _ = get_model_wrapper(device=DEVICE, cfg=cfg["model"])
        
        # Use a random tensor with the dinov2 giant input size
        dummy_input = torch.randn(1, 3, 518, 518, device=DEVICE)
        INPUT_LAYER_NAME = "model.patch_embed.proj"

        # 2. Create the "pure" testing environment
        with BiasManager(model_wrapper), \
             DINOPatcher(model_wrapper, conservation_test=True), \
             LRPConservationChecker(model_wrapper, input_layer_name=INPUT_LAYER_NAME) as checker:

            violations = compute_conservation_pass(
                model_wrapper=model_wrapper,
                input_tensor=dummy_input,
                checker=checker,
                verbose=True
            )

            # The final assertion: the test passes if there are no violations.
            self.assertEqual(len(violations), 0, "Found relevance conservation violations!")
            print("--- LRP Conservation Test Passed ---")


if __name__ == "__main__":
    unittest.main()