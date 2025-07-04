import itertools
import torch
import torch.nn as nn
from typing import List

from lrp_helpers import compute_simple_attnlrp_pass, compute_knn_attnlrp_pass, LRPConservationChecker
from dino_patcher import DINOPatcher

CONV_GAMMAS = [0.1, 0.25, 1.0]
LIN_GAMMAS = [0.0, 0.05, 0.1, 0.25]

def run_gamma_sweep(
    model_wrapper: nn.Module, 
    input_tensor: torch.Tensor,
    conv_gamma_values: List[float] = CONV_GAMMAS,
    lin_gamma_values: List[float] = LIN_GAMMAS
):
    """
    Runs a sweep over gamma parameters, managing patches efficiently.
    """
    all_relevances = {}
    all_violations = {}

    print("--- Starting Gamma Sweep ---")
    print("Patching model for LRP and Conservation Checking for the duration of the sweep...")

    # The context managers are now OUTSIDE the loop.
    # They are set up once and torn down once.
    with DINOPatcher(model_wrapper), LRPConservationChecker(model_wrapper) as checker:
        
        param_combinations = list(itertools.product(conv_gamma_values, lin_gamma_values))
        
        for i, (conv_gamma, lin_gamma) in enumerate(param_combinations):
            print(f"\n--- Running Pass {i+1}/{len(param_combinations)} ---")
            
            # Call the inner-loop function
            relevance, violations = compute_simple_attnlrp_pass(
                conv_gamma=conv_gamma,
                lin_gamma=lin_gamma,
                model_wrapper=model_wrapper,
                input_tensor=input_tensor,
                checker=checker,
                verbose=True  # Control verbosity of the check here
            )
            
            # Store the results
            key = (conv_gamma, lin_gamma)
            all_relevances[key] = relevance.detach().cpu()
            all_violations[key] = violations

    print("\n--- Gamma Sweep Complete ---")
    print("Model has been restored to its original state.")
    
    return all_relevances, all_violations