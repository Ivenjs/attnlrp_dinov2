import itertools
import torch
import torch.nn as nn
from typing import List
import yaml
import os
from PIL import Image

from lrp_helpers import compute_simple_attnlrp_pass, compute_knn_attnlrp_pass, LRPConservationChecker
from basemodel import TimmWrapper

from dino_patcher import DINOPatcher

CONV_GAMMAS = [0.1, 0.25, 1.0]
LIN_GAMMAS = [0.0, 0.05, 0.1, 0.25]

def run_gamma_sweep(
    model_wrapper: TimmWrapper, 
    input_tensor: torch.Tensor,
    mode: str = "simple",  # "simple" or "knn"
    db_embeddings: torch.Tensor = None,  
    db_labels: List[str] = None,  
    ground_truth_label: str  = None,
    k_neighbors: int = 5,
    conv_gamma_values: List[float] = CONV_GAMMAS,
    lin_gamma_values: List[float] = LIN_GAMMAS,
    verbose: bool = False
):
    """
    Runs a sweep over gamma parameters, managing patches efficiently.
    """
    all_relevances = {}
    all_violations = {}

    print("--- Starting Gamma Sweep ---")
    print("Patching model for LRP and Conservation Checking for the duration of the sweep...")


    with DINOPatcher(model_wrapper), LRPConservationChecker(model_wrapper) as checker:
        
        param_combinations = list(itertools.product(conv_gamma_values, lin_gamma_values))
        
        for i, (conv_gamma, lin_gamma) in enumerate(param_combinations):
            print(f"\n--- Running Pass {i+1}/{len(param_combinations)} ---")
            
            # Call the inner-loop function
            if mode == "simple":
                relevance, violations = compute_simple_attnlrp_pass(
                    conv_gamma=conv_gamma,
                    lin_gamma=lin_gamma,
                    model_wrapper=model_wrapper,
                    input_tensor=input_tensor,
                    checker=checker,
                    verbose=verbose  
                )
            elif mode == "knn":
                assert db_embeddings is not None, "db_embeddings must be provided for 'knn' mode."
                assert db_labels is not None, "db_labels must be provided for 'knn'' mode."
                assert ground_truth_label is not None, "ground_truth_label must be provided for 'knn' mode."

                relevance, violations = compute_knn_attnlrp_pass(
                    conv_gamma=conv_gamma,
                    lin_gamma=lin_gamma,
                    model_wrapper=model_wrapper,
                    input_tensor=input_tensor,
                    checker=checker,
                    verbose=verbose ,
                    db_embeddings=db_embeddings,  
                    db_labels=db_labels,  
                    ground_truth_label=ground_truth_label,  
                    k_neighbors=k_neighbors  
                )
            
            # Store the results
            key = (conv_gamma, lin_gamma)
            all_relevances[key] = relevance.detach().cpu()
            all_violations[key] = violations

    print("\n--- Gamma Sweep Complete ---")
    print("Model has been restored to its original state.")
    
    return all_relevances, all_violations