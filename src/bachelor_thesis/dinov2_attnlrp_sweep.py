import itertools
import torch
import torch.nn as nn
from typing import List
import yaml
import os
from PIL import Image

from lrp_helpers import compute_simple_attnlrp_pass, compute_knn_attnlrp_pass, LRPConservationChecker
from knn_helpers import load_knn_db, fill_knn_db
from basemodel import TimmWrapper

from dino_patcher import DINOPatcher

CONV_GAMMAS = [0.1, 0.25, 1.0]
LIN_GAMMAS = [0.0, 0.05, 0.1, 0.25]

def run_gamma_sweep(
    model_wrapper: TimmWrapper, 
    transforms: nn.Module,
    input_image_path: str,
    device: torch.device,
    mode: str = "simple",  # "simple" or "knn"
    conv_gamma_values: List[float] = CONV_GAMMAS,
    lin_gamma_values: List[float] = LIN_GAMMAS
):
    """
    Runs a sweep over gamma parameters, managing patches efficiently.
    """
    ground_truth_label = os.path.basename(input_image_path).split("_")[0]
    image = Image.open(input_image_path).convert(
        "RGB"
    )
    input_tensor = transforms(image).unsqueeze(0).to(device)

    all_relevances = {}
    all_violations = {}

    if mode == "knn":
        db_embeddings = []
        db_labels = []

        model_config_path = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/configs/model_config.yaml"
        with open(model_config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        checkpoint_name = os.path.basename(cfg["checkpoint_path"])
        checkpoint_base = os.path.splitext(checkpoint_name)[0]

        knn_db_dir = "/workspaces/bachelor_thesis_code/knn_db"
        files_in_dir = os.listdir(knn_db_dir)
        matching_checkpoints = [f for f in files_in_dir if checkpoint_base in f]
        if matching_checkpoints:
            print(f"KNN database for checkpoint {checkpoint_base} already exists. Loading the KNN database...")
            db_embeddings, db_labels = load_knn_db(os.path.join(knn_db_dir, matching_checkpoints[0]), device)

        else:
            print(f"KNN database for checkpoint {checkpoint_base} does not exist. Filling the KNN database...")
            db_embeddings, db_labels = fill_knn_db(
                image_dir="/workspaces/vast-gorilla/gorillawatch/data/eval_body_squared_cleaned_open_2024_bigval/train",
                model=model_wrapper,
                device=device,
                output_dir=knn_db_dir,
                model_checkpoint=checkpoint_name,
                transform=transforms
            )
    

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
                    verbose=False  
                )
            elif mode == "knn":
                relevance, violations = compute_knn_attnlrp_pass(
                    conv_gamma=conv_gamma,
                    lin_gamma=lin_gamma,
                    model_wrapper=model_wrapper,
                    input_tensor=input_tensor,
                    checker=checker,
                    verbose=False ,
                    db_embeddings=db_embeddings,  
                    db_labels=db_labels,  
                    ground_truth_label=ground_truth_label,  
                    k_neighbors=5  
                )
            
            # Store the results
            key = (conv_gamma, lin_gamma)
            all_relevances[key] = relevance.detach().cpu()
            all_violations[key] = violations

    print("\n--- Gamma Sweep Complete ---")
    print("Model has been restored to its original state.")
    
    return all_relevances, all_violations