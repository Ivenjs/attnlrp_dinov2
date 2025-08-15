import torch
import torch.nn as nn
from typing import Dict, List, Optional, Dict, Any
import os
import zennit.rules as z_rules
from zennit.composites import LayerMapComposite
import itertools
from basemodel import TimmWrapper
from torch.utils.data import DataLoader
from dino_patcher import DINOPatcher
from tqdm import tqdm
from dataset import GorillaReIDDataset
from typing import Tuple

from knn_helpers import compute_knn_proxy_soft
            
def compute_simple_attnlrp_pass(
    conv_gamma: float, 
    lin_gamma: float, 
    model_wrapper: nn.Module, 
    input_tensor: torch.Tensor,
    verbose: bool = False
) -> torch.Tensor:
    """
    Computes a single LRP forward/backward pass for a given set of gamma rules.

    ASSUMES that the model is already patched by DINOPatcher, that zennit has been patched
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

        relevance = (input_tensor * input_tensor.grad).sum(dim=1, keepdim=True)

    finally:
        zennit_comp.remove()
    
    return relevance


def compute_knn_attnlrp_pass(
    conv_gamma: float, 
    lin_gamma: float, 
    model_wrapper: nn.Module, 
    input_tensor: torch.Tensor,
    # parameters required for the k-NN score
    query_label: str,         
    query_filename: str,      
    db_embeddings: torch.Tensor,
    db_labels: list,
    db_filenames: list,
    distance_metric: str = "euclidean",
    proxy_temp: float = 0.1,
    verbose: bool = False
) -> torch.Tensor:
    """
    Computes a single LRP pass explaining a k-NN classification decision.

    This function calculates a differentiable proxy score based on the k-NN
    outcome and backpropagates from it to generate the relevance map.
    """
    # Reset gradients for this specific pass
    input_tensor.grad = None

    # Zennit rules MUST be set and removed for each pass
    zennit_comp = LayerMapComposite( #TODO: patch with this the rest of the model
        [
            (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
            (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
        ]
    )
    
    try:
        zennit_comp.register(model_wrapper)

        query_embedding = model_wrapper(input_tensor.requires_grad_())
        knn_score = compute_knn_proxy_soft(
            query_embedding=query_embedding,
            query_label=query_label,
            query_filename=query_filename,
            db_embeddings=db_embeddings,
            db_labels=db_labels,
            db_filenames=db_filenames,
            distance_metric=distance_metric,
            temp=proxy_temp
        )
        if verbose:
            print(f"Explaining k-NN proxy score: {knn_score.item():.4f} for Gammas (Conv: {conv_gamma}, Lin: {lin_gamma})")

        knn_score.backward()
        
        if input_tensor.grad is None:
            if verbose:
                print(f"WARNING: No gradient for LRP on '{query_filename}'. "
                      "Producing a zero relevance map.")
            relevance = torch.zeros_like(input_tensor.sum(1, keepdim=True))
        else:
            # Standard LRP relevance calculation when gradients are present
            relevance = (input_tensor * input_tensor.grad).sum(1, keepdim=True)

    finally:
        zennit_comp.remove()

    return relevance

def generate_relevances(
    model_wrapper: TimmWrapper,
    dataloader: DataLoader,
    device: torch.device,
    # --- Parameters to sweep over ---
    conv_gamma_values: List[float],
    lin_gamma_values: List[float],
    distance_metrics: List[str] = ["cosine"],
    proxy_temp_values: List[float] = [0.1],
    # --- Mode and control flags ---
    mode: str = "knn",
    verbose: bool = False,
    # --- Database arguments for 'knn' mode ---
    db_embeddings: Optional[torch.Tensor] = None,
    db_filenames: Optional[List[str]] = None,
    db_labels: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    A unified function to generate relevance maps for a dataset.

    It can perform a sweep over multiple parameters and optionally include
    segmentation masks in its output.

    Args:
        model_wrapper: The model to explain.
        dataloader: DataLoader providing image batches.
        device: The torch device to run on.
        conv_gamma_values: List of convolutional gamma values for LRP.
        lin_gamma_values: List of linear layer gamma values for LRP.
        distance_metrics: List of distance metrics for KNN mode.
        proxy_temp_values: List of proxy temperatures for KNN mode.
        mode: Calculation mode, either "simple" or "knn".
        include_masks: If True, expects 'mask' in the dataloader batch and
                       includes it in the output.
        verbose: If True, prints detailed LRP pass information.
        db_embeddings: Database embeddings for KNN mode.
        db_filenames: Database filenames for KNN mode.
        db_labels: Database labels for KNN mode.

    Returns:
        A list of dictionaries. Each dictionary contains the results for a
        single image and a single parameter combination, with keys like:
        'filename', 'params', 'relevance', and 'mask' (if requested).
    """
    all_results = []

    print("--- Starting Relevance Generation ---")

    # Validate inputs for KNN mode
    if mode == "knn":
        assert db_embeddings is not None, "db_embeddings must be provided for 'knn' mode."
        assert db_filenames is not None, "db_filenames must be provided for 'knn' mode."

    # Create all combinations of parameters to iterate over.
    # This works even if lists have only one element.
    param_combinations = list(itertools.product(
        conv_gamma_values, lin_gamma_values, distance_metrics, proxy_temp_values
    ))
    
    print(f"Total parameter combinations to process: {len(param_combinations)}")
    print("Patching model for LRP...")

    with DINOPatcher(model_wrapper):
        for i, (conv_gamma, lin_gamma, distance_metric, proxy_temp) in enumerate(param_combinations):
            print(f"\n=== Processing Param Combination {i+1}/{len(param_combinations)}: "
                  f"conv_γ={conv_gamma}, lin_γ={lin_gamma}, dist={distance_metric}, proxy_temp={proxy_temp} ===")

            for batch in tqdm(dataloader, desc="Processing batches"):
                input_batch = batch["image"].to(device)
                labels_batch = batch["label"]
                filenames_batch = batch["filename"]
                
                # Safely get masks only if needed
                mask_batch = batch.get("mask") 
                if mask_batch is None:
                    raise ValueError("No 'mask' key found in the dataloader batch.")

                for j, filename in enumerate(filenames_batch):
                    input_tensor_single = input_batch[j].unsqueeze(0)
                    label_single = labels_batch[j]
                    
                    relevance_single = None
                    if mode == "simple":
                        relevance_single = compute_simple_attnlrp_pass(
                            model_wrapper=model_wrapper,
                            input_tensor=input_tensor_single,
                            conv_gamma=conv_gamma,
                            lin_gamma=lin_gamma,
                            verbose=verbose
                        )
                    elif mode == "knn":
                        relevance_single = compute_knn_attnlrp_pass(
                            model_wrapper=model_wrapper,
                            input_tensor=input_tensor_single,
                            query_label=label_single,
                            query_filename=filename,
                            db_embeddings=db_embeddings,
                            db_labels=db_labels,
                            db_filenames=db_filenames,
                            conv_gamma=conv_gamma,
                            lin_gamma=lin_gamma,
                            distance_metric=distance_metric,
                            proxy_temp=proxy_temp,
                            verbose=verbose
                        )

                    # Prepare the mask if requested
                    mask_single = None
                    mask_tensor_single = mask_batch[j]
                    if mask_tensor_single is not None:
                            mask_single = mask_tensor_single.cpu().numpy()

                    result_item = {
                        "filename": filename,
                        "params": {
                            "conv_gamma": conv_gamma,
                            "lin_gamma": lin_gamma,
                            "distance_metric": distance_metric,
                            "proxy_temp": proxy_temp,
                        },
                        "relevance": relevance_single.detach().cpu(),
                        "mask": mask_single,
                    }
                    all_results.append(result_item)

    print("\n--- Relevance Generation Complete ---")
    print("Model has been restored to its original state.")
    return all_results

def get_relevances(
    db_path: str,
    model_wrapper: TimmWrapper,
    dataloader: DataLoader,
    device: torch.device,
    recompute: bool=False,
    **kwargs
) -> Dict[str, Tuple[torch.Tensor, any]]:
    if os.path.exists(db_path) and not recompute:
        print(f"Loading cached relevance dictionary from: {db_path}")
        return torch.load(db_path, map_location="cpu")

    # 2. If no cache, perform the computation
    print(f"Relevance db not found. Computing relevances...")

    conv_gamma = kwargs.pop('conv_gamma')
    lin_gamma = kwargs.pop('lin_gamma')
    distance_metric = kwargs.pop('distance_metric', 'cosine')
    proxy_temp = kwargs.pop('proxy_temp')

    # Convert single values to the list format required by generate_relevances
    kwargs['conv_gamma_values'] = [conv_gamma]
    kwargs['lin_gamma_values'] = [lin_gamma]
    kwargs['distance_metrics'] = [distance_metric]
    kwargs['proxy_temp_values'] = [proxy_temp]

    results_list = generate_relevances(
        model_wrapper=model_wrapper,
        dataloader=dataloader,
        device=device,
        **kwargs 
    )

    print(f"Saving new relevance dictionary to: {db_path}")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    torch.save(results_list, db_path)
    
    return results_list
