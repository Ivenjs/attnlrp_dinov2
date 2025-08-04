import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Dict, Any


import zennit.rules as z_rules
from zennit.composites import LayerMapComposite

import datetime
import os
from zennit.image import imgify

from knn_helpers import compute_knn_proxy_score, compute_knn_proxy_score_batched
            
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

def compute_simple_attnlrp_pass_batched(
    conv_gamma: float, 
    lin_gamma: float, 
    model_wrapper: nn.Module, 
    input_batch: torch.Tensor, 
    verbose: bool = False
) -> torch.Tensor: 
    """
    Computes LRP passes for a BATCH of inputs explaining their k-NN decisions.
    """
    batch_size = input_batch.shape[0]
    input_batch.grad = None

    # Set Zennit rules once for the whole batch pass
    zennit_comp = LayerMapComposite(
        [
            (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
            (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
        ]
    )
    
    relevances_list = []

    try:
        zennit_comp.register(model_wrapper)

        query_embedding_batch = model_wrapper(input_batch.requires_grad_())

        for i in range(batch_size):
            # Isolate the data for the i-th sample
            query_embedding_i = query_embedding_batch[i].unsqueeze(0) # Shape [1, D]
            most_active_feature_idx = torch.argmax(query_embedding_i, dim=1).item()
            
            if verbose:
                print(f"  [Sample {i+1}/{batch_size}] Explaining k-NN score: {knn_score.item():.4f}")


            model_wrapper.zero_grad()  # Clear previous sample's gradients
            query_embedding_i[0, most_active_feature_idx].backward(retain_graph=True)

            relevance_i = (input_batch[i] * input_batch.grad[i]).sum(1, keepdim=True)
            relevances_list.append(relevance_i.detach().clone())
            


    finally:
        zennit_comp.remove()
        if input_batch.grad is not None:
            input_batch.grad = None
            
    relevance_batch = torch.stack(relevances_list)

    return relevance_batch

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
    k_neighbors: int = 5,
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
            query_label=query_label,
            query_filename=query_filename,
            db_embeddings=db_embeddings,
            db_labels=db_labels,
            db_filenames=db_filenames,
            distance_metric=distance_metric,
            k=k_neighbors
        )
        if verbose:
            print(f"Explaining k-NN proxy score: {knn_score.item():.4f} for Gammas (Conv: {conv_gamma}, Lin: {lin_gamma})")

        knn_score.backward()
        
        if input_tensor.grad is None:
            # This happens if the score was a constant (e.g., no friends found in top-k).
            # In this case, we return a zero relevance map as there's nothing to explain.
            # This avoids the TypeError and is conceptually correct for this scenario.
            if verbose:
                print(f"WARNING: No gradient for LRP on '{query_filename}'. "
                      f"This likely means no friends were found in its top-{k_neighbors} neighbors. "
                      "Producing a zero relevance map.")
            relevance = torch.zeros_like(input_tensor.sum(1, keepdim=True))
        else:
            # Standard LRP relevance calculation when gradients are present
            relevance = (input_tensor * input_tensor.grad).sum(1, keepdim=True)
        
    finally:
        zennit_comp.remove()

    return relevance

def compute_knn_attnlrp_pass_batched(
    conv_gamma: float, 
    lin_gamma: float, 
    model_wrapper: nn.Module, 
    input_batch: torch.Tensor, 
    query_labels_batch: List[str],         
    query_filenames_batch: List[str],      
    db_embeddings: torch.Tensor,
    db_labels: list,
    db_filenames: list,
    distance_metric: str = "euclidean",
    k_neighbors: int = 5,
    verbose: bool = False
) -> torch.Tensor: 
    """
    Computes LRP passes for a BATCH of inputs using a single, efficient backward pass.
    """
    input_batch.grad = None

    zennit_comp = LayerMapComposite(
        [
            (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
            (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
        ]
    )
    
    try:
        zennit_comp.register(model_wrapper)

        print(f"Computing k-NN scores for batch of size {input_batch.shape[0]}...") 


        # 1. Single forward pass
        query_embedding_batch = model_wrapper(input_batch.requires_grad_())

        # 2. Compute all scores for the batch at once
        knn_scores_vector = compute_knn_proxy_score_batched( # Use the new batched function
            query_embedding_batch=query_embedding_batch,
            query_labels_batch=query_labels_batch,
            query_filenames_batch=query_filenames_batch,
            db_embeddings=db_embeddings,
            db_labels=db_labels,
            db_filenames=db_filenames,
            distance_metric=distance_metric,
            k=k_neighbors
        ) # This returns a tensor of shape [batch_size]
        
        if verbose:
            print(f"Explaining k-NN proxy scores for batch. Mean score: {knn_scores_vector.mean().item():.4f}")

        # 3. Single, collective backward pass. No loop, no retain_graph!
        # Backpropagating from the sum achieves the same result as looping, but efficiently.
        knn_scores_vector.sum().backward()

        # 4. Compute relevance for the entire batch in one vectorized operation.
        relevance_batch = (input_batch * input_batch.grad).sum(1, keepdim=True)

    finally:
        zennit_comp.remove()

    return relevance_batch
