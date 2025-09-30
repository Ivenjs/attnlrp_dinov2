import torch
import torch.nn as nn
from typing import Dict, List, Optional, Dict, Any
import os
import torch.nn.functional as F
import zennit.rules as z_rules
from zennit.composites import LayerMapComposite
import itertools
from basemodel import TimmWrapper
from torch.utils.data import DataLoader
from dino_patcher import DINOPatcher
from lxt.efficient.rules import identity_rule_implicit
from tqdm import tqdm
from typing import Tuple
from knn_helpers import create_exclusion_mask

from zennit.composites import NameLayerMapComposite
from zennit.rules import Epsilon, Gamma, Pass
from zennit.layer import Sum
            
class AttentionEpsilon(z_rules.Epsilon):
    def __call__(self, module, *args, **kwargs):
        # only apply if module is inside attention
        if "attn" in module.__repr__():
            return super().__call__(module, *args, **kwargs)
        # otherwise: return None → Zennit will keep searching for another match
        return None

class FFNGamma(z_rules.Gamma):
    def __call__(self, module, *args, **kwargs):
        # only apply if NOT in attention
        if "attn" not in module.__repr__():
            return super().__call__(module, *args, **kwargs)
        return None

def create_vit_lrp_composite_also_old(conv_gamma: float, lin_gamma: float, epsilon: float = 1e-6):
    """
    Vision Transformer LRP composite:
    - Conv2d → Gamma
    - Linear in attention → Epsilon
    - Linear outside attention → Gamma
    """
    return LayerMapComposite([
        (nn.Conv2d, z_rules.Gamma(gamma=conv_gamma)),
        (nn.Linear, AttentionEpsilon(epsilon=epsilon)),
        (nn.Linear, FFNGamma(gamma=lin_gamma)),
    ])

def create_old_composite(conv_gamma: float, lin_gamma: float):
    return LayerMapComposite([
        (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
        (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
    ])

def create_dinov2_lrp_composite(
    model: nn.Module,
    conv_gamma: float,
    lin_gamma: float,
    epsilon: float = 1e-6,
    verbose: bool = False
):
    """
    Creates a Zennit composite for a ViT using NameLayerMapComposite.
    - Conv2d layers -> Gamma(conv_gamma)
    - Linear layers in FFN -> Gamma(ffn_gamma)
    - Linear layers in Attention -> Epsilon(epsilon)
    """
    if verbose:
        print("--- Building DINOv2 LRP Composite ---")

    # 1. Find the names of all Linear layers
    all_linear_names = {name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}
    
    # 2. Find the names of attention-specific linear layers to override
    attn_linear_names = {name for name in all_linear_names if '.attn.' in name}
    
    # 3. The rest are FFN linear layers
    ffn_linear_names = all_linear_names - attn_linear_names

    if verbose:
        print(f"Found {len(attn_linear_names)} attention Linear layers to be mapped to Epsilon.")
        print("  - Sample:", list(attn_linear_names)[:5])
        print(f"Found {len(ffn_linear_names)} FFN/other Linear layers to be mapped to Gamma.")
        print("  - Sample:", list(ffn_linear_names)[:5])


    name_map = [
        (list(attn_linear_names), Epsilon(epsilon=epsilon)),
    ]

    layer_map = [
        (nn.Conv2d, Gamma(gamma=conv_gamma)),
        (nn.Linear, Gamma(gamma=lin_gamma)), 
    ]
    
    
    if verbose:
        print("---------------------------------------")
        
    # NameLayerMapComposite checks name_map first, then layer_map.
    return NameLayerMapComposite(name_map, layer_map)

def compute_similarity_proto_margin_pass(
    conv_gamma: float,
    lin_gamma: float,
    model_wrapper: torch.nn.Module,
    input_tensor: torch.Tensor,
    query_label: str,
    query_filename: str,
    query_video_id: str,
    db_embeddings: torch.Tensor,    # (N, D)
    db_labels: list,                # len N
    db_filenames: list,             # len N
    db_video_ids: list,             # len N
    distance_metric: str = "cosine",
    temp: float = 0.05,
    topk_neg: int = 50,
    cross_encounter: bool = True,
    verbose: bool = False
) -> torch.Tensor:
    """
    Prototype-Margin LRP: proto_pos = softmax over friend sims, proto_neg = softmax over topk foe sims.
    score = sim(query, proto_pos) - sim(query, proto_neg)
    """
    input_tensor.grad = None

    zennit_comp = create_dinov2_lrp_composite(model_wrapper, conv_gamma, lin_gamma)
    try:
        zennit_comp.register(model_wrapper)

        query_embedding = model_wrapper(input_tensor.requires_grad_())
        score = compute_knn_proto_margin(
            query_embedding=query_embedding,
            query_label=query_label,
            query_filename=query_filename,
            query_video_id=query_video_id,
            db_embeddings=db_embeddings,
            db_labels=db_labels,
            db_filenames=db_filenames,
            db_video_ids=db_video_ids,
            distance_metric=distance_metric,
            temp=temp,
            topk_neg=topk_neg,
            cross_encounter=cross_encounter
        )

        score.backward()
        if verbose:
            print(f"Explaining k-NN score ({score.item():.4f}) for Gammas (Conv: {conv_gamma}, Lin: {lin_gamma})")

        if input_tensor.grad is None:
            relevance = torch.zeros_like(input_tensor.sum(1, keepdim=True))
            
        else:
            relevance = (input_tensor * input_tensor.grad).sum(1, keepdim=True)

    finally:
        zennit_comp.remove()
    if relevance is None:
        relevance = torch.zeros_like(input_tensor.sum(1, keepdim=True))
        
    return relevance



def compute_similarity_lrp_pass(
    conv_gamma: float, 
    lin_gamma: float, 
    model_wrapper: nn.Module, 
    input_tensor: torch.Tensor,
    query_label: str,
    query_filename: str,
    query_video_id: str,
    db_embeddings: torch.Tensor,
    db_labels: list,
    db_filenames: list,
    db_video_ids: list,
    distance_metric: str = "cosine",
    cross_encounter: bool = True,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Computes LRP by explaining the cosine similarity between the output embedding
    and a reference embedding of the same identity. For simple comparisons, the
    reference embedding and its index are being returned
    """
        
    input_tensor.grad = None

    zennit_comp = create_dinov2_lrp_composite(model_wrapper, conv_gamma, lin_gamma)
    
    try:
        zennit_comp.register(model_wrapper)

        query_embedding = model_wrapper(input_tensor.requires_grad_())

        similarity_score, reference_embedding, ref_idx = compute_similarity_score(
            query_embedding=query_embedding, 
            query_label=query_label, 
            query_filename=query_filename, 
            query_video_id=query_video_id,
            db_embeddings=db_embeddings, 
            db_labels=db_labels, 
            db_filenames=db_filenames,
            db_video_ids=db_video_ids,
            cross_encounter=cross_encounter,
            distance_metric=distance_metric
        )

        if verbose:
            print(f"Explaining similarity score ({similarity_score.item():.4f}) for Gammas (Conv: {conv_gamma}, Lin: {lin_gamma})")

        similarity_score.backward()

        if input_tensor.grad is None:
            
            relevance = torch.zeros_like(input_tensor.sum(1, keepdim=True))
        else:
            # Standard LRP relevance calculation when gradients are present
            relevance = (input_tensor * input_tensor.grad).sum(1, keepdim=True)

    finally:
        zennit_comp.remove()
    if relevance is None:
        relevance = torch.zeros_like(input_tensor.sum(1, keepdim=True))
        
    return relevance, reference_embedding, ref_idx

def compute_knn_topk_attnlrp_pass(
    conv_gamma: float, 
    lin_gamma: float, 
    model_wrapper: nn.Module, 
    input_tensor: torch.Tensor,
    # parameters required for the k-NN score
    query_label: str,         
    query_filename: str,      
    query_video_id: str,
    db_embeddings: torch.Tensor,
    db_labels: list,
    db_filenames: list,
    db_video_ids: list,
    distance_metric: str = "cosine",
    proxy_temp: float = 0.1,
    topk: int = 50,
    cross_encounter: bool = True,
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
    zennit_comp = create_dinov2_lrp_composite(model_wrapper, conv_gamma, lin_gamma)
    
    try:
        zennit_comp.register(model_wrapper)

        query_embedding = model_wrapper(input_tensor.requires_grad_())
        knn_score = compute_knn_proxy_soft_topk(
            query_embedding=query_embedding,
            query_label=query_label,
            query_filename=query_filename,
            query_video_id=query_video_id,
            db_embeddings=db_embeddings,
            db_labels=db_labels,
            db_filenames=db_filenames,
            db_video_ids=db_video_ids,
            distance_metric=distance_metric,
            temp=proxy_temp,
            topk=topk,
            cross_encounter=cross_encounter
        )
        if verbose:
            print(f"Explaining k-NN proxy score: {knn_score.item():.4f} for Gammas (Conv: {conv_gamma}, Lin: {lin_gamma})")

        knn_score.backward()
        
        if input_tensor.grad is None:
            
            relevance = torch.zeros_like(input_tensor.sum(1, keepdim=True))
        else:
            # Standard LRP relevance calculation when gradients are present
            relevance = (input_tensor * input_tensor.grad).sum(1, keepdim=True)

    finally:
        zennit_comp.remove()

    if relevance is None:
        
        relevance = torch.zeros_like(input_tensor.sum(1, keepdim=True))
    return relevance

def compute_knn_all_attnlrp_pass(
    conv_gamma: float, 
    lin_gamma: float, 
    model_wrapper: nn.Module, 
    input_tensor: torch.Tensor,
    # parameters required for the k-NN score
    query_label: str,         
    query_filename: str,      
    query_video_id: str,
    db_embeddings: torch.Tensor,
    db_labels: list,
    db_filenames: list,
    db_video_ids: list,
    distance_metric: str = "cosine",
    proxy_temp: float = 0.1,
    cross_encounter: bool = True,
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
    zennit_comp = create_dinov2_lrp_composite(model_wrapper, conv_gamma, lin_gamma)

    try:
        zennit_comp.register(model_wrapper)

        query_embedding = model_wrapper(input_tensor.requires_grad_())
        
        knn_score = compute_knn_proxy_soft_all(
            query_embedding=query_embedding,
            query_label=query_label,
            query_filename=query_filename,
            query_video_id=query_video_id,
            db_embeddings=db_embeddings,
            db_labels=db_labels,
            db_filenames=db_filenames,
            db_video_ids=db_video_ids,
            distance_metric=distance_metric,
            temp=proxy_temp,
            cross_encounter=cross_encounter
        )
        if verbose:
            print(f"Explaining k-NN proxy score: {knn_score.item():.4f} for Gammas (Conv: {conv_gamma}, Lin: {lin_gamma})")

        knn_score.backward()
        
        if input_tensor.grad is None:
            
            relevance = torch.zeros_like(input_tensor.sum(1, keepdim=True))
        else:
            # Standard LRP relevance calculation when gradients are present
            relevance = (input_tensor * input_tensor.grad).sum(1, keepdim=True)

    finally:
        zennit_comp.remove()

    if relevance is None:
        
        relevance = torch.zeros_like(input_tensor.sum(1, keepdim=True))
    return relevance

def generate_relevances(
    model_wrapper: TimmWrapper,
    dataloader: DataLoader,
    device: torch.device,
    # --- Parameters to sweep over ---
    conv_gamma_values: List[float],
    lin_gamma_values: List[float],
    # --- Mode and mode-specific parameters ---
    mode: str,
    distance_metrics: List[str] = ["cosine"],
    proxy_temp_values: List[float] = [0.1],
    topk_values: List[int] = [50],
    param_combinations_list: Optional[List[Dict[str, Any]]] = None,
    # --- Control flags ---
    verbose: bool = False,
    # --- Database arguments for relevant modes ---
    db_embeddings: Optional[torch.Tensor] = None,
    db_filenames: Optional[List[str]] = None,
    db_labels: Optional[List[str]] = None,
    db_video_ids: Optional[List[str]] = None,
    cross_encounter: bool = True,
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
        distance_metrics: List[str] = ["cosine"],
        proxy_temp_values: List[float] = [0.1],
        topk_values: List[int] = [50],
        mode: str,
        include_masks: bool = False,
        verbose: bool = False,
        db_embeddings: Optional[torch.Tensor] = None,
        db_filenames: Database filenames for KNN mode.
        db_labels: Database labels for KNN mode.

    Returns:
        A list of dictionaries. Each dictionary contains the results for a
        single image and a single parameter combination, with keys like:
        'filename', 'params', 'relevance', and 'mask' (if requested).
    """
    all_results = []

    print(f"--- Starting Relevance Generation (Mode: {mode}) ---")
    if mode in ["knn", "proto_margin", "similarity"]:
        assert db_embeddings is not None, f"db_embeddings must be provided for '{mode}' mode."
        assert db_filenames is not None, f"db_filenames must be provided for '{mode}' mode."
        assert db_labels is not None, f"db_labels must be provided for '{mode}' mode."

    if param_combinations_list is not None:
        print(f"Running with {len(param_combinations_list)} specific parameter combinations.")
        param_combinations_structured = []
        for p_dict in param_combinations_list:
            base_tuple = (p_dict['conv_gamma'], p_dict['lin_gamma'], p_dict['distance_metric'])
            if mode == "soft_knn_margin_all":
                mode_tuple = (p_dict['proxy_temp'],)
                param_combinations_structured.append((base_tuple, mode_tuple))
            elif mode in ["proto_margin", "soft_knn_margin_topk"]:
                mode_tuple = (p_dict['proxy_temp'], p_dict['topk'])
                param_combinations_structured.append((base_tuple, mode_tuple))
            else: 
                param_combinations_structured.append(base_tuple)
    else:
        base_params = list(itertools.product(conv_gamma_values, lin_gamma_values, distance_metrics))
        if mode == "soft_knn_margin_all":
            mode_specific_params = list(itertools.product(proxy_temp_values))
            param_combinations_structured = list(itertools.product(base_params, mode_specific_params))
        elif mode in ["proto_margin", "soft_knn_margin_topk"]:
            mode_specific_params = list(itertools.product(proxy_temp_values, topk_values))
            param_combinations_structured = list(itertools.product(base_params, mode_specific_params))
        else: 
            param_combinations_structured = base_params


    with DINOPatcher(model_wrapper):
        for params in param_combinations_structured:
            if mode == "soft_knn_margin_all":
                (conv_gamma, lin_gamma, distance_metric), (proxy_temp,) = params
                topk = None # This mode doesn't use topk
            elif mode in ["proto_margin", "soft_knn_margin_topk"]:
                (conv_gamma, lin_gamma, distance_metric), (proxy_temp, topk) = params
            else: 
                conv_gamma, lin_gamma, distance_metric = params
                proxy_temp, topk = None, None

            for batch in tqdm(dataloader, desc="Processing batches"):
                input_batch = batch["image"].to(device)
                labels_batch = batch["label"]
                filenames_batch = batch["filename"]
                video_ids_batch = batch["video"]

                # Safely get masks only if needed
                mask_batch = batch.get("mask") 
                if mask_batch is None:
                    raise ValueError("No 'mask' key found in the dataloader batch.")

                for j, filename in enumerate(filenames_batch):
                    input_tensor_single = input_batch[j].unsqueeze(0)
                    label_single = labels_batch[j]
                    video_id = video_ids_batch[j]

                    relevance_single = None
                    extra_info = {} # To store mode-specific data like reference_embedding
                    if mode == "soft_knn_margin_all":
                        relevance_single = compute_knn_all_attnlrp_pass(
                            model_wrapper=model_wrapper, input_tensor=input_tensor_single,
                            query_label=label_single, query_filename=filename, query_video_id=video_id,
                            db_embeddings=db_embeddings, db_labels=db_labels, db_filenames=db_filenames,
                            db_video_ids=db_video_ids,
                            conv_gamma=conv_gamma, lin_gamma=lin_gamma,
                            distance_metric=distance_metric, proxy_temp=proxy_temp, cross_encounter=cross_encounter, verbose=verbose
                        )
                    elif mode == "soft_knn_margin_topk":
                        relevance_single = compute_knn_topk_attnlrp_pass(
                            model_wrapper=model_wrapper, input_tensor=input_tensor_single,
                            query_label=label_single, query_filename=filename, query_video_id=video_id,
                            db_embeddings=db_embeddings, db_labels=db_labels, db_filenames=db_filenames,
                            db_video_ids=db_video_ids,
                            conv_gamma=conv_gamma, lin_gamma=lin_gamma,
                            distance_metric=distance_metric, proxy_temp=proxy_temp, topk=topk, cross_encounter=cross_encounter, verbose=verbose
                        )
                    elif mode == "proto_margin":
                        relevance_single = compute_similarity_proto_margin_pass(
                             model_wrapper=model_wrapper, input_tensor=input_tensor_single,
                             query_label=label_single, query_filename=filename, query_video_id=video_id,
                             db_embeddings=db_embeddings, db_labels=db_labels, db_filenames=db_filenames,
                             db_video_ids=db_video_ids,
                             conv_gamma=conv_gamma, lin_gamma=lin_gamma,
                             distance_metric=distance_metric, temp=proxy_temp, topk_neg=topk, cross_encounter=cross_encounter, verbose=verbose
                        )
                    elif mode == "similarity":
                        relevance_single, reference_embedding, ref_idx = compute_similarity_lrp_pass(
                            model_wrapper=model_wrapper, input_tensor=input_tensor_single,
                            query_label=label_single, query_filename=filename, query_video_id=video_id,
                            db_embeddings=db_embeddings, db_labels=db_labels, db_filenames=db_filenames,
                            db_video_ids=db_video_ids,
                            conv_gamma=conv_gamma, lin_gamma=lin_gamma, distance_metric=distance_metric,
                            cross_encounter=cross_encounter, verbose=verbose
                        )
                        # Save the reference embedding used, for evaluation
                        extra_info["reference_embedding"] = reference_embedding.cpu() if reference_embedding is not None else reference_embedding
                        extra_info["reference_filename"] = db_filenames[ref_idx]

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
                            "topk": topk
                        },
                        "mode": mode,
                        "relevance": relevance_single.detach().cpu(),
                        "mask": mask_single, #this should probably not be included here. we have access to the mask with a normal dataloader anyway
                        **extra_info
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
        return torch.load(db_path, map_location="cpu", weights_only=False)

    # 2. If no cache, perform the computation
    print(f"Relevance db at {db_path} not found. Computing relevances...")

    conv_gamma = kwargs.pop('conv_gamma')
    lin_gamma = kwargs.pop('lin_gamma')
    distance_metric = kwargs.pop('distance_metric', 'cosine')
    proxy_temp = kwargs.pop('proxy_temp')
    topk = kwargs.pop('topk')

    # Convert single values to the list format required by generate_relevances
    kwargs['conv_gamma_values'] = [conv_gamma]
    kwargs['lin_gamma_values'] = [lin_gamma]
    kwargs['distance_metrics'] = [distance_metric]
    kwargs['proxy_temp_values'] = [proxy_temp]
    kwargs['topk_values'] = [topk]

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

def compute_knn_proto_margin(
    query_embedding: torch.Tensor,
    query_label: str,
    query_filename: str,
    query_video_id: str,
    db_embeddings: torch.Tensor,
    db_labels: list,
    db_filenames: list,
    db_video_ids: list,
    distance_metric: str = "cosine",
    temp: float = 0.05,
    topk_neg: int = 50,
    cross_encounter: bool = True,
    exclude_self: bool = True
):
    """Computes a margin score based on prototypes of positive and hard-negative neighbors.

    This score is conceptually similar to a hard triplet loss but uses stable
    prototypes instead of single instances. It constructs two prototypes:
    1.  A 'positive prototype' by averaging embeddings of the same class.
    2.  A 'hard-negative prototype' by averaging the `topk_neg` most confusing
        embeddings from different classes.
    The final score is the margin: sim(query, proto_pos) - sim(query, proto_neg).
    Maximizing this score pushes the query to be more like its class archetype
    and less like its hardest distractors.

    Interpretation of the score:
      - Near +2.0: Ideal. Query is a perfect archetype of its class and is
                    maximally dissimilar from hard negatives.
      -      0.0: Ambiguous. Query is equidistant from the positive and
                   hard-negative prototypes.
      - Near -2.0: Catastrophic. Query is the opposite of its own class prototype
                    and perfectly matches the hard-negative prototype.

    Args:
        query_embedding: The embedding for the query item.
        query_label: The label of the query item.
        query_filename: The filename of the query, for self-exclusion.
        db_embeddings: A tensor of all embeddings in the database.
        db_labels: A list of all labels in the database.
        db_filenames: A list of all filenames, for self-exclusion.
        distance_metric: The metric used; only 'cosine' is supported.
        temp: Temperature for softmax weighting of prototypes.
        topk_neg: The number of hard negatives to average for the prototype.
        cross_encounter: If True, only allows cross-encounter comparisons.
        exclude_self: If True, the query is excluded from its own neighborhood.

    Returns:
        torch.Tensor: A scalar tensor representing the prototype-based margin score,
                      with a value in the range [-2, 2].
    """
    if distance_metric != "cosine":
        raise NotImplementedError("This proto margin is optimized for cosine similarity.")

    if query_embedding.dim() == 1:
        query_embedding = query_embedding.view(1, -1)

    qn = identity_rule_implicit(lambda t: F.normalize(t, p=2, dim=1), query_embedding)
    dbn = identity_rule_implicit(lambda t: F.normalize(t, p=2, dim=1), db_embeddings)

    # convert to input dtype, which is float32 for lrp
    if dbn.dtype != qn.dtype:
        dbn = dbn.to(qn.dtype)

    sims = F.linear(dbn, qn).squeeze(1)  # (N,)

    exclusion_mask = create_exclusion_mask(
        query_filename=query_filename,
        query_video_id=query_video_id,
        db_filenames=db_filenames,
        db_video_ids=db_video_ids,
        device=sims.device,
        exclude_self=exclude_self,
        cross_encounter=cross_encounter
    )

    sims[exclusion_mask] = torch.finfo(sims.dtype).min

    # masks
    device = sims.device
    labels_tensor = torch.tensor([1 if l == query_label else 0 for l in db_labels], device=device)

    pos_mask = (labels_tensor == 1)
    valid_pos_sims = sims[pos_mask]
    if valid_pos_sims.numel() > 0 and valid_pos_sims.max() > -1e8: # Check if any valid positives exist, e.g. cross video not itself
        pos_idx_original = torch.nonzero(pos_mask).squeeze(1)
        sims_pos = sims[pos_idx_original]
        alpha_pos = F.softmax(sims_pos / temp, dim=0)
        proto_pos = (alpha_pos.unsqueeze(1) * dbn[pos_idx_original]).sum(dim=0, keepdim=True)
    else:
        # Fallback if no valid cross-video positives are found
        proto_pos = torch.zeros_like(qn)

    # proto_neg: topk among negatives
    neg_mask = (labels_tensor == 0)
    neg_idxs = torch.nonzero(neg_mask).squeeze(1)
    if neg_idxs.numel() == 0:
        proto_neg = torch.zeros_like(proto_pos)
    else:
        sims_negs = sims[neg_idxs]
        k = min(topk_neg, sims_negs.numel())
        topk_vals, topk_idx_in_negs = sims_negs.topk(k)
        chosen = neg_idxs[topk_idx_in_negs]
        alpha_neg = F.softmax(topk_vals / temp, dim=0)
        proto_neg = (alpha_neg.unsqueeze(1) * dbn[chosen]).sum(dim=0, keepdim=True)  # (1, D)

    # similarity scalars
    sim_pos = (qn * proto_pos).sum()
    sim_neg = (qn * proto_neg).sum()
    return sim_pos - sim_neg

def compute_knn_proxy_soft_all(
    query_embedding: torch.Tensor,
    query_label: str,
    query_filename: str,
    query_video_id: str,
    db_embeddings: torch.Tensor,
    db_labels: list,
    db_filenames: list,
    db_video_ids: list,
    distance_metric: str = "cosine",
    temp: float = 0.05,
    cross_encounter: bool = True,
    exclude_self: bool = True
) -> torch.Tensor:
    """Computes a differentiable proxy for a k-NN classifier's confidence.

    This function provides a "soft" version of a k-NN decision. Instead of a
    hard top-k vote, it uses a softmax over all database similarities to create
    a probability-like distribution. The final score is a margin between the
    total weight assigned to "friends" (same label) and "foes" (different labels).
    Maximizing this score encourages the formation of clean, well-separated
    class clusters in the embedding space.

    Interpretation of the score:
      - +1.0: Maximum confidence. Query is deep in its correct class cluster.
      -  0.0: Maximum confusion. Query is on the decision boundary, equally
               close to friends and foes.
      - -1.0: Catastrophically wrong. Query is deep in an incorrect class cluster.

    Args:
        query_embedding: The embedding for the query item.
        query_label: The label of the query item.
        query_filename: The filename of the query, for self-exclusion.
        db_embeddings: A tensor of all embeddings in the database.
        db_labels: A list of all labels in the database.
        db_filenames: A list of all filenames, for self-exclusion.
        distance_metric: The metric used; only 'cosine' is supported.
        temp: Temperature for the softmax. Lower values create a "harder" k-NN
              approximation by focusing on the nearest neighbors.
        exclude_self: If True, the query is excluded from its own neighborhood.

    Returns:
        torch.Tensor: A scalar tensor representing the soft k-NN margin score,
                      with a value in the range [-1, 1].
    """
    if distance_metric != "cosine":
        raise NotImplementedError("This soft proxy is optimized for cosine similarity.")

    # Ensure embeddings are normalized (standard for cosine similarity)
    q_emb = query_embedding.view(1, -1) if query_embedding.dim()==1 else query_embedding
    # Achtibat:
    q_norm = identity_rule_implicit(lambda t: F.normalize(t, p=2, dim=1), q_emb)
    db_norm = identity_rule_implicit(lambda t: F.normalize(t, p=2, dim=1), db_embeddings)

    # convert to input dtype, which is float32 for lrp
    if db_norm.dtype != q_norm.dtype:
        db_norm = db_norm.to(q_norm.dtype)
    # Calculate cosine similarity (higher is better)
    # Note: F.linear(db_norm, q_norm) is equivalent to db_norm @ q_norm.T
    similarities = F.linear(db_norm, q_norm).squeeze(1) # Shape: (N,)

    exclusion_mask = create_exclusion_mask(
        query_filename=query_filename,
        query_video_id=query_video_id,
        db_filenames=db_filenames,
        db_video_ids=db_video_ids,
        device=similarities.device,
        exclude_self=exclude_self,
        cross_encounter=cross_encounter
    )

    similarities[exclusion_mask] = torch.finfo(similarities.dtype).min

    # Differentiable soft neighbor weights via softmax. `temp` controls sharpness.
    # Low temp -> focuses on the very nearest neighbors.
    # High temp -> considers more neighbors.
    weights = F.softmax(similarities / temp, dim=0)

    # Create a mask to identify friends in the database
    device = weights.device
    friend_mask = torch.tensor([1.0 if label == query_label else 0.0 for label in db_labels], device=device)
    
    # Calculate the total "probability mass" assigned to friends vs. foes
    prob_friends = (weights * friend_mask).sum()
    prob_foes = (weights * (1.0 - friend_mask)).sum() # or 1.0 - prob_friends

    # The margin score is the most faithful proxy for a contrastive decision.
    # Maximizing this score means maximizing friend probability and minimizing foe probability.
    score_margin = prob_friends - prob_foes

    return score_margin

def compute_knn_proxy_soft_topk(
    query_embedding: torch.Tensor,
    query_label: str,
    query_filename: str,
    query_video_id: str,
    db_embeddings: torch.Tensor,
    db_labels: list,
    db_filenames: list,
    db_video_ids: list,
    distance_metric: str = "cosine",
    temp: float = 0.05,
    topk: int = 5,
    cross_encounter: bool = True,
    exclude_self: bool = True
) -> torch.Tensor:
    """Computes a differentiable proxy for a k-NN classifier's confidence.

    This function provides a "soft" version of a k-NN decision. It first
    performs a hard selection of the top-k nearest neighbors and then uses a
    softmax over their similarities to create a probability-like distribution.
    The final score is a margin between the total weight assigned to "friends"
    (same label) and "foes" (different labels) within that top-k set.
    Maximizing this score encourages the query's true matches to be among its
    closest neighbors.

    Interpretation of the score:
      - +1.0: Maximum confidence. All top-k neighbors are friends.
      -  0.0: Maximum confusion. Friend/foe weights are balanced in the top-k.
      - -1.0: Catastrophically wrong. All top-k neighbors are foes.

    Args:
        query_embedding: The embedding for the query item.
        query_label: The label of the query item.
        query_filename: The filename of the query, for self-exclusion.
        query_video_id: The video ID of the query, for cross-encounter setting.
        db_embeddings: A tensor of all embeddings in the database.
        db_labels: A list of all labels in the database.
        db_filenames: A list of all filenames, for self-exclusion.
        db_video_ids: A list of all video IDs, for cross-encounter setting.
        k: The number of nearest neighbors to consider.
        distance_metric: The metric used; only 'cosine' is supported.
        temp: Temperature for the softmax. Lower values create a "harder" vote
              by focusing on the single nearest neighbor.
        cross_encounter: If True, excludes all items from the same video as the query.
        exclude_self: If True, the query is excluded from its own neighborhood
                      (has no effect if cross_encounter is True).

    Returns:
        torch.Tensor: A scalar tensor representing the soft k-NN margin score,
                      with a value in the range [-1, 1].
    """
    if distance_metric != "cosine":
        raise NotImplementedError("This soft proxy is optimized for cosine similarity.")

    q_emb = query_embedding.view(1, -1) if query_embedding.dim() == 1 else query_embedding
    q_norm = identity_rule_implicit(lambda t: F.normalize(t, p=2, dim=1), q_emb)
    db_norm = identity_rule_implicit(lambda t: F.normalize(t, p=2, dim=1), db_embeddings)

    # convert to input dtype, which is float32 for lrp
    if db_norm.dtype != q_norm.dtype:
        db_norm = db_norm.to(q_norm.dtype)
    
    all_similarities = F.linear(db_norm, q_norm).squeeze(1) # Shape: (N,)

    exclusion_mask = create_exclusion_mask(
        query_filename=query_filename,
        query_video_id=query_video_id,
        db_filenames=db_filenames,
        db_video_ids=db_video_ids,
        device=all_similarities.device,
        exclude_self=exclude_self,
        cross_encounter=cross_encounter
    )
    all_similarities[exclusion_mask] = torch.finfo(all_similarities.dtype).min

    num_valid_db_items = (~exclusion_mask).sum().item()
    effective_k = min(topk, num_valid_db_items)

    if effective_k == 0:
        return torch.tensor(0.0, device=query_embedding.device)

    top_k_similarities, top_k_indices = torch.topk(
        all_similarities, k=effective_k, dim=0
    )

    top_k_weights = F.softmax(top_k_similarities / temp, dim=0)

    device = top_k_weights.device
    friend_mask_full = torch.tensor(
        [1.0 if label == query_label else 0.0 for label in db_labels],
        device=device
    )
    top_k_friend_mask = friend_mask_full[top_k_indices]

    prob_friends = (top_k_weights * top_k_friend_mask).sum()
    prob_foes = (top_k_weights * (1.0 - top_k_friend_mask)).sum()

    score_margin = prob_friends - prob_foes

    return score_margin

def compute_similarity_score(
        query_embedding: torch.Tensor, 
        query_label: str, query_filename: str, 
        query_video_id: str,
        db_embeddings: torch.Tensor, 
        db_labels: list, 
        db_filenames: list,
        db_video_ids: list,
        cross_encounter: bool = True,
        distance_metric: str = "cosine",
        exclude_self: bool = True,
        reference_embedding: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Computes the cosine similarity between two embeddings of the same class.

    This score measures the direct, one-to-one alignment between two vectors.
    It is calculated as the dot product of the L2-normalized embeddings.

    Interpretation of the score:
      - +1.0: Perfect alignment. Vectors point in the same direction.
      -  0.0: Orthogonal. Vectors are unrelated.
      - -1.0: Perfect opposition. Vectors point in opposite directions.

    Args:
        query_embedding (torch.Tensor): The embedding for the query item.
        reference_embedding (torch.Tensor): The embedding for the reference item.

    Returns:
        torch.Tensor: A scalar tensor representing the cosine similarity,
                      with a value in the range [-1, 1].
    """
    if distance_metric != "cosine":
        raise NotImplementedError("This soft proxy is optimized for cosine similarity.")
    
    ref_idx = -1
    if reference_embedding is None:
        
        positive_indices = [
            idx for idx, label in enumerate(db_labels) if label == query_label
        ]
        
        if not positive_indices:
            print(f"Warning: No positive samples found for label {query_label}. Returning zero score.")
            return query_embedding.sum() * 0, None, -1

        exclusion_mask = create_exclusion_mask(
            query_filename=query_filename,
            query_video_id=query_video_id,
            db_filenames=db_filenames,
            db_video_ids=db_video_ids,
            device=query_embedding.device,
            exclude_self=exclude_self,
            cross_encounter=cross_encounter
        )
            
        # Filter the positive indices to find valid ones
        valid_positive_indices = [
            idx for idx in positive_indices if not exclusion_mask[idx].item()
        ]

        if not valid_positive_indices:
            print(f"Warning: No valid positive samples found for {query_label} ({query_filename}) after applying filter '{filter_mode}'. Returning zero score.")
            return query_embedding.sum() * 0, None, -1

        ref_idx = valid_positive_indices[0]
        reference_embedding = db_embeddings[ref_idx].unsqueeze(0).to(query_embedding.device)

    assert reference_embedding.ndim == 2 and reference_embedding.shape[0] == 1, \
        "reference_embedding should be of shape [1, embedding_dim]"
    query_embedding_norm = identity_rule_implicit(lambda t: F.normalize(t, p=2, dim=1), query_embedding)
    reference_embedding_norm = identity_rule_implicit(lambda t: F.normalize(t, p=2, dim=1), reference_embedding)
    # The similarity score is the dot product of the normalized vectors
    similarity_score = (query_embedding_norm * reference_embedding_norm).sum()
    return similarity_score, reference_embedding_norm, ref_idx


