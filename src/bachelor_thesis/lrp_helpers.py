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
            
def compute_similarity_proto_margin_pass(
    conv_gamma: float,
    lin_gamma: float,
    model_wrapper: torch.nn.Module,
    input_tensor: torch.Tensor,
    db_embeddings: torch.Tensor,    # (N, D)
    db_labels: list,                # len N
    db_filenames: list,             # len N
    query_label: str,
    query_filename: str = None,
    distance_metric: str = "cosine",
    temp: float = 0.05,
    topk_neg: int = 50,
    verbose: bool = False
) -> torch.Tensor:
    #TODO: the effectiveness of this has not been tested
    #Prototype-Margin (Geometric Explanation): This method explains the model's decision based on the query's geometric position relative to class-conditional 
    # "centers of mass" (the prototypes) in the embedding space. It's a very model-centric view.It's like explaining a simplified, implicit contrastive or triplet loss head.
    """
    Prototype-Margin LRP: proto_pos = softmax over friend sims, proto_neg = softmax over topk foe sims.
    score = sim(query, proto_pos) - sim(query, proto_neg)
    """
    input_tensor.grad = None
    zennit_comp = LayerMapComposite([
        (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
        (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
    ])
    try:
        zennit_comp.register(model_wrapper)

        query_emb = model_wrapper(input_tensor.requires_grad_())
        score = compute_knn_proto_margin(
            query_emb=query_emb,
            db_embeddings=db_embeddings,
            db_labels=db_labels,
            db_filenames=db_filenames,
            query_label=query_label,
            query_filename=query_filename,
            distance_metric=distance_metric,
            temp=temp,
            topk_neg=topk_neg,
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

    return relevance



def compute_similarity_lrp_pass(
    conv_gamma: float, 
    lin_gamma: float, 
    model_wrapper: nn.Module, 
    input_tensor: torch.Tensor,
    query_label: str,
    query_filename: str,
    db_embeddings: torch.Tensor,
    db_labels: list,
    db_filenames: list,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Computes LRP by explaining the cosine similarity between the output embedding
    and a reference embedding of the same identity. For simple comparisons, the
    reference embedding and its index are being returned
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

        query_embedding = model_wrapper(input_tensor.requires_grad_())

        similarity_score, reference_embedding, ref_idx = compute_similarity_score(query_embedding=query_embedding, query_label=query_label, query_filename=query_filename, db_embeddings=db_embeddings, db_labels=db_labels, db_filenames=db_filenames,)

        if verbose:
            print(f"Explaining similarity score ({similarity_score.item():.4f}) for Gammas (Conv: {conv_gamma}, Lin: {lin_gamma})")

        similarity_score.backward()

        relevance = (input_tensor * input_tensor.grad).sum(dim=1, keepdim=True)

    finally:
        zennit_comp.remove()

    return relevance, reference_embedding, ref_idx


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
    distance_metric: str = "cosine",
    proxy_temp: float = 0.1,
    verbose: bool = False
) -> torch.Tensor:    
    # kNN-Proxy (Retrieval Explanation): This method is more faithful to the downstream task. Since Re-ID is ultimately used 
    # for retrieval (finding nearest neighbors), this method directly explains the outcome of that retrieval process. 
    # It's a very task-centric view. It's like explaining the retrieval and voting mechanism of a soft k-NN classifier.
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
    # --- Mode and mode-specific parameters ---
    mode: str,
    distance_metrics: List[str] = ["cosine"],
    proxy_temp_values: List[float] = [0.1],
    topk_neg_values: List[int] = [50],
    # --- Control flags ---
    verbose: bool = False,
    # --- Database arguments for relevant modes ---
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
        distance_metrics: List[str] = ["cosine"],
        proxy_temp_values: List[float] = [0.1],
        topk_neg_values: List[int] = [50],
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

    # Create all combinations of parameters to iterate over.
    # This works even if lists have only one element.
    base_params = list(itertools.product(conv_gamma_values, lin_gamma_values))
    if mode == "soft_knn_margin":
        mode_specific_params = list(itertools.product(distance_metrics, proxy_temp_values))
        param_combinations = list(itertools.product(base_params, mode_specific_params))
    elif mode == "proto_margin":
        mode_specific_params = list(itertools.product(distance_metrics, proxy_temp_values, topk_neg_values))
        param_combinations = list(itertools.product(base_params, mode_specific_params))
    else: # similarity or other future modes
        param_combinations = base_params
    
    print(f"Total parameter combinations to process: {len(param_combinations)}")
    print("Patching model for LRP...")

    with DINOPatcher(model_wrapper):
        for params in param_combinations:
            if mode=="soft_knn_margin":
                (conv_gamma, lin_gamma), (distance_metric, proxy_temp) = params
                topk_neg = None
            elif mode=="proto_margin":
                (conv_gamma, lin_gamma), (distance_metric, proxy_temp, topk_neg) = params
            else: # similarity
                conv_gamma, lin_gamma = params
                distance_metric, proxy_temp, topk_neg = None, None, None 

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
                    extra_info = {} # To store mode-specific data like reference_embedding

                    if mode == "soft_knn_margin":
                        relevance_single = compute_knn_attnlrp_pass(
                            model_wrapper=model_wrapper, input_tensor=input_tensor_single,
                            query_label=label_single, query_filename=filename,
                            db_embeddings=db_embeddings, db_labels=db_labels, db_filenames=db_filenames,
                            conv_gamma=conv_gamma, lin_gamma=lin_gamma,
                            distance_metric=distance_metric, proxy_temp=proxy_temp, verbose=verbose
                        )
                    elif mode == "proto_margin":
                        relevance_single = compute_similarity_proto_margin_pass(
                             model_wrapper=model_wrapper, input_tensor=input_tensor_single,
                             query_label=label_single, query_filename=filename,
                             db_embeddings=db_embeddings, db_labels=db_labels, db_filenames=db_filenames,
                             conv_gamma=conv_gamma, lin_gamma=lin_gamma,
                             distance_metric=distance_metric, temp=proxy_temp, topk_neg=topk_neg, verbose=verbose
                        )
                    elif mode == "similarity":
                        relevance_single, reference_embedding, ref_idx = compute_similarity_lrp_pass(
                            model_wrapper=model_wrapper, input_tensor=input_tensor_single,
                            query_label=label_single, query_filename=filename,
                            db_embeddings=db_embeddings, db_labels=db_labels, db_filenames=db_filenames,
                            conv_gamma=conv_gamma, lin_gamma=lin_gamma, verbose=verbose
                        )
                        # Save the reference embedding used, for evaluation
                        extra_info["reference_embedding"] = reference_embedding.cpu()
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
                            "topk_neg": topk_neg
                        },
                        "mode": mode,
                        "relevance": relevance_single.detach().cpu(),
                        "mask": mask_single,
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
    print(f"Relevance db not found. Computing relevances...")

    conv_gamma = kwargs.pop('conv_gamma')
    lin_gamma = kwargs.pop('lin_gamma')
    distance_metric = kwargs.pop('distance_metric', 'cosine')
    proxy_temp = kwargs.pop('proxy_temp')
    topk_neg = kwargs.pop('topk_neg')

    # Convert single values to the list format required by generate_relevances
    kwargs['conv_gamma_values'] = [conv_gamma]
    kwargs['lin_gamma_values'] = [lin_gamma]
    kwargs['distance_metrics'] = [distance_metric]
    kwargs['proxy_temp_values'] = [proxy_temp]
    kwargs['topk_neg_values'] = [topk_neg]

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

#TODO also mask out same video for all three scores
def compute_knn_proto_margin(
    query_emb: torch.Tensor,
    query_label: str,
    query_filename: str,
    db_embeddings: torch.Tensor,
    db_labels: list,
    db_filenames: list,
    distance_metric: str = "cosine",
    temp: float = 0.05,
    topk_neg: int = 50,
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
        query_emb: The embedding for the query item.
        query_label: The label of the query item.
        query_filename: The filename of the query, for self-exclusion.
        db_embeddings: A tensor of all embeddings in the database.
        db_labels: A list of all labels in the database.
        db_filenames: A list of all filenames, for self-exclusion.
        distance_metric: The metric used; only 'cosine' is supported.
        temp: Temperature for softmax weighting of prototypes.
        topk_neg: The number of hard negatives to average for the prototype.
        exclude_self: If True, the query is excluded from its own neighborhood.

    Returns:
        torch.Tensor: A scalar tensor representing the prototype-based margin score,
                      with a value in the range [-2, 2].
    """
    if distance_metric != "cosine":
        raise NotImplementedError("This proto margin is optimized for cosine similarity.")

    if query_emb.dim() == 1:
        query_emb = query_emb.view(1, -1)

    qn = identity_rule_implicit(lambda t: F.normalize(t, p=2, dim=1), query_emb)
    dbn = identity_rule_implicit(lambda t: F.normalize(t, p=2, dim=1), db_embeddings)

    sims = F.linear(dbn, qn).squeeze(1)  # (N,)

    if exclude_self and (query_filename in db_filenames):
        try:
            qidx = db_filenames.index(query_filename)
            sims[qidx] = -1e9
        except ValueError:
            pass

    # masks
    device = sims.device
    labels_tensor = torch.tensor([1 if l == query_label else 0 for l in db_labels], device=device)

    # proto_pos (if no friends found, fallback to nearest same-file or zero)
    pos_idx = torch.nonzero(labels_tensor).squeeze(1) if labels_tensor.sum() > 0 else torch.tensor([], device=device, dtype=torch.long)
    if pos_idx.numel() == 0:
        # fallback: take the single best-matching embedding with same filename if any, else top1 overall
        proto_pos = dbn[sims.argmax()].unsqueeze(0)
    else:
        sims_pos = sims[pos_idx]
        alpha_pos = F.softmax(sims_pos / temp, dim=0)
        proto_pos = (alpha_pos.unsqueeze(1) * dbn[pos_idx]).sum(dim=0, keepdim=True)  # (1, D)

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

def compute_knn_proxy_soft(
    query_embedding: torch.Tensor,
    query_label: str,
    query_filename: str,
    db_embeddings: torch.Tensor,
    db_labels: list,
    db_filenames: list,
    distance_metric: str = "cosine",
    temp: float = 0.05,
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
    #TODO: gleiche videos genauso wie sich selbst auch wegmaskieren?
    # Achtibat:
    q_norm = identity_rule_implicit(lambda t: F.normalize(t, p=2, dim=1), q_emb)
    db_norm = identity_rule_implicit(lambda t: F.normalize(t, p=2, dim=1), db_embeddings)


    # Calculate cosine similarity (higher is better)
    # Note: F.linear(db_norm, q_norm) is equivalent to db_norm @ q_norm.T
    similarities = F.linear(db_norm, q_norm).squeeze(1) # Shape: (N,)

    # Exclude the query from its own neighbors if it exists in the database
    if exclude_self and query_filename in db_filenames:
        try:
            query_idx = db_filenames.index(query_filename)
            # Set similarity to a very low number to give it near-zero weight after softmax
            similarities[query_idx] = -1e9
        except ValueError:
            pass # Query not found, nothing to do

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
    score_prob = prob_friends 
    score_margin = prob_friends - prob_foes

    return score_margin

def compute_similarity_score(
        query_embedding: torch.Tensor, 
        query_label: str, query_filename: str, 
        db_embeddings: torch.Tensor, 
        db_labels: list, 
        db_filenames: list,
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
    ref_idx = -1
    if reference_embedding is None:
        label_to_indices = {} # for similarity score
        for idx, label in enumerate(db_labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        # Find a positive reference embedding from the database
        positive_indices = [idx for idx in label_to_indices.get(query_label, []) if db_filenames[idx] != query_filename]
        if not positive_indices:
            print(f"Warning: No other positive samples found for {query_label} ({query_filename}). Skipping.")
            return 0, None, -1

        ref_idx = positive_indices[0] 
        device = query_embedding.device
        reference_embedding = db_embeddings[ref_idx].unsqueeze(0).to(device)

    assert reference_embedding.ndim == 2 and reference_embedding.shape[0] == 1, \
        "reference_embedding should be of shape [1, embedding_dim]"
    query_embedding_norm = identity_rule_implicit(lambda t: F.normalize(t, p=2, dim=1), query_embedding)
    reference_embedding_norm = identity_rule_implicit(lambda t: F.normalize(t, p=2, dim=1), reference_embedding)
    # The similarity score is the dot product of the normalized vectors
    similarity_score = (query_embedding_norm * reference_embedding_norm).sum()
    return similarity_score, reference_embedding_norm, ref_idx