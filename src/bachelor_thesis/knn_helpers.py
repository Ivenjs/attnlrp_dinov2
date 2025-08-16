from basemodel import TimmWrapper
import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from typing import Tuple
from torch.utils.data import DataLoader
from lxt.efficient.rules import identity_rule_implicit
import torch

import numpy as np

from dataset import GorillaReIDDataset, custom_collate_fn

def fill_knn_db(
    dataset: GorillaReIDDataset, 
    model_wrapper: TimmWrapper, 
    output_path: str,
    device: torch.device, 
    batch_size: int = 64, 
) -> Tuple[torch.Tensor, list]:
    """
    Generates and saves embeddings for a given dataset.
    Saves embeddings, labels, and filenames.
    """
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    print(f"Generating embeddings for {len(dataloader.dataset)} images...")
        
    all_embeddings_list = []
    all_labels = []
    all_filenames = []
    all_videos = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Generating embeddings for {os.path.basename(output_path)}"):
            images = batch["image"]
            labels = batch["label"]
            filenames = batch["filename"]
            videos = batch["video"]

            images = images.to(device)
            embeddings = model_wrapper(images)
            
            all_embeddings_list.append(embeddings.cpu())
            all_labels.extend(labels)
            all_filenames.extend(filenames)
            all_videos.extend(videos)

    final_embeddings = torch.cat(all_embeddings_list, dim=0)

    db_data = {
        "embeddings": final_embeddings,
        "labels": all_labels,
        "filenames": all_filenames,
        "videos": all_videos
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(db_data, output_path)
    
    print(f"Saved {len(all_filenames)} embeddings to {output_path}")

    return final_embeddings.to(device), all_labels, all_filenames, all_videos

def get_knn_db(
    db_path: str,
    dataset: GorillaReIDDataset,
    model_wrapper: TimmWrapper, 
    batch_size: int,
    device: torch.device,
    recompute: bool=False
) -> Tuple[torch.Tensor, list, list]:
    """
    Loads a pre-computed k-NN database or creates it if it doesn't exist.
    The database name is a combination of the model dataset_name and the data split.
    """
    if os.path.exists(db_path):
        print(f"Loading existing k-NN database: {db_path}")
        db_data = torch.load(db_path, map_location=device, weights_only=False)
        return db_data["embeddings"], db_data["labels"], db_data["filenames"], db_data["videos"]
    else:
        print(f"k-NN database not found. Creating new one at: {db_path}")
        return fill_knn_db(
            dataset=dataset,
            model_wrapper=model_wrapper,
            output_path=db_path,
            device=device,
            batch_size=batch_size
        )


def compute_distances(
    query_embedding: torch.Tensor,
    db_embeddings: torch.Tensor,
    metric: str = "cosine"
) -> torch.Tensor:
    """
    Computes distances between a query embedding and a database of embeddings.

    This function follows best practices by L2-normalizing features for both
    cosine and euclidean distances to ensure a fair comparison and robustness.

    Args:
        query_embedding (torch.Tensor): A single query embedding of shape (1, D).
        db_embeddings (torch.Tensor): Database embeddings of shape (N, D).
        metric (str): The distance metric, 'cosine' or 'euclidean'.

    Returns:
        torch.Tensor: A tensor of distances of shape (N,). Smaller is better.
    """

    # Ensure query_embedding is shape (1, D) for broadcasting
    if query_embedding.dim() == 1:
        query_embedding = query_embedding.unsqueeze(0)

    if metric == "cosine":
        # L2-normalize features
        norm_query = F.normalize(query_embedding, p=2, dim=1)
        norm_db = F.normalize(db_embeddings, p=2, dim=1)
        
        # Calculate cosine similarity
        cosine_sim = F.linear(norm_query, norm_db).squeeze()
        
        # Convert similarity to distance (0=identical, 2=opposite)
        distances = 1 - cosine_sim
        return distances
    
    elif metric == "euclidean":
        # For a fair comparison, also use normalized features for Euclidean.
        norm_query = F.normalize(query_embedding, p=2, dim=1)
        norm_db = F.normalize(db_embeddings, p=2, dim=1)
        
        # Calculate L2 distance. torch.cdist is highly efficient.
        distances = torch.cdist(norm_query, norm_db, p=2.0).squeeze()
        return distances
        
    else:
        raise ValueError(f"Unknown metric: {metric}. Choose 'cosine' or 'euclidean'.")

def compute_knn_proto_margin(
    query_emb: torch.Tensor,
    db_embeddings: torch.Tensor,
    db_labels: list,
    db_filenames: list,
    query_label: str,
    query_filename: str = None,
    temp: float = 0.05,
    topk_neg: int = 50,
    exclude_self: bool = True
):
    if query_emb.dim() == 1:
        query_emb = query_emb.view(1, -1)

    qn = identity_rule_implicit(lambda t: F.normalize(t, p=2, dim=1), query_emb)
    dbn = identity_rule_implicit(lambda t: F.normalize(t, p=2, dim=1), db_embeddings)

    sims = F.linear(dbn, qn).squeeze(1)  # (N,)

    if exclude_self and (query_filename is not None) and (query_filename in db_filenames):
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
    """
    Computes a differentiable, contrastive proxy score for k-NN based on
    softmax-weighted similarities. This avoids the non-differentiable top-k
    and creates a score that balances friends vs. foes.
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
    score_prob = prob_friends #TODO try this
    score_margin = prob_friends - prob_foes

    return score_margin
    #return score_prob

def get_query_performance_metrics(
    query_embedding: torch.Tensor,
    query_label: str,
    query_filename: str,
    db_embeddings: torch.Tensor,
    db_labels: list[str],
    db_filenames: list[str],
    distance_metric: str = "cosine",
    k: int = 5 
) -> dict:
    """
    Computes multiple performance metrics for a single query against a database.
    """
    device = query_embedding.device
    db_labels_np = np.array(db_labels)
    db_filenames_np = np.array(db_filenames)

    with torch.no_grad():
        distances = compute_distances(query_embedding, db_embeddings, distance_metric)
        try:
            query_idx = db_filenames.index(query_filename)
            distances[query_idx] = torch.inf
        except ValueError:
            pass

        sorted_indices = torch.argsort(distances)
        sorted_labels = db_labels_np[sorted_indices.cpu().numpy()]

    match_indices = np.where(sorted_labels == query_label)[0]
    rank = match_indices[0] + 1 if len(match_indices) > 0 else -1

    gt_positive_mask = (db_labels_np == query_label) & (db_filenames_np != query_filename)
    gt_positive_indices = np.where(gt_positive_mask)[0]
    
    gt_similarity = -1.0
    if gt_positive_indices.size > 0:
        min_gt_distance = torch.min(distances[gt_positive_indices])
        gt_similarity = 1.0 - min_gt_distance.item()

    recall_at_k = 0.0
    num_gt_positives = gt_positive_indices.size
    if num_gt_positives > 0:
        # Get the indices of the top-k results
        top_k_indices = sorted_indices[:k].cpu().numpy()
        
        # Count how many of the top-k results are ground-truth positives
        # Using sets is efficient for this intersection
        top_k_set = set(top_k_indices)
        gt_positives_set = set(gt_positive_indices)
        
        num_correct_in_top_k = len(top_k_set.intersection(gt_positives_set))
        
        recall_at_k = num_correct_in_top_k / num_gt_positives

    return {
        'rank': rank,
        'gt_similarity': gt_similarity,
        'recall_at_k': recall_at_k
    }
