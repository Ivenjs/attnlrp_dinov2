from basemodel import TimmWrapper
from torchvision import transforms
import os
from PIL import Image
import torch
from tqdm import tqdm
import torch.nn.functional as F
from typing import Tuple
from torch.utils.data import DataLoader

import numpy as np

import yaml
from utils import get_class_label
from dataset import GorillaReIDDataset, custom_collate_fn


#TODO: rather use the transforms from the model wrapper
TRANSFORM = transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

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
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Generating embeddings for {os.path.basename(output_path)}"):
            images = batch["image"]
            labels = batch["label"]
            filenames = batch["filename"]

            images = images.to(device)
            embeddings = model_wrapper(images)
            
            all_embeddings_list.append(embeddings.cpu())
            all_labels.extend(labels)
            all_filenames.extend(filenames)

    final_embeddings = torch.cat(all_embeddings_list, dim=0)

    db_data = {
        "embeddings": final_embeddings,
        "labels": all_labels,
        "filenames": all_filenames
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(db_data, output_path)
    
    print(f"Saved {len(all_filenames)} embeddings to {output_path}")

    return final_embeddings.to(device), all_labels, all_filenames

def get_knn_db(
    db_dir: str,
    split_name: str, # e.g., "train" or "val"
    dataset: GorillaReIDDataset,
    model_wrapper: TimmWrapper, 
    model_checkpoint_path: str,
    batch_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, list, list]:
    """
    Loads a pre-computed k-NN database or creates it if it doesn't exist.
    The database name is a combination of the model dataset_name and the data split.
    """

    checkpoint_name = os.path.splitext(os.path.basename(model_checkpoint_path))[0]
    db_filename = f"{checkpoint_name}_{dataset.dataset_name}_{split_name}_db.pt"
    db_path = os.path.join(db_dir, db_filename)

    db_embeddings = []
    db_filenames = []

    if os.path.exists(db_path):
        print(f"Loading existing k-NN database: {db_path}")
        db_data = torch.load(db_path, map_location=device)
        return db_data["embeddings"], db_data["labels"], db_data["filenames"]
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

def compute_distances_batched(
    query_embeddings_batch: torch.Tensor,
    db_embeddings: torch.Tensor,
    metric: str = "cosine"
) -> torch.Tensor:
    """
    Computes a matrix of distances between a BATCH of query embeddings and a 
    database of embeddings in a fully vectorized way.

    This function follows best practices by L2-normalizing features for both
    cosine and euclidean distances to ensure a fair comparison and robustness.

    Args:
        query_embeddings_batch (torch.Tensor): A batch of query embeddings of shape (B, D).
        db_embeddings (torch.Tensor): Database embeddings of shape (N, D).
        metric (str): The distance metric, 'cosine' or 'euclidean'.

    Returns:
        torch.Tensor: A tensor of distances of shape (B, N). smaller is better.
                      dist_matrix[i, j] is the distance between the i-th query
                      and the j-th database item.
    """
    # The input check for a 1D tensor is removed, as we now expect a (B, D) batch.
    
    if metric == "cosine":
        # L2-normalize both batches of features along the feature dimension (dim=1).
        # This operation is naturally batched.
        norm_query_batch = F.normalize(query_embeddings_batch, p=2, dim=1) # Shape: (B, D)
        norm_db = F.normalize(db_embeddings, p=2, dim=1)                   # Shape: (N, D)
        
        # Calculate cosine similarity for the whole batch via matrix multiplication.
        # (B, D) @ (D, N) -> (B, N)
        cosine_sim = torch.matmul(norm_query_batch, norm_db.T)
        
        # Convert similarity to distance. The result is already the correct (B, N) shape.
        # No .squeeze() is needed.
        distances = 1 - cosine_sim
        return distances
    
    elif metric == "euclidean":
        # Also use normalized features for a fair comparison.
        norm_query_batch = F.normalize(query_embeddings_batch, p=2, dim=1) # Shape: (B, D)
        norm_db = F.normalize(db_embeddings, p=2, dim=1)                   # Shape: (N, D)
        
        distances = torch.cdist(norm_query_batch, norm_db, p=2.0)
        return distances
        
    else:
        raise ValueError(f"Unknown metric: {metric}. Choose 'cosine' or 'euclidean'.")

def knn(
    embedding: torch.Tensor,
    db_embeddings: torch.Tensor,
    db_labels: list,
    device: torch.device,
    k: int = 5,
    metric: str = "euclidean",
) -> str:
    distances = compute_distances(embedding, db_embeddings, metric=metric)

    top_k_indices = torch.topk(distances, k, largest=False).indices
    top_k_labels = [db_labels[i] for i in top_k_indices]
    label = max(set(top_k_labels), key=top_k_labels.count)
    return label


def compute_knn_proxy_score(
    query_embedding: torch.Tensor,
    query_label: str,         
    query_filename: str,      
    db_embeddings: torch.Tensor,
    db_labels: list,
    db_filenames: list,
    distance_metric: str = "cosine",
    k: int = 5,
    return_indices=False
) -> torch.Tensor:
    """
    Computes a differentiable proxy score for a k-NN classifier's decision.

    The score is defined as: S = - mean(sim_friends) as LRP wants to backpropagate from a value
    that should be maximized, so we use the negative distance to friends (ecause we want to minimize dist_friends).
    """

    
    num_db_samples = len(db_filenames)
    
    is_query_in_db = query_filename in db_filenames
    
    num_available_neighbors = num_db_samples - 1 if is_query_in_db else num_db_samples
    
    # Ensure k is not larger than the number of available neighbors.
    # If there are no available neighbors, effective_k will be 0.
    effective_k = min(k, num_available_neighbors)

    # If there are no possible neighbors to find, we can't compute a score.
    # Return a neutral score (0) or handle as an error. NORMALLY, THIS SHOULD NOT HAPPEN.
    if effective_k <= 0:
        # Returning a neutral score is often a safe default.
        logging.warning(f"No available neighbors for query '{query_filename}'. Returning neutral score. THIS IS VERY UNUSUAL!")
        return torch.tensor(0.0, device=query_embedding.device, requires_grad=True)
    # We must use a detached version of the query for the distance calculation
    # to find the neighbors. This is because topk is not nicely differentiable
    # and we only need the *identities* of the neighbors, not their gradient path.
    with torch.no_grad():
        distances = compute_distances(query_embedding.detach(), db_embeddings, distance_metric)
        if is_query_in_db:
            try:
                query_idx = db_filenames.index(query_filename)
                distances[query_idx] = torch.inf
            except ValueError:
                pass
        top_k_indices = torch.topk(distances, effective_k, largest=False).indices
    
    friends_indices = []
    foes_indices = []
    for idx_tensor in top_k_indices:
        idx = idx_tensor.item() 
        if db_labels[idx] == query_label:
            friends_indices.append(idx)
        else:
            foes_indices.append(idx)

    differentiable_distances = compute_distances(query_embedding, db_embeddings, distance_metric)

    if friends_indices:
        # We want to MINIMIZE dist_friends, which is equivalent to MAXIMIZING -dist_friends.
        # LRP explains what contributes to MAXIMIZING the output.
        score = -differentiable_distances[friends_indices].mean() 
    else:
        # If no friends, there's nothing to explain. We can return a zero map
        # or handle it gracefully. A neutral score of 0 is a safe bet.
        # Backpropagating from a constant gives zero gradients.
        score = torch.tensor(0.0, device=query_embedding.device, requires_grad=True)
        
    if return_indices:
        return score, friends_indices, foes_indices
    else:
        return score

def compute_knn_proxy_score_batched(
    query_embedding_batch: torch.Tensor, # Shape: [B, D]
    query_labels_batch: list[str],       # List of B strings
    query_filenames_batch: list[str],    # List of B strings
    db_embeddings: torch.Tensor,         # Shape: [N, D]
    db_labels: list[str],                # List of N strings
    db_filenames: list[str],             # List of N strings
    distance_metric: str = "cosine",
    k: int = 5,
) -> torch.Tensor:                       # Returns Tensor of B scores
    """
    Computes a differentiable proxy score for k-NN decisions for a BATCH of queries.
    
    The score is defined as: S = mean(dist_foes) - mean(dist_friends).
    This function is fully vectorized and avoids Python loops over the batch dimension.
    """
    device = query_embedding_batch.device
    batch_size = query_embedding_batch.shape[0]
    num_db_samples = len(db_filenames)

    # --- Step 1: Pre-computation and Label Conversion ---
    # Convert string labels to integer IDs for efficient tensor comparisons.
    all_labels = sorted(list(set(db_labels + query_labels_batch)))
    label_to_id = {label: i for i, label in enumerate(all_labels)}
    
    db_labels_ids = torch.tensor([label_to_id[lbl] for lbl in db_labels], device=device)       # Shape: [N]
    query_labels_ids = torch.tensor([label_to_id[lbl] for lbl in query_labels_batch], device=device) # Shape: [B]

    # Create a mapping from DB filename to its index for self-exclusion.
    db_filename_to_idx = {fname: i for i, fname in enumerate(db_filenames)}

    # --- Step 2: Find Neighbor Indices (Non-differentiable part) ---
    with torch.no_grad():
        # Compute distances for the entire batch to the entire DB.
        distances = compute_distances_batched(
            query_embedding_batch.detach(), db_embeddings, distance_metric
        ) # Shape: [B, N]

        # Exclude self-matches. For each query, if its filename is in the DB,
        # set its distance to itself to infinity so it's not picked as a neighbor.
        for i, fname in enumerate(query_filenames_batch):
            if fname in db_filename_to_idx:
                db_idx = db_filename_to_idx[fname]
                distances[i, db_idx] = torch.inf

        # Ensure k is not larger than the number of available neighbors.
        # This check is now simpler; it assumes at least one non-self neighbor exists.
        effective_k = min(k, num_db_samples - 1)
        if effective_k <= 0:
            logging.warning("No available neighbors in the database to compare against.")
            return torch.zeros(batch_size, device=device)

        # Get the indices of the top k nearest neighbors for EACH query in the batch.
        # topk on dim=1 works row-wise, which is exactly what we need.
        top_k_indices = torch.topk(distances, effective_k, largest=False, dim=1).indices # Shape: [B, k]

    # --- Step 3: Categorize Neighbors (Vectorized) ---
    # Use the integer IDs to find friends and foes for the whole batch at once.
    # 1. Get the labels of the top k neighbors for each query.
    neighbor_labels_ids = db_labels_ids[top_k_indices] # Shape: [B, k]
    # 2. Compare with the query labels. `unsqueeze(1)` enables broadcasting.
    # query_labels_ids shape [B] -> [B, 1]
    # neighbor_labels_ids shape [B, k]
    # Resulting shape is [B, k]
    is_friend_mask = (neighbor_labels_ids == query_labels_ids.unsqueeze(1))
    is_foe_mask = ~is_friend_mask
    
    # --- Step 4: Compute Differentiable Distances and Scores (Vectorized) ---
    # Re-compute distances with the original, gradient-enabled query embeddings.
    differentiable_distances = compute_distances_batched(
        query_embedding_batch, db_embeddings, distance_metric
    ) # Shape: [B, N]

    # Use `torch.gather` to select the differentiable distances of the top k neighbors.
    # This is a key step to connect the non-differentiable indices to differentiable values.
    top_k_dists = torch.gather(differentiable_distances, 1, top_k_indices) # Shape: [B, k]
    
    # --- Step 5: Calculate Mean Friend/Foe Distances Safely ---
    MAX_DISTANCE = 2.0 # A fallback value for distance.
    
    # Calculate sum of distances for friends and foes by zeroing out the non-relevant ones.
    dist_friends_sum = (top_k_dists * is_friend_mask).sum(dim=1) # Shape: [B]
    dist_foes_sum = (top_k_dists * is_foe_mask).sum(dim=1)     # Shape: [B]
    
    # Count the number of friends and foes for each query to compute the mean safely.
    num_friends = is_friend_mask.sum(dim=1) # Shape: [B]
    num_foes = is_foe_mask.sum(dim=1)       # Shape: [B]

    # Initialize mean distances with the max fallback value.
    mean_dist_friends = torch.full((batch_size,), MAX_DISTANCE, device=device, dtype=torch.float32)
    mean_dist_foes = torch.full((batch_size,), MAX_DISTANCE, device=device, dtype=torch.float32)

    # Create masks for queries that have at least one friend or foe.
    has_friends = num_friends > 0
    has_foes = num_foes > 0

    # Compute the mean only for those queries, avoiding division by zero.
    mean_dist_friends[has_friends] = dist_friends_sum[has_friends] / num_friends[has_friends]
    mean_dist_foes[has_foes] = dist_foes_sum[has_foes] / num_foes[has_foes]
    
    # The final score for the entire batch.
    scores = mean_dist_foes - mean_dist_friends # Shape: [B]

    return scores

def compute_evaluation_score(
    current_embedding: torch.Tensor,
    db_embeddings: torch.Tensor,
    friends_indices: list,
    distance_metric: str,
) -> torch.Tensor:
    """
    Computes an intuitive, similarity-like score for evaluation purposes.
    This score is designed to be high when the query is close to its friends.

    - For cosine distance, it converts the [0, 2] distance back to a [-1, 1] similarity.
    - For Euclidean distance, it maps the [0, 2] distance to a [1, 0] similarity.

    Args:
        current_embedding (torch.Tensor): The embedding of the (perturbed) query image.
        db_embeddings (torch.Tensor): The database embeddings.
        friends_indices (list): The list of indices for the *fixed* set of friends.
        distance_metric (str): The metric used, 'cosine' or 'euclidean'.

    Returns:
        torch.Tensor: A single scalar score. Larger is better.
    """
    if not friends_indices:
        # If there are no friends, the score is undefined. Return 0.
        return torch.tensor(0.0, device=current_embedding.device)

    # We don't need gradients here, but using the main distance function is fine.
    all_distances = compute_distances(current_embedding, db_embeddings, distance_metric)
    dist_to_friends = all_distances[friends_indices]

    if distance_metric == "cosine":
        # Cosine distance = 1 - similarity.
        # So, similarity = 1 - distance.
        # This will be in the range [-1, 1].
        similarity_score = 1 - dist_to_friends.mean()
    elif distance_metric == "euclidean":
        # Euclidean distance on the unit hypersphere is in [0, 2].
        # We can map this to a [1, 0] similarity range to make it intuitive.
        # score = 1 - (dist / 2).
        # When dist=0, score=1. When dist=2, score=0.
        similarity_score = 1 - (dist_to_friends.mean() / 2.0)
    else:
        raise ValueError(f"Unknown metric for evaluation: {distance_metric}")

    return similarity_score

def get_query_performance_metrics(
    query_embedding: torch.Tensor,
    query_label: str,
    query_filename: str,
    db_embeddings: torch.Tensor,
    db_labels: list[str],
    db_filenames: list[str],
    distance_metric: str = "cosine"
) -> dict:
    """
    Computes multiple performance metrics for a single query against a database.

    Args:
        query_embedding (torch.Tensor): The (1, D) embedding of the query image.
        query_label (str): The ground-truth label of the query.
        ... and other db parameters

    Returns:
        dict: A dictionary containing 'rank', 'gt_similarity', and 'proxy_score'.
    """
    device = query_embedding.device
    db_labels_np = np.array(db_labels)
    db_filenames_np = np.array(db_filenames)

    # --- 1. Compute Distances (non-differentiable) ---
    with torch.no_grad():
        distances = compute_distances(query_embedding, db_embeddings, distance_metric)
        # Exclude self-match from the ranking if the query is in the DB
        try:
            query_idx = db_filenames.index(query_filename)
            distances[query_idx] = torch.inf
        except ValueError:
            pass # Query is not in the DB, no need to exclude anything

        sorted_indices = torch.argsort(distances)
        sorted_labels = db_labels_np[sorted_indices.cpu().numpy()]

    # --- 2. Calculate Rank ---
    # Find the first occurrence of the correct label in the sorted list
    match_indices = np.where(sorted_labels == query_label)[0]
    rank = match_indices[0] + 1 if len(match_indices) > 0 else -1 # Use -1 or float('inf') for no match

    # --- 3. Calculate Ground-Truth Similarity ---
    # Find all embeddings in the DB that are from the same individual (ground-truth positives)
    gt_positive_mask = (db_labels_np == query_label) & (db_filenames_np != query_filename)
    gt_positive_indices = np.where(gt_positive_mask)[0]

    gt_similarity = -1.0 # Default if no other images of the same individual exist
    if gt_positive_indices.size > 0:
        # Get distances to all ground-truth positives
        gt_distances = distances[gt_positive_indices]
        # Find the minimum distance (i.e., most similar)
        min_gt_distance = torch.min(gt_distances)
        # Convert distance back to similarity (1 - distance)
        gt_similarity = 1.0 - min_gt_distance.item()


    return {
        'rank': rank,
        'gt_similarity': gt_similarity
    }

