from basemodel import TimmWrapper
from torchvision import transforms
import os
from PIL import Image
import torch
from tqdm import tqdm
import torch.nn.functional as F
from typing import Tuple
from torch.utils.data import DataLoader


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
    db_filename = f"{checkpoint_name}_{dataset.get_dataset_name()}_{split_name}_db.pt"
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
) -> torch.Tensor:
    """
    Computes a differentiable proxy score for a k-NN classifier's decision.

    The score is defined as: S = mean(sim_friends) - mean(sim_foes).
    The score [-2,2] is larger when the query is more similar to its friends than to its foes.
    This creates a differentiable objective that LRP can backpropagate through.
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

    """print(f"Query: {query_filename}, Friends: {len(friends_indices)}, Foes: {len(foes_indices)}")
    for friends in friends_indices:
        print(f"Found friend: {db_filenames[friends]} with label {get_class_label(db_filenames[friends])}")
    for foes in foes_indices:
        print(f"Found foe: {db_filenames[foes]} with label {get_class_label(db_filenames[foes])}")"""

    differentiable_distances = compute_distances(query_embedding, db_embeddings, distance_metric)

    MAX_DISTANCE = 2.0 # both euclidean and cosine distances are in [0, 2]

    if friends_indices:
        dist_friends = differentiable_distances[friends_indices].mean()
    else:
        # this case is a bit weird, since we only have foes and the score will be arbitrarily better, when the foes are closer...
        dist_friends = torch.tensor(MAX_DISTANCE, device=query_embedding.device)

    if foes_indices:
        dist_foes = differentiable_distances[foes_indices].mean()
    else:
        dist_foes = torch.tensor(MAX_DISTANCE, device=query_embedding.device)

    score = dist_foes - dist_friends
    return score

