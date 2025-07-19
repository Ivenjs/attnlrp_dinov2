from basemodel import TimmWrapper
from torchvision import transforms
import os
from PIL import Image
import torch
from tqdm import tqdm
import torch.nn.functional as F
from typing import Tuple


import yaml

from basemodel import get_model_wrapper
from utils import get_class_label


#TODO: rather use the transforms from the model wrapper
TRANSFORM = transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

def fill_knn_db(
    image_dir: str, model_wrapper: TimmWrapper, output_dir: str, model_checkpoint: str, device: torch.device, batch_size: int = 64, transform: transforms.Compose = TRANSFORM
) -> Tuple[torch.Tensor, list]:
    """
    Generates and saves embeddings for all images in a directory using manual batch processing.
    """
    # 1. Get all image file paths and filenames first
    all_files = sorted(os.listdir(image_dir)) # sorted for consistent order
    image_filenames = [f for f in all_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not image_filenames:
        print(f"Warning: No images found in {image_dir}")
        return torch.empty(0), []
        
    all_embeddings_list: List[torch.Tensor] = []
    processed_filenames: List[str] = []
    
    image_batch: List[torch.Tensor] = []
    filename_batch: List[str] = []
    
    with torch.no_grad():
        for filename in tqdm(image_filenames, desc="Generating embeddings in batches"):
            image_path = os.path.join(image_dir, filename)
            
            image_tensor = transform(Image.open(image_path).convert("RGB"))
            
            image_batch.append(image_tensor)
            filename_batch.append(filename.split(".")[0])  

            # 2. When the batch is full, process it
            if len(image_batch) == batch_size:
                stacked_images = torch.stack(image_batch).to(device)
                
                embeddings = model_wrapper(stacked_images) # should return [B, D]
                
                all_embeddings_list.append(embeddings.cpu())
                processed_filenames.extend(filename_batch)
                
                image_batch = []
                filename_batch = []

        # Process any remaining images in the last batch
        if image_batch:
            stacked_images = torch.stack(image_batch).to(device)
            embeddings = model_wrapper(stacked_images)
            all_embeddings_list.append(embeddings.cpu())
            processed_filenames.extend(filename_batch)

    final_embeddings = torch.cat(all_embeddings_list, dim=0)

    dataset_to_save = {
        "embeddings": final_embeddings,  
        "filenames": processed_filenames
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/{model_checkpoint.split('/')[-1]}_{image_dir.split('/')[-1]}.pt"
    torch.save(dataset_to_save, out_path)
    
    print(f"Saved {len(processed_filenames)} embeddings to {out_path}")

    return final_embeddings.to(device), processed_filenames

def get_knn_db(knn_db_dir: str, image_dir: str, model_wrapper: TimmWrapper, transforms: transforms.Compose, device: torch.device) -> Tuple[torch.Tensor, list]:
    db_embeddings = []
    db_filenames = []

    model_config_path = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/configs/model.yaml"
    with open(model_config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    #TODO: also include data name here for comparison so that i can have multiple databases for different datasets
    checkpoint_name = os.path.basename(cfg["checkpoint_path"]).split('.')[0]
    dataset_name = image_dir.split('/')[-1]
    db_name = f"{checkpoint_name}_{dataset_name}"

    files_in_dir = os.listdir(knn_db_dir)
    matching_checkpoints = [f for f in files_in_dir if db_name in f]
    if matching_checkpoints:
        print(f"KNN database {db_name} already exists. Loading the KNN database...")
        dataset = torch.load(os.path.join(knn_db_dir, matching_checkpoints[0]))
        db_embeddings = dataset["embeddings"].to(device)
        db_filenames = dataset["filenames"]
    else:
        print(f"KNN database {db_name} does not exist. Filling the KNN database...")
        db_embeddings, db_filenames = fill_knn_db(
            image_dir=image_dir,
            model_wrapper=model_wrapper,
            output_dir=knn_db_dir,
            model_checkpoint=checkpoint_name,
            device=device,
            transform=transforms
        )
    return db_embeddings, db_filenames


# TODO use the GPU enhanced versions from the model_evaluation.py (gorillawatch repo)
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
    query_filename: str,
    db_embeddings: torch.Tensor,
    db_filenames: list,
    distance_metric: str = "cosine",
    k: int = 5,
) -> torch.Tensor:
    """
    Computes a differentiable proxy score for a k-NN classifier's decision.

    The score is defined as: S = mean(sim_friends) - mean(sim_foes)
    This creates a differentiable objective that LRP can backpropagate through.

    Args:
        query_embedding (torch.Tensor): The (1, D) embedding of the input image.
                                        This tensor MUST have requires_grad=True.
        query_filename (str): The filename of the input image, used to infer the ground truth label
        db_embeddings (torch.Tensor): The (N, D) embeddings in the k-NN database.
        db_filenames (list): A list of N filenames corresponding to the db_embeddings. The labels can be infered from the filenames.
        k (int): The number of nearest neighbors to consider.
        metric (str): The similarity metric to use, 'cosine' or 'euclidean'.

    Returns:
        torch.Tensor: A single scalar tensor representing the proxy score,
                      with its computation graph attached to the query_embedding.
    """
    # We must use a detached version of the query for the distance calculation
    # to find the neighbors. This is because topk is not nicely differentiable
    # and we only need the *identities* of the neighbors, not their gradient path.

    ground_truth_label = query_filename.split("_")[0]

    with torch.no_grad():
        distances = compute_distances(query_embedding.detach(), db_embeddings, distance_metric)
        try:
            query_idx = db_filenames.index(query_filename)
            distances[query_idx] = torch.inf
        except ValueError:
            # It's fine if the query isn't in the DB, just means no mask is needed.
            pass
        top_k_indices = torch.topk(distances, k, largest=False).indices
    
    friends_indices = []
    foes_indices = []
    for idx_tensor in top_k_indices:
        idx = idx_tensor.item() 
        neighbor_label = get_class_label(db_filenames[idx])
        if neighbor_label == ground_truth_label:
            friends_indices.append(idx)
        else:
            foes_indices.append(idx)

    differentiable_distances = compute_distances(query_embedding, db_embeddings, distance_metric)

    MAX_DISTANCE = 2.0 # both euclidean and cosine distances are in [0, 2]

    if friends_indices:
        dist_friends = differentiable_distances[friends_indices].mean()
    else:
        dist_friends = torch.tensor(MAX_DISTANCE, device=query_embedding.device)

    if foes_indices:
        dist_foes = differentiable_distances[foes_indices].mean()
    else:
        dist_foes = torch.tensor(MAX_DISTANCE, device=query_embedding.device)

    score = dist_foes - dist_friends
    return score

