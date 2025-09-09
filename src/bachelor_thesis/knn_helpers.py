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
    num_workers: int = 4,
    use_amp: bool = True,
) -> Tuple[torch.Tensor, list, list, list]:
    """
    Generates and saves embeddings for a given dataset using performance optimizations.
    Saves embeddings, labels, and filenames.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True, 
        collate_fn=custom_collate_fn,
    )
    print(f"Generating embeddings for {len(dataloader.dataset)} images...")

    all_embeddings_list = []
    all_labels = []
    all_filenames = []
    all_videos = []

    model_wrapper.model.eval()

    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        for batch in tqdm(dataloader, desc=f"Generating embeddings for {os.path.basename(output_path)}"):
            images = batch["image"]
            labels = batch["label"]
            filenames = batch["filename"]
            videos = batch["video"]

            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)

            embeddings = model_wrapper(images)

            all_embeddings_list.append(embeddings)

            all_labels.extend(labels)
            all_filenames.extend(filenames)
            all_videos.extend(videos)

    final_embeddings_gpu = torch.cat(all_embeddings_list, dim=0)

    db_data = {
        "embeddings": final_embeddings_gpu.cpu(),
        "labels": all_labels,
        "filenames": all_filenames,
        "videos": all_videos,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(db_data, output_path)

    print(f"Saved {len(all_filenames)} embeddings to {output_path}")

    # Return the GPU tensor and CPU lists, as per the original function's intent
    return final_embeddings_gpu, all_labels, all_filenames, all_videos

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



def calculate_distance_batched(db_embeddings, query_batch_embeddings, metric):
    """
    Calculates distances from a batch of queries to the entire database.
    
    Args:
        db_embeddings (Tensor): Shape [db_size, dim]
        query_batch_embeddings (Tensor): Shape [batch_size, dim]
        metric (str): 'cosine' or 'euclidean'
    
    Returns:
        Tensor: Distance matrix of shape [batch_size, db_size]
    """
    if metric == "cosine":
        # Normalize both sets of embeddings
        db_norm = F.normalize(db_embeddings, p=2, dim=1)
        query_norm = F.normalize(query_batch_embeddings, p=2, dim=1)
        # Calculate similarity matrix using matrix multiplication
        similarity_matrix = torch.matmul(query_norm, db_norm.T)
        # Convert similarity to distance
        return 1 - similarity_matrix
    else: # Default to Euclidean
        return torch.cdist(query_batch_embeddings, db_embeddings)
    
def calculate_distance(embeddings, test_embedding, metric):
    embeddings = embeddings.to(embeddings.device)
    if metric == "euclidean":
        distance = torch.norm(embeddings - test_embedding, dim=1)
    elif metric == "cosine":
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        normalized_test_embedding = F.normalize(test_embedding, p=2, dim=0)
        cosine_similarity = torch.matmul(normalized_embeddings, normalized_test_embedding.transpose(0, 1)).squeeze()
        distance = 1 - cosine_similarity
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")
    return distance

