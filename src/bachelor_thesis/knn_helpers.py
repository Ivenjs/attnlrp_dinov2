from basemodel import TimmWrapper
import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from typing import Tuple, List
from torch.utils.data import DataLoader
import torch
from utils import parse_encounter_id
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

    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
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



def calculate_distance_batched_normalized(db_embeddings, query_batch_embeddings, metric):
    """
    Calculates the NORMALIZED distances from a batch of queries to the entire database.
    
    Args:
        db_embeddings (Tensor): Shape [db_size, dim]
        query_batch_embeddings (Tensor): Shape [batch_size, dim]
        metric (str): 'cosine' or 'euclidean'
    
    Returns:
        Tensor: Distance matrix of shape [batch_size, db_size]
    """
    db_norm = F.normalize(db_embeddings, p=2, dim=1)
    query_norm = F.normalize(query_batch_embeddings, p=2, dim=1)
    if metric == "cosine":
        similarity_matrix = torch.matmul(query_norm, db_norm.T)
        return 1 - similarity_matrix
    else:
        return torch.cdist(query_norm, db_norm)
    
def calculate_distance_normalized(embeddings, test_embedding, metric):
    """calculates the NORMALIZED distance from a single test embedding to all embeddings."""
    embeddings = F.normalize(embeddings, p=2, dim=1)
    test_embedding = F.normalize(test_embedding, p=2, dim=1)
    if metric == "euclidean":
        distance = torch.norm(embeddings - test_embedding, dim=1)
    elif metric == "cosine":
        cosine_similarity = torch.matmul(embeddings, test_embedding.transpose(0, 1)).squeeze()
        distance = 1 - cosine_similarity
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")
    return distance

def create_exclusion_mask(
    query_filename: str,
    query_video_id: str,
    db_filenames: List[str],
    db_video_ids: List[str],
    device: torch.device,
    exclude_self: bool = False,
    cross_video: bool = False,
    cross_encounter: bool = False
) -> torch.Tensor:
    """
    Creates a boolean mask to exclude items from the database based on flags.

    Args:
        query_filename: Filename of the query item.
        query_video_id: Video ID of the query item.
        db_filenames: List of all database filenames.
        db_video_ids: List of all database video IDs.
        device: The torch device to create the mask on.
        exclude_self: If True, excludes the item with the exact same filename.
        cross_video: If True, excludes all items from the same video_id.
        cross_encounter: If True, excludes all items from the same camera on the same day.

    Returns:
        A boolean tensor where True indicates an item should be excluded.
    """
    n_db = len(db_filenames)
    exclusion_mask = torch.zeros(n_db, dtype=torch.bool, device=device)

    # 1. Self-exclusion (always applied if the flag is True)
    if exclude_self:
        try:
            q_idx = db_filenames.index(query_filename)
            exclusion_mask[q_idx] = True
        except ValueError:
            pass

    # 2. Video-based exclusion
    if cross_video and query_video_id and db_video_ids:
        same_video_mask = torch.tensor(
            [vid == query_video_id for vid in db_video_ids],
            dtype=torch.bool, device=device
        )
        exclusion_mask |= same_video_mask

    # 3. Encounter-based exclusion
    if cross_encounter and query_video_id and db_video_ids:
        q_cam, q_date = parse_encounter_id(query_video_id)
        if q_cam and q_date:
            same_encounter_mask = torch.tensor([
                (cam == q_cam and date == q_date)
                for cam, date in (parse_encounter_id(vid) for vid in db_video_ids)
            ], dtype=torch.bool, device=device)
            exclusion_mask |= same_encounter_mask

    return exclusion_mask