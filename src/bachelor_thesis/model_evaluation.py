import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from knn_helpers import get_knn_db
from basemodel import TimmWrapper
from utils import load_config, get_db_path
from basemodel import get_model_wrapper
from dataset import GorillaReIDDataset, custom_collate_fn
from torch.utils.data import DataLoader, Subset


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

def KNN_CV_GPU(model_wrapper, query_source_dataset, db_embeddings, images_to_check, db_labels_int, db_video_ids_int, cfg, device, distance_metric="euclidean"):
    """
    Performs BATCHED Cross-Video K-Nearest Neighbors classification on the GPU.
    This version is significantly faster by processing queries in batches.
    """
    # --- 1. Setup Database Tensors ---
    embeddings = db_embeddings.to(device)
    labels_tensor = db_labels_int.to(device)
    video_ids_tensor = db_video_ids_int.to(device) # Shape: [db_size]

    num_queries = len(images_to_check)
    print(f"Number of queries: {num_queries}")
    print(f"Number of database embeddings: {len(embeddings)}")

    # --- 2. Create a DataLoader for the specific query images ---
    # A Subset is the idiomatic way to select specific indices from a dataset.
    query_subset = Subset(query_source_dataset, images_to_check)
    # The custom collate_fn is important if your dataset returns dicts with Nones
    query_loader = DataLoader(
        query_subset, 
        batch_size=cfg["data"]["batch_size"], 
        shuffle=False, # Maintain order
        collate_fn=custom_collate_fn 
    )

    # --- 3. Process Queries in Batches ---
    all_predictions = []
    all_actuals = []
    num_neighbors = cfg["knn"]["k"]

    for batch in tqdm(query_loader, desc="Running Batched KNN-CV"):
        # `batch` is now a dictionary of tensors, e.g., batch['image'], batch['label']
        query_image_batch = batch["image"].to(device)
        
        # --- MAJOR CHANGE 1: Batch Forward Pass ---
        # Get embeddings for the entire batch in one go.
        query_embeddings_batch = model_wrapper(query_image_batch)
        # Shape: [batch_size, dim]

        # --- MAJOR CHANGE 2: Batch Distance Calculation ---
        # Get a distance matrix of [batch_size, db_size]
        distance_matrix = calculate_distance_batched(embeddings, query_embeddings_batch, distance_metric)
        
        # --- MAJOR CHANGE 3: Batch Masking (using broadcasting) ---
        # Get the original indices and video IDs for this specific batch
        query_original_indices = batch["original_index"] # You'll need to modify your dataset/collate to pass this
        query_video_ids_batch = video_ids_tensor[query_original_indices].to(device) # Shape: [batch_size]

        # Create a [batch_size, db_size] mask where True means a DB item is from the same video
        # We reshape query_video_ids_batch to [batch_size, 1] to enable broadcasting
        same_video_mask = (query_video_ids_batch.view(-1, 1) == video_ids_tensor)
        
        # Invalidate same-video neighbors by setting their distance to infinity
        distance_matrix[same_video_mask] = float('inf')

        # Also invalidate the query image itself from being its own neighbor
        # This uses advanced indexing to set specific cells to infinity
        batch_indices = torch.arange(len(query_original_indices), device=device)
        distance_matrix[batch_indices, query_original_indices] = float('inf')
        
        # --- MAJOR CHANGE 4: Batch Top-K and Prediction ---
        k = min(num_neighbors, distance_matrix.shape[1])
        # Find top K for each row (dim=1) of the distance matrix
        _, top_k_db_indices = torch.topk(distance_matrix, k=k, largest=False, dim=1)
        # Shape: [batch_size, k]
        
        # Get labels of the top k neighbors for the entire batch
        neighbor_labels = labels_tensor[top_k_db_indices] # Shape: [batch_size, k]
        
        # Get the most frequent label for each query in the batch
        predicted_labels_batch = torch.mode(neighbor_labels, dim=1)[0] # Shape: [batch_size]
        
        # Store results
        all_predictions.append(predicted_labels_batch)
        all_actuals.append(labels_tensor[query_original_indices])

    # --- 4. Calculate Final Accuracy ---
    if not all_predictions:
        return 0.0

    predictions_tensor = torch.cat(all_predictions)
    actuals_tensor = torch.cat(all_actuals)

    correct_predictions = (predictions_tensor == actuals_tensor).sum().item()
    total_predictions = len(actuals_tensor)
    accuracy = correct_predictions / total_predictions
    
    print(f"Correct predictions: {correct_predictions} / {total_predictions}")
    return accuracy

def evaluate_model(model_wrapper, val_dataset, cfg, device, db_embeddings, db_labels, db_videos,query_dataset = None):
    """
    Evaluates a model using Cross-Video KNN. This version uses the pre-filtered
    list of query images from the dataset object.

    Args:
        model_wrapper: The model object (e.g., TimmWrapper).
        val_dataset (GorillaReIDDataset_v2): The validation dataset object.
        config: A configuration object/namespace.
        model_checkpoint_path (str): Path to the model checkpoint for caching.
    """
    torch.backends.cudnn.benchmark = True
    model_wrapper.eval()

    # --- 2. Map String Labels and Video IDs to Integers ---
    unique_labels = sorted(list(set(db_labels)))
    unique_videos = sorted(list(set(db_videos)))
    
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    video_to_id = {video: i for i, video in enumerate(unique_videos)}
    
    db_labels_int = torch.tensor([label_to_id[s] for s in db_labels], dtype=torch.long)
    db_video_ids_int = torch.tensor([video_to_id[s] for s in db_videos], dtype=torch.long)
    
    # --- 3. Get Filtered Query Images from Dataset and Run KNN ---
    if query_dataset is None:
        print("Using standard validation set for queries.")
        query_source_dataset = val_dataset
    else:
        print("Using provided perturbed dataset for queries.")
        query_source_dataset = query_dataset

    if not query_source_dataset.images_for_cv_knn:
        print("Warning: No images were found suitable for Cross-Video KNN evaluation. Returning 0 accuracy.")
        return 0.0

    with torch.no_grad():
        mean_accuracy = KNN_CV_GPU(
            model_wrapper=model_wrapper, 
            query_source_dataset=query_source_dataset, 
            db_embeddings=db_embeddings,
            images_to_check=query_source_dataset.images_for_cv_knn,
            db_labels_int=db_labels_int,
            db_video_ids_int=db_video_ids_int,
            cfg=cfg, 
            device=device,
            distance_metric=cfg["knn"]["distance_metric"] 
        )

    print(f"\nFinal Cross-Video KNN@{cfg['knn']['k']} Accuracy: {mean_accuracy:.4f}")
    return mean_accuracy


if __name__ == '__main__':
    # --- 1. Configuration (Note the change in the knn part) ---
    cfg = load_config("finetuned", [])
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- 2. Setup Model, Transforms, and Dataset ---
    model_wrapper, image_transforms, _ = get_model_wrapper(device=DEVICE, cfg=cfg["model"])
    split_name = "validation"
    val_dir = os.path.join(cfg["data"]["dataset_dir"], split_name)
    val_files = [f for f in os.listdir(val_dir) if f.lower().endswith((".jpg", ".png"))]

    # KEY CHANGE HERE: Pass 'k' to the dataset constructor.
    val_dataset = GorillaReIDDataset(
        image_dir=val_dir,
        filenames=val_files,
        transform=image_transforms,
        k=cfg["knn"]["k"], 
    )

    db_path = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset=val_dataset,
        split_name=split_name,
        db_dir=cfg["knn"]["db_embeddings_dir"]
    )
    val_db_embeddings, val_db_labels, val_db_filenames, val_db_videos = get_knn_db(
        db_path=db_path,
        dataset=val_dataset,
        model_wrapper=model_wrapper,
        batch_size=cfg["data"]["batch_size"],
        device=DEVICE
    )

    # --- 3. Run Evaluation ---
    evaluate_model(
        model_wrapper=model_wrapper,
        val_dataset=val_dataset,
        cfg=cfg,
        device=DEVICE,
        db_embeddings=val_db_embeddings,
        db_labels=val_db_labels,
        db_videos=val_db_videos
    )