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
from knn_helpers import calculate_distance_batched, calculate_distance


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
    query_loader = DataLoader(
        query_subset, 
        num_workers=0, #try this to reduce random noise
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
        
        # Get embeddings for the entire batch in one go.
        query_embeddings_batch = model_wrapper(query_image_batch)
        # Shape: [batch_size, dim]

        # Get a distance matrix of [batch_size, db_size]
        distance_matrix = calculate_distance_batched(embeddings, query_embeddings_batch, distance_metric)
        
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

def evaluate_model(model_wrapper, val_dataset, cfg, device, db_embeddings, db_labels, db_videos, query_dataset=None):
    """
    Evaluates a model using Cross-Video KNN with a unified, high-performance function.

    - If `query_dataset` is None, it uses the fast, pre-computed embedding path.
    - If `query_dataset` is provided (e.g., perturbed images), it computes their
      embeddings on-the-fly.
    """
    torch.backends.cudnn.benchmark = True
    model_wrapper.eval()

    # --- 1. Map String Labels and Video IDs to Integers ---
    # This mapping must be consistent for both database and query sources.
    unique_labels = sorted(list(set(db_labels)))
    unique_videos = sorted(list(set(db_videos)))
    
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    video_to_id = {video: i for i, video in enumerate(unique_videos)}
    
    db_labels_int = torch.tensor([label_to_id.get(s) for s in db_labels], dtype=torch.long)
    db_video_ids_int = torch.tensor([video_to_id.get(s) for s in db_videos], dtype=torch.long)
    
    # --- 2. Determine Query Source and Run KNN ---
    images_to_check = []
    mean_accuracy = 0.0

    with torch.no_grad():
        # Case 1: Standard evaluation. Queries are from the validation set itself.
        # Use the fast path with pre-computed embeddings.
        if query_dataset is None:
            print("Mode: Standard evaluation. Using pre-computed embeddings for queries.")
            query_source_dataset = val_dataset
            images_to_check = query_source_dataset.images_for_cv_knn

            if not images_to_check:
                print("Warning: No images found for Cross-Video KNN. Returning 0 accuracy.")
                return 0.0

            mean_accuracy = KNN_CV_GPU_unified(
                db_embeddings=db_embeddings,
                db_labels=db_labels_int,
                db_video_ids=db_video_ids_int,
                query_indices=images_to_check,
                cfg=cfg,
                device=device,
                # --- Mode 2 Args: Provide source tensors for queries ---
                query_embeddings_source=db_embeddings,
                query_labels_source=db_labels_int,
                query_video_ids_source=db_video_ids_int,
                distance_metric=cfg["knn"]["distance_metric"]
            )

        # Case 2: A separate query dataset is provided (e.g., for robustness tests).
        # Must compute embeddings on-the-fly.
        else:
            query_source_dataset = query_dataset
            images_to_check = query_source_dataset.images_for_cv_knn
            
            if not images_to_check:
                print("Warning: No images found for Cross-Video KNN in the query dataset. Returning 0 accuracy.")
                return 0.0

            mean_accuracy = KNN_CV_GPU_unified(
                db_embeddings=db_embeddings,
                db_labels=db_labels_int,
                db_video_ids=db_video_ids_int,
                query_indices=images_to_check,
                cfg=cfg,
                device=device,
                # --- Mode 1 Args: Provide model and dataset to generate embeddings ---
                model_wrapper=model_wrapper,
                query_source_dataset=query_source_dataset,
                distance_metric=cfg["knn"]["distance_metric"]
            )

    print(f"\nFinal Cross-Video KNN@{cfg['knn']['k']} Accuracy: {mean_accuracy:.4f}")
    return mean_accuracy

def KNN_CV_GPU_unified(
    # --- Database Information (Always required) ---
    db_embeddings,
    db_labels,
    db_video_ids,
    
    # --- Query Specification (Always required) ---
    query_indices,

    # --- Configuration (Always required) ---
    cfg,
    device,

    # --- Mode 1: To compute query embeddings on-the-fly ---
    model_wrapper=None,
    query_source_dataset=None,

    # --- Mode 2: To use pre-computed query embeddings ---
    query_embeddings_source=None,
    query_labels_source=None,
    query_video_ids_source=None,

    # --- Optional ---
    distance_metric="euclidean"
):
    """
    Performs BATCHED Cross-Video K-Nearest Neighbors on the GPU with two modes.
    """
    # --- 1. Input Validation and Mode Selection ---
    use_precomputed = query_embeddings_source is not None
    
    if use_precomputed:
        if query_labels_source is None or query_video_ids_source is None:
            raise ValueError("If using pre-computed embeddings, you must also provide source labels and video IDs.")
        print("Using pre-computed embeddings.")
        query_embeddings_source = query_embeddings_source.to(device)
        query_labels_source = query_labels_source.to(device)
        query_video_ids_source = query_video_ids_source.to(device)
    else:
        if model_wrapper is None or query_source_dataset is None:
            raise ValueError("If not using pre-computed embeddings, you must provide a model and source dataset.")
        print("Computing embeddings on-the-fly.")

    # --- 2. Setup Database Tensors on GPU ---
    db_embeddings = db_embeddings.to(device)
    db_labels_tensor = db_labels.to(device)
    db_video_ids_tensor = db_video_ids.to(device)

    num_queries = len(query_indices)
    batch_size = cfg["data"]["batch_size"]
    num_neighbors = cfg["knn"]["k"]
    
    print(f"Number of queries: {num_queries}")
    print(f"Number of database embeddings: {len(db_embeddings)}")

    all_predictions = []
    all_actuals = []

    # --- 3. Process Queries in Batches ---
    query_iterator = None
    if not use_precomputed:
        query_subset = Subset(query_source_dataset, query_indices)
        query_loader = DataLoader(
            query_subset, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=custom_collate_fn
        )
        query_iterator = query_loader
    else:
        query_iterator = range(0, num_queries, batch_size)
        query_indices_tensor = torch.as_tensor(query_indices, device=device)

    for batch_data in tqdm(query_iterator, desc="Running Batched KNN-CV"):
        if not use_precomputed:
            query_image_batch = batch_data["image"].to(device)
            query_original_indices = batch_data["original_index"].to(device)
            query_embeddings_batch = model_wrapper(query_image_batch)
            # Assuming query is a subset of db, so labels/videos can be indexed directly
            query_labels_batch = db_labels_tensor[query_original_indices]
            query_video_ids_batch = db_video_ids_tensor[query_original_indices]
        else:
            start_idx = batch_data
            end_idx = start_idx + batch_size
            batch_query_subset_indices = query_indices_tensor[start_idx:end_idx]
            query_original_indices = batch_query_subset_indices
            query_embeddings_batch = query_embeddings_source[query_original_indices].to(device)
            query_labels_batch = query_labels_source[query_original_indices].to(device)
            query_video_ids_batch = query_video_ids_source[query_original_indices].to(device)
        
        distance_matrix = calculate_distance_batched(db_embeddings, query_embeddings_batch, distance_metric)
        same_video_mask = (query_video_ids_batch.view(-1, 1) == db_video_ids_tensor)
        distance_matrix[same_video_mask] = float('inf')
        
        batch_indices = torch.arange(len(query_original_indices), device=device)
        distance_matrix[batch_indices, query_original_indices] = float('inf')
        
        k = min(num_neighbors, distance_matrix.shape[1] - 1)
        if k <= 0: continue
        _, top_k_db_indices = torch.topk(distance_matrix, k=k, largest=False, dim=1)
        
        neighbor_labels = db_labels_tensor[top_k_db_indices]
        predicted_labels_batch = torch.mode(neighbor_labels, dim=1)[0]
        
        all_predictions.append(predicted_labels_batch)
        all_actuals.append(query_labels_batch)

    # --- 4. Calculate Final Accuracy ---
    if not all_predictions:
        print("No queries were processed.")
        return 0.0

    predictions_tensor = torch.cat(all_predictions)
    actuals_tensor = torch.cat(all_actuals)

    correct_predictions = (predictions_tensor == actuals_tensor).sum().item()
    total_predictions = len(actuals_tensor)
    
    if total_predictions == 0: return 0.0
    accuracy = correct_predictions / total_predictions
    
    print(f"\nFinal Accuracy: {accuracy:.4f} ({correct_predictions} / {total_predictions})")
    return accuracy


if __name__ == '__main__':
    # --- 1. Configuration (Note the change in the knn part) ---
    cfg = load_config("finetuned", [])
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- 2. Setup Model, Transforms, and Dataset ---
    model_wrapper, image_transforms, _ = get_model_wrapper(device=DEVICE, cfg=cfg["model"])
    split_name = cfg["data"]["analysis_split"]
    split_dir = os.path.join(cfg["data"]["dataset_dir"], split_name)
    split_files = [f for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".png"))]

    dataset = GorillaReIDDataset(
        image_dir=split_dir,
        filenames=split_files,
        transform=image_transforms,
        k=cfg["knn"]["k"], 
    )

    db_path = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset=dataset,
        split_name=split_name,
        db_dir=cfg["knn"]["db_embeddings_dir"]
    )
    db_embeddings, db_labels, db_filenames, db_videos = get_knn_db(
        db_path=db_path,
        dataset=dataset,
        model_wrapper=model_wrapper,
        batch_size=cfg["data"]["batch_size"],
        device=DEVICE
    )

    # --- 3. Run Evaluation ---
    evaluate_model(
        model_wrapper=model_wrapper,
        val_dataset=dataset,
        cfg=cfg,
        device=DEVICE,
        db_embeddings=db_embeddings,
        db_labels=db_labels,
        db_videos=db_videos
    )