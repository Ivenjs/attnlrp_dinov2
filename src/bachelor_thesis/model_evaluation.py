import argparse
import warnings
import torch
from tqdm import tqdm
from utils import load_config, get_db_path, parse_encounter_id
from basemodel import get_model_wrapper
from dataset import GorillaReIDDataset, custom_collate_fn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from knn_helpers import get_knn_db, calculate_distance_batched
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import os

def _perform_knn_ce_evaluation(
    query_embeddings,
    query_labels_int,
    query_encounter_ids_int,
    query_original_indices,
    db_embeddings,
    db_labels_int,
    db_encounter_ids_int,
    k,
    batch_size,
    device,
    distance_metric="euclidean"
):
    """
    Performs the core, batched Cross-Video K-Nearest Neighbors calculation on the GPU.

    This is a low-level utility function that expects all data to be pre-computed tensors.
    """
    num_queries = query_embeddings.shape[0]
    all_predictions = []
    all_actuals = []

    # Process queries in batches to manage memory
    for i in tqdm(range(0, num_queries, batch_size), desc="Running Batched KNN-CV"):
        # --- 1. Select Batch ---
        batch_end = min(i + batch_size, num_queries)
        query_embeddings_batch = query_embeddings[i:batch_end]
        query_encounter_ids_batch = query_encounter_ids_int[i:batch_end]
        query_indices_batch = query_original_indices[i:batch_end] if query_original_indices is not None else None

        # --- 2. Calculate Distances ---
        # Shape: [batch_size, db_size]
        distance_matrix = calculate_distance_batched(db_embeddings, query_embeddings_batch, distance_metric)

        # --- 3. Apply Cross-Encounter Mask ---
        # Create a [batch_size, db_size] mask where True means a DB item is from the same encounter
        same_encounter_mask = (query_encounter_ids_batch.view(-1, 1) == db_encounter_ids_int)
        distance_matrix[same_encounter_mask] = float('inf')

        # --- 4. Apply Self-Match Mask (if query is subset of db) ---
        if query_indices_batch is not None:
            batch_indices = torch.arange(len(query_indices_batch), device=device)
            distance_matrix[batch_indices, query_indices_batch] = float('inf')

        # --- 5. Find K-Nearest Neighbors and Predict ---
        num_neighbors = min(k, distance_matrix.shape[1] - 1)
        if num_neighbors <= 0:
            continue  # Skip if no valid neighbors exist

        # Find top K for each row (dim=1) of the distance matrix
        _, top_k_db_indices = torch.topk(distance_matrix, k=num_neighbors, largest=False, dim=1)

        # Get labels of the top k neighbors and find the most frequent one
        neighbor_labels = db_labels_int[top_k_db_indices]
        predicted_labels_batch = torch.mode(neighbor_labels, dim=1)[0]

        all_predictions.append(predicted_labels_batch)
        all_actuals.append(query_labels_int[i:batch_end])

    # --- 6. Calculate Final Accuracy ---
    if not all_predictions:
        print("Warning: No predictions were made.")
        return 0.0

    predictions_tensor = torch.cat(all_predictions).cpu().numpy()
    actuals_tensor = torch.cat(all_actuals).cpu().numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # Ignore warnings for classes with no predictions
        accuracy = accuracy_score(actuals_tensor, predictions_tensor)
        balanced_accuracy = balanced_accuracy_score(actuals_tensor, predictions_tensor)

    print(f"\nFinal Accuracy: {accuracy:.4f}")
    print(f"Final Balanced Accuracy: {balanced_accuracy:.4f}")
    return balanced_accuracy


def evaluate_model(model_wrapper, dataset, cfg, device, db_embeddings, db_labels, db_videos, query_dataset=None):
    """
    Evaluates a model using Cross-Video KNN, supporting both pre-computed and on-the-fly embeddings.

    Args:
        model_wrapper: The model to evaluate.
        dataset: The source dataset for standard evaluation (e.g., test set).
        cfg: The configuration dictionary.
        device: The torch device to use.
        db_embeddings (torch.Tensor): Embeddings for the entire search database (gallery).
        db_labels (list[str]): String labels for the database.
        db_videos (list[str]): String video IDs for the database.
        query_dataset (Dataset, optional): If provided, this dataset will be used to generate
            query embeddings on-the-fly (e.g., for perturbed images). If None, queries are
            taken from the `dataset` using pre-computed embeddings from `db_embeddings`.
    """
    torch.backends.cudnn.benchmark = True
    model_wrapper.eval()

    # --- 1. Prepare Database Tensors ---
    # Map string labels/videos to integer IDs for efficient processing
    unique_labels = sorted(list(set(db_labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    db_labels_int = torch.tensor([label_to_id[s] for s in db_labels], dtype=torch.long, device=device)

    
    db_encounters = [parse_encounter_id(v) for v in db_videos]
    
    # Map string encounters to integer IDs for efficient processing
    unique_encounters = sorted(list(set(db_encounters)))
    encounter_to_id = {enc: i for i, enc in enumerate(unique_encounters)}
    db_encounter_ids_int = torch.tensor([encounter_to_id[s] for s in db_encounters], dtype=torch.long, device=device)
    """unique_videos = sorted(list(set(db_videos)))
    video_to_id = {video: i for i, video in enumerate(unique_videos)}
    db_video_ids_int = torch.tensor([video_to_id[s] for s in db_videos], dtype=torch.long, device=device)"""

    db_embeddings_gpu = db_embeddings.to(device)
    
    query_embeddings = None
    query_labels_int = None
    #query_video_ids_int = None
    query_encounter_ids_int = None
    query_original_indices = None # Used to mask self-matches if queries are from the DB

    with torch.no_grad():
        # --- 2. Prepare Query Tensors ---
        if query_dataset is None:
            print("Mode: Standard evaluation. Using pre-computed embeddings for queries.")
            query_indices = dataset.images_for_ce_knn
            if not query_indices:
                print("Warning: No images selected for Cross-Encounter KNN. Returning 0 accuracy.")
                return 0.0

            query_original_indices = torch.tensor(query_indices, dtype=torch.long, device=device)
            query_embeddings = db_embeddings_gpu[query_original_indices]
            query_labels_int = db_labels_int[query_original_indices]
            #query_video_ids_int = db_video_ids_int[query_original_indices]
            query_encounter_ids_int = db_encounter_ids_int[query_original_indices]
        else:
            print("Mode: On-the-fly evaluation. Generating embeddings for the query dataset.")
            query_indices = query_dataset.images_for_ce_knn
            if not query_indices:
                print("Warning: No images found for Cross-Encounter KNN in the query dataset. Returning 0 accuracy.")
                return 0.0
            
            query_subset = Subset(query_dataset, query_indices)
            query_loader = DataLoader(
                query_subset,
                batch_size=cfg["data"]["batch_size"],
                num_workers=0,
                shuffle=False,
                collate_fn=custom_collate_fn
            )

            # Generate embeddings and collect metadata for all queries
            q_embeddings, q_labels, q_videos, q_indices = [], [], [], []
            for batch in tqdm(query_loader, desc="Generating query embeddings"):
                query_image_batch = batch["image"].to(device)
                embeddings_batch = model_wrapper(query_image_batch)
                
                q_embeddings.append(embeddings_batch)
                q_labels.extend(batch["label"])
                q_videos.extend(batch["video"])
                # The 'original_index' from the collate_fn gives us the position in the *original DB*
                q_indices.append(batch["original_index"]) 

            query_embeddings = torch.cat(q_embeddings)
            query_original_indices = torch.cat(q_indices).to(device)
            # Map collected string labels/videos to the same integer IDs as the database
            query_labels_int = torch.tensor([label_to_id[s] for s in q_labels], dtype=torch.long, device=device)
            q_encounters = [parse_encounter_id(v) for v in q_videos]
            query_encounter_ids_int = torch.tensor([encounter_to_id[s] for s in q_encounters], dtype=torch.long, device=device)
            #query_video_ids_int = torch.tensor([video_to_id[s] for s in q_videos], dtype=torch.long, device=device)

    # --- 3. Run Evaluation ---
    print(f"Number of queries: {query_embeddings.shape[0]}")
    print(f"Number of database embeddings: {db_embeddings.shape[0]}")
    
    mean_accuracy = _perform_knn_ce_evaluation( # Renamed for clarity
        query_embeddings=query_embeddings,
        query_labels_int=query_labels_int,
        query_encounter_ids_int=query_encounter_ids_int, # Pass ENCOUNTER IDs
        query_original_indices=query_original_indices,
        db_embeddings=db_embeddings_gpu,
        db_labels_int=db_labels_int,
        db_encounter_ids_int=db_encounter_ids_int, # Pass ENCOUNTER IDs
        k=cfg["knn"]["k"],
        batch_size=cfg["data"]["batch_size"],
        device=device,
        distance_metric=cfg["knn"]["distance_metric"]
    )
    
    return mean_accuracy


def main(cfg):
    """Main function to run a standard evaluation."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper, image_transforms, _ = get_model_wrapper(device=DEVICE, cfg=cfg["model"])

    # --- Setup Datasets ---
    split_name = cfg["data"]["analysis_split"]
    split_dir = os.path.join(cfg["data"]["dataset_dir"], split_name)
    split_files = [f for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".png"))]
    split_dataset = GorillaReIDDataset(
        image_dir=split_dir, filenames=split_files, transform=image_transforms, k=cfg["knn"]["k"]
    )

    train_dir = os.path.join(cfg["data"]["dataset_dir"], "train")
    train_files = [f for f in os.listdir(train_dir) if f.lower().endswith((".jpg", ".png"))]
    train_dataset = GorillaReIDDataset(
        image_dir=train_dir, filenames=train_files, transform=image_transforms
    )
    
    # The database consists of both training and validation images
    full_db_dataset = ConcatDataset([train_dataset, split_dataset])

    # --- Get KNN Database Embeddings ---
    db_path = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name="all_dataset", # Use a descriptive name
        split_name=f"train+{split_name}",
        db_dir=cfg["knn"]["db_embeddings_dir"]
    )
    db_embeddings, db_labels, _, db_videos = get_knn_db(
        db_path=db_path,
        dataset=full_db_dataset,
        model_wrapper=model_wrapper,
        batch_size=cfg["data"]["batch_size"],
        device=DEVICE
    )

    # --- Run Evaluation ---
    # In a standard run, the query images are from the validation set,
    # and we use the fast path (query_dataset=None).
    evaluate_model(
        model_wrapper=model_wrapper,
        dataset=split_dataset,
        cfg=cfg,
        device=DEVICE,
        db_embeddings=db_embeddings,
        db_labels=db_labels,
        db_videos=db_videos
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model evaluation script.")
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="The name of the experiment config (e.g., 'finetuned', 'non_finetuned')."
    )
    args, unknown_args = parser.parse_known_args()
    config = load_config(args.config_name, unknown_args)
    main(config)