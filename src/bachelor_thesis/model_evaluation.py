import argparse
import warnings
import torch
from tqdm import tqdm
from utils import load_config, get_db_path, parse_encounter_id
from basemodel import get_model_wrapper
from dataset import GorillaReIDDataset, custom_collate_fn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from typing import List
from knn_helpers import get_knn_db, calculate_distance_batched_normalized
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from torchvision import transforms
from torchvision.transforms import RandAugment
import os
import operator
import torch.nn.functional as F
from collections import defaultdict
from knn_helpers import create_exclusion_mask

def analyze_predictions_by_class(prediction_details, id_to_label, n=50):
    """
    Findet für korrekte und inkorrekte Vorhersagen jeweils die zuversichtlichste
    Vorhersage pro Klasse und gibt die Top N Ergebnisse davon aus.
    """
    if not prediction_details:
        return

    correct_preds = [p for p in prediction_details if p['is_correct']]
    incorrect_preds = [p for p in prediction_details if not p['is_correct']]

    best_correct_per_class = {}
    for pred in correct_preds:
        label = pred['actual_label']
        if label not in best_correct_per_class or pred['confidence_distance'] < best_correct_per_class[label]['confidence_distance']:
            best_correct_per_class[label] = pred

    sorted_best_correct = sorted(best_correct_per_class.values(), key=operator.itemgetter('confidence_distance'))

    print(f"\n--- Top {n} most confident CORRECT predictions (one per class) ---")
    for pred in sorted_best_correct[:n]:
        label_name = id_to_label.get(pred['actual_label'], 'Unknown')
        print(
            f"File: {pred['filename']}, "
            f"Predicted: {label_name}, "
            f"Distance: {pred['confidence_distance']:.4f}"
        )

    best_incorrect_per_class = {}
    for pred in incorrect_preds:
        label = pred['actual_label']
        if label not in best_incorrect_per_class or pred['confidence_distance'] < best_incorrect_per_class[label]['confidence_distance']:
            best_incorrect_per_class[label] = pred

    sorted_best_incorrect = sorted(best_incorrect_per_class.values(), key=operator.itemgetter('confidence_distance'))

    print(f"\n--- Top {n} most confident INCORRECT predictions (one per class) ---")
    for pred in sorted_best_incorrect[:n]:
        actual_name = id_to_label.get(pred['actual_label'], 'Unknown')
        predicted_name = id_to_label.get(pred['predicted_label'], 'Unknown')
        print(
            f"File: {pred['filename']}, "
            f"Actual: {actual_name}, "
            f"Predicted: {predicted_name}, "
            f"Distance: {pred['confidence_distance']:.4f}"
        )

#TODO: use this in perform_knn_ce_evaluation for a unified approach alongside the lrp masking. But would need to update the masking to be wiht integers for lrp of non-integers here
def create_batched_exclusion_mask(
    query_filenames_batch: List[str],
    query_video_ids_batch: List[str],
    db_filenames: List[str],
    db_video_ids: List[str],
    device: torch.device,
    exclude_self: bool = True,
    cross_encounter: bool = True
) -> torch.Tensor:
    """
    A wrapper that applies `create_exclusion_mask` to a batch of queries.
    """
    all_masks = []
    for q_filename, q_video_id in zip(query_filenames_batch, query_video_ids_batch):
        single_mask = create_exclusion_mask(
            query_filename=q_filename,
            query_video_id=q_video_id,
            db_filenames=db_filenames,
            db_video_ids=db_video_ids,
            device=device,
            exclude_self=exclude_self,
            cross_encounter=cross_encounter
        )
        all_masks.append(single_mask)

    if all_masks:
        return torch.stack(all_masks, dim=0)
    else:
        return torch.empty((0, len(db_filenames)), dtype=torch.bool, device=device)
    
def perform_knn_ce_evaluation(
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
    distance_metric="euclidean",
    query_filenames=None
):
    """
    Performs the core, batched Cross-Video K-Nearest Neighbors calculation on the GPU.

    This is a low-level utility function that expects all data to be pre-computed tensors.
    """
    num_queries = query_embeddings.shape[0]
    all_predictions = []
    all_actuals = []
    prediction_details = []  

    # Process queries in batches to manage memory
    for i in tqdm(range(0, num_queries, batch_size), desc="Running Batched KNN-CE"):
        batch_end = min(i + batch_size, num_queries)
        query_embeddings_batch = query_embeddings[i:batch_end]
        query_labels_batch = query_labels_int[i:batch_end]
        query_encounter_ids_batch = query_encounter_ids_int[i:batch_end]
        query_indices_batch = query_original_indices[i:batch_end] if query_original_indices is not None else None
        query_filenames_batch = query_filenames[i:batch_end] if query_filenames is not None else None

        distance_matrix = calculate_distance_batched_normalized(db_embeddings, query_embeddings_batch, distance_metric)

        # --- Apply Cross-Encounter Mask ---
        same_encounter_mask = (query_encounter_ids_batch.view(-1, 1) == db_encounter_ids_int)

        #same label mask
        same_label_mask = (query_labels_batch.view(-1, 1) == db_labels_int)

        final_mask = same_encounter_mask & same_label_mask #use this to still allow other labels in same encounter. We would also need to update the cross encounter filtering for proxy scores
        distance_matrix[same_encounter_mask] = float('inf')

        #distance_matrix[same_encounter_mask] = float('inf')

        # --- Apply Self-Match Mask (if query is subset of db) --- Probably not needed with encounter masking but just in case
        if query_indices_batch is not None:
            batch_indices = torch.arange(len(query_indices_batch), device=device)
            distance_matrix[batch_indices, query_indices_batch] = float('inf')

        num_neighbors = min(k, distance_matrix.shape[1] - 1)
        if num_neighbors <= 0:
            continue  # Skip if no valid neighbors exist

        top_k_distances, top_k_db_indices = torch.topk(distance_matrix, k=num_neighbors, largest=False, dim=1)

        neighbor_labels = db_labels_int[top_k_db_indices]
        predicted_labels_batch = torch.mode(neighbor_labels, dim=1)[0]
        actual_labels_batch = query_labels_int[i:batch_end]

        if query_filenames_batch is not None:
            for j in range(len(predicted_labels_batch)):
                prediction_details.append({
                    'filename': query_filenames_batch[j],
                    'actual_label': actual_labels_batch[j].item(),
                    'predicted_label': predicted_labels_batch[j].item(),
                    'is_correct': (actual_labels_batch[j] == predicted_labels_batch[j]).item(),
                    'confidence_distance': top_k_distances[j, 0].item()
                })

        all_predictions.append(predicted_labels_batch)
        all_actuals.append(actual_labels_batch)

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
    return balanced_accuracy, prediction_details


def evaluate_model(model_wrapper, query_indices_in_db, cfg, device, db_embeddings, db_labels, db_videos, query_dataset=None,  db_filenames=None):
    """
    Evaluates a model using Cross-Video KNN, supporting both pre-computed and on-the-fly embeddings.

    Args:
        model_wrapper: The model to evaluate.
        query_indices_in_db: List of indices in the DB that correspond to the query set. Needed to make order of concat dataset not matter
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
    id_to_label = {i: label for label, i in label_to_id.items()}
    db_labels_int = torch.tensor([label_to_id[s] for s in db_labels], dtype=torch.long, device=device)

    
    db_encounters = [parse_encounter_id(v) for v in db_videos]
    # Map string encounters to integer IDs for efficient processing
    unique_encounters = sorted(list(set(db_encounters)))
    encounter_to_id = {enc: i for i, enc in enumerate(unique_encounters)}
    db_encounter_ids_int = torch.tensor([encounter_to_id[s] for s in db_encounters], dtype=torch.long, device=device)


    db_embeddings_gpu = db_embeddings.to(device)
    
    query_embeddings = None
    query_labels_int = None
    query_encounter_ids_int = None
    query_original_indices = None # Used to mask self-matches if queries are from the DB

    query_filenames = None
    if query_dataset is None:
        if db_filenames is not None:
            query_filenames = [db_filenames[i] for i in query_indices_in_db]
    else:
        # For on-the-fly mode, get filenames from the query_dataset
        local_query_indices = query_dataset.images_for_ce_knn
        if db_filenames is not None:
            query_filenames = [query_dataset.filenames[i] for i in local_query_indices]

    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        # --- 2. Prepare Query Tensors ---
        if query_dataset is None:
            print("Mode: Standard evaluation. Using pre-computed embeddings for queries.")
            query_original_indices = torch.tensor(query_indices_in_db, dtype=torch.long, device=device)
            query_embeddings = db_embeddings_gpu[query_original_indices]
            query_labels_int = db_labels_int[query_original_indices]
            query_encounter_ids_int = db_encounter_ids_int[query_original_indices]
        else:
            print("Mode: On-the-fly evaluation. Generating embeddings for the query dataset.")
            local_query_indices = query_dataset.images_for_ce_knn
            if not local_query_indices:
                print("Warning: No images found for Cross-Encounter KNN in the query dataset. Returning 0 accuracy.")
                return 0.0
            
            # The on-the-fly mode also needs the global indices for self-masking.
            if len(local_query_indices) != len(query_indices_in_db):
                 raise ValueError("Mismatch between on-the-fly query indices and provided global indices.")
            
            query_original_indices = torch.tensor(query_indices_in_db, dtype=torch.long, device=device)

            query_subset = Subset(query_dataset, local_query_indices)
            query_loader = DataLoader(
                query_subset,
                batch_size=cfg["data"]["batch_size"],
                num_workers=0,
                shuffle=False,
                collate_fn=custom_collate_fn
            )

            q_embeddings, q_labels, q_videos = [], [], []
            for batch in tqdm(query_loader, desc="Generating query embeddings"):
                query_image_batch = batch["image"].to(device)
                embeddings_batch = model_wrapper(query_image_batch)                
                q_embeddings.append(embeddings_batch)
                q_labels.extend(batch["label"])
                q_videos.extend(batch["video"])

            query_embeddings = torch.cat(q_embeddings)
            query_labels_int = torch.tensor([label_to_id.get(s) for s in q_labels], dtype=torch.long, device=device)
            q_encounters = [parse_encounter_id(v) for v in q_videos]
            query_encounter_ids_int = torch.tensor([encounter_to_id.get(s) for s in q_encounters], dtype=torch.long, device=device)

    # --- 3. Run Evaluation ---
    print(f"Number of queries: {query_embeddings.shape[0]}")
    print(f"Number of database embeddings: {db_embeddings.shape[0]}")
    print("Unique db encounters (len):", len(torch.unique(db_encounter_ids_int, dim=0 if db_encounter_ids_int.dim()>1 else None)))
    print("Unique query encounters (len):", len(torch.unique(query_encounter_ids_int, dim=0 if query_encounter_ids_int.dim()>1 else None)))

    mean_accuracy, prediction_details = perform_knn_ce_evaluation( 
        query_embeddings=query_embeddings,
        query_labels_int=query_labels_int,
        query_encounter_ids_int=query_encounter_ids_int,
        query_original_indices=query_original_indices,
        db_embeddings=db_embeddings_gpu,
        db_labels_int=db_labels_int,
        db_encounter_ids_int=db_encounter_ids_int,
        k=cfg["knn"]["k"],
        batch_size=cfg["data"]["batch_size"],
        device=device,
        distance_metric=cfg["knn"]["distance_metric"],
        query_filenames=query_filenames
    )

    analyze_predictions_by_class(prediction_details, id_to_label, n=50)
    
    return mean_accuracy


def main(cfg):
    """Main function to run a standard evaluation."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper, image_transforms, _ = get_model_wrapper(device=DEVICE, cfg=cfg["model"])

    """image_transforms = transforms.Compose([
        transforms.Resize((cfg["model"]["img_size"], cfg["model"]["img_size"])),
        RandAugment(num_ops=0, magnitude=8), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])"""

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
    datasets = [split_dataset, train_dataset]
    full_db_dataset = ConcatDataset(datasets)
    full_dataset_splits = "+".join([os.path.basename(d.image_dir) for d in datasets])

    all_db_filenames = []
    for d in datasets:
        all_db_filenames.extend(d.filenames)

    query_dataset_offset = 0
    found = False
    for d in datasets:
        if d is split_dataset:
            found = True
            break
        query_dataset_offset += len(d)

    print("Query dataset offset in DB:", query_dataset_offset)

    if not found:
        raise RuntimeError("Query dataset (split_dataset) not found in db_constituents.")

    local_query_indices = split_dataset.images_for_ce_knn

    global_query_indices = [idx + query_dataset_offset for idx in local_query_indices]

    # --- Get KNN Database Embeddings ---
    db_path = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name=train_dataset.dataset_name,
        split_name=full_dataset_splits,
        bp_transforms=cfg["model"]["bp_transforms"],
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
        query_indices_in_db=global_query_indices, # Pass the new global indices
        cfg=cfg,
        device=DEVICE,
        db_embeddings=db_embeddings,
        db_labels=db_labels,
        db_videos=db_videos,
        #query_dataset=split_dataset,
        db_filenames=all_db_filenames
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