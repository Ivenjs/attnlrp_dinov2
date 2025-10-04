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
from sklearn.model_selection import train_test_split
import operator
import torch.nn.functional as F
from collections import defaultdict
from knn_helpers import create_exclusion_mask
import json


import operator
import json
import os

def analyze_predictions_by_class(prediction_details, id_to_label, cfg, n=50, output_dir="./prediction_infos"):
    """
    Finds the most confident correct and incorrect prediction for each class,
    prints the top N results, and saves all prediction details to a JSON file.
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
        closest_neighbor_file = pred.get('top_k_neighbor_filenames', ['N/A'])[0]
        print(
            f"File: {pred['filename']}, "
            f"Predicted: {label_name}, "
            f"Distance: {pred['confidence_distance']:.4f}, "
            f"Closest Match: {closest_neighbor_file}"
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
        closest_neighbor_file = pred.get('top_k_neighbor_filenames', ['N/A'])[0]
        print(
            f"File: {pred['filename']}, "
            f"Actual: {actual_name}, "
            f"Predicted: {predicted_name}, "
            f"Distance: {pred['confidence_distance']:.4f}, "
            f"Closest Match (cause): {closest_neighbor_file}"
        )
    
    finetuned_str = "finetuned" if cfg["model"].get("finetuned", False) else "base"
    
    json_data = {
        "correct_predictions": [
            {
                "filename": p['filename'],
                "actual_label": id_to_label.get(p['actual_label'], 'Unknown'),
                "predicted_label": id_to_label.get(p['predicted_label'], 'Unknown'),
                "confidence_distance": p['confidence_distance'],
                "top_k_distances": p.get('top_k_distances', []),
                "top_k_neighbor_filenames": p.get('top_k_neighbor_filenames', []),
                "top_k_neighbor_labels": [id_to_label.get(l, "Unknown") for l in p.get('top_k_neighbor_labels', [])],
            }
            for p in correct_preds
        ],
        "incorrect_predictions": [
            {
                "filename": p['filename'],
                "actual_label": id_to_label.get(p['actual_label'], 'Unknown'),
                "predicted_label": id_to_label.get(p['predicted_label'], 'Unknown'),
                "confidence_distance": p['confidence_distance'],
                "top_k_distances": p.get('top_k_distances', []),
                "top_k_neighbor_filenames": p.get('top_k_neighbor_filenames', []),
                "top_k_neighbor_labels": [id_to_label.get(l, "Unknown") for l in p.get('top_k_neighbor_labels', [])],
            }
            for p in incorrect_preds
        ],
    }

    os.makedirs(output_dir, exist_ok=True)

    zoo_str = "_zoo" if "zoo" in cfg["data"]["dataset_dir"].lower() else ""

    output_path = os.path.join(output_dir, f"{finetuned_str}{zoo_str}_predictions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    print(f"\n[INFO] Full prediction details written to: {output_path}")

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
    query_filenames=None,
    db_filenames=None,
    return_raw_preds=False
):
    """
    Performs the core, batched Cross-Video K-Nearest Neighbors calculation on the GPU.

    This is a low-level utility function that expects all data to be pre-computed tensors.
    """
    num_queries = query_embeddings.shape[0]
    all_predictions = []
    all_actuals = []
    prediction_details = []  

    for i in tqdm(range(0, num_queries, batch_size), desc="Running Batched KNN-CE"):
        batch_end = min(i + batch_size, num_queries)
        query_embeddings_batch = query_embeddings[i:batch_end]
        query_labels_batch = query_labels_int[i:batch_end]
        query_encounter_ids_batch = query_encounter_ids_int[i:batch_end]
        query_indices_batch = query_original_indices[i:batch_end] if query_original_indices is not None else None
        query_filenames_batch = query_filenames[i:batch_end] if query_filenames is not None else None

        distance_matrix = calculate_distance_batched_normalized(db_embeddings, query_embeddings_batch, distance_metric)

        same_encounter_mask = (query_encounter_ids_batch.view(-1, 1) == db_encounter_ids_int)

        #Cross-Encounter Mask
        distance_matrix[same_encounter_mask] = float('inf')


        #Self-Match Mask (if query is subset of db). Probably not needed with encounter masking but just in case
        if query_indices_batch is not None:
            batch_indices = torch.arange(len(query_indices_batch), device=device)
            distance_matrix[batch_indices, query_indices_batch] = float('inf')

        num_neighbors = min(k, distance_matrix.shape[1] - 1)
        if num_neighbors <= 0:
            continue  

        top_k_distances, top_k_db_indices = torch.topk(distance_matrix, k=num_neighbors, largest=False, dim=1)

        neighbor_labels = db_labels_int[top_k_db_indices]
        predicted_labels_batch = torch.mode(neighbor_labels, dim=1)[0]
        actual_labels_batch = query_labels_int[i:batch_end]

        if query_filenames_batch is not None and db_filenames is not None: # 
            for j in range(len(predicted_labels_batch)):
                k_neighbor_db_indices = top_k_db_indices[j].cpu().tolist()
                
                k_neighbor_filenames = [db_filenames[idx] for idx in k_neighbor_db_indices]
                
                all_k_distances = top_k_distances[j].cpu().tolist()
                neighbor_labels_list = neighbor_labels[j].cpu().tolist()

                prediction_details.append({
                    'filename': query_filenames_batch[j],
                    'actual_label': actual_labels_batch[j].item(),
                    'predicted_label': predicted_labels_batch[j].item(),
                    'is_correct': (actual_labels_batch[j] == predicted_labels_batch[j]).item(),
                    
                    'confidence_distance': all_k_distances[0],
                    'top_k_distances': all_k_distances,
                    
                    'top_k_neighbor_filenames': k_neighbor_filenames, 
                    
                    'top_k_neighbor_labels': neighbor_labels_list
                })

        all_predictions.append(predicted_labels_batch)
        all_actuals.append(actual_labels_batch)

    if not all_predictions:
        if return_raw_preds:
            return (torch.tensor([], dtype=torch.long), 
                    torch.tensor([], dtype=torch.long), 
                    [])
        else:
            print("Warning: No predictions were made.")
            return 0.0, []

    predictions_tensor = torch.cat(all_predictions)
    actuals_tensor = torch.cat(all_actuals)

    if return_raw_preds:
        return predictions_tensor, actuals_tensor, prediction_details

    predictions_numpy = predictions_tensor.cpu().numpy()
    actuals_numpy = actuals_tensor.cpu().numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        accuracy = accuracy_score(actuals_numpy, predictions_numpy)
        balanced_accuracy = balanced_accuracy_score(actuals_numpy, predictions_numpy)

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

    unique_labels = sorted(list(set(db_labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    db_labels_int = torch.tensor([label_to_id[s] for s in db_labels], dtype=torch.long, device=device)

    
    db_encounters = [parse_encounter_id(v) for v in db_videos]
    unique_encounters = sorted(list(set(db_encounters)))
    encounter_to_id = {enc: i for i, enc in enumerate(unique_encounters)}
    db_encounter_ids_int = torch.tensor([encounter_to_id[s] for s in db_encounters], dtype=torch.long, device=device)


    db_embeddings_gpu = db_embeddings.to(device)
    
    query_embeddings = None
    query_labels_int = None
    query_encounter_ids_int = None
    query_original_indices = None 

    query_filenames = None
    if query_dataset is None:
        if db_filenames is not None:
            query_filenames = [db_filenames[i] for i in query_indices_in_db]
    else:
        local_query_indices = query_dataset.images_for_ce_knn
        if db_filenames is not None:
            query_filenames = [query_dataset.filenames[i] for i in local_query_indices]

    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
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
        query_filenames=query_filenames,
        db_filenames=db_filenames
    )

    analyze_predictions_by_class(prediction_details, id_to_label, cfg, n=50)
    
    return mean_accuracy


def main(cfg):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper, image_transforms, _ = get_model_wrapper(device=DEVICE, cfg=cfg["model"])



    dataset_dir = cfg["data"]["dataset_dir"]
    is_zoo = "zoo" in dataset_dir.lower()

    if not is_zoo:
        print("Using standard dataset with train/analysis splits.")
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
        
        datasets = [split_dataset, train_dataset]
        full_db_dataset = ConcatDataset(datasets)
        full_dataset_splits = "+".join([os.path.basename(d.image_dir) for d in datasets])
        
        print(f"Full database contains {len(full_db_dataset)} images from splits: {full_dataset_splits}")

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

    else:
        print("Using Zoo dataset for evaluation.")
        all_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith((".jpg", ".png"))]

        subsample_fraction = cfg["data"].get("zoo_subsample_fraction", 1.0)

        if subsample_fraction < 1.0:
            print(f"Subsampling Zoo dataset to {subsample_fraction:.0%} of its original size.")
            labels = [f.split('_')[0] for f in all_files]
            subsampled_files, _ = train_test_split(
                all_files,
                test_size=(1.0 - subsample_fraction),
                stratify=labels,
                random_state=cfg["seed"]
            )
            split_name_suffix = f"_subsampled_{int(subsample_fraction*100)}pct"
        else:
            print("Using the full Zoo dataset (no subsampling).")
            subsampled_files = all_files
            split_name_suffix = "_full"
        
        print(f"Using {len(subsampled_files)} images for the Zoo evaluation.")

        split_dataset = GorillaReIDDataset(
            image_dir=dataset_dir,
            filenames=subsampled_files,
            transform=image_transforms,
            k=cfg["knn"]["k"],
        )



        # For the zoo dataset, the full DB is just the split_dataset itself.
        # These variables are defined to match the structure of the other branch.
        query_dataset_offset = 0
        full_db_dataset = split_dataset
        full_dataset_splits = os.path.basename(dataset_dir) + split_name_suffix
        split_name = full_dataset_splits
        all_db_filenames = split_dataset.filenames

    
    local_query_indices = split_dataset.images_for_ce_knn
    global_query_indices = [idx + query_dataset_offset for idx in local_query_indices]

    db_path = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name=split_dataset.dataset_name,
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

    evaluate_model(
        model_wrapper=model_wrapper,
        query_indices_in_db=global_query_indices, 
        cfg=cfg,
        device=DEVICE,
        db_embeddings=db_embeddings,
        db_labels=db_labels,
        db_videos=db_videos,
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