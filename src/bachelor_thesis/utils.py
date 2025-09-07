import os
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import random
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
import torch

from torch.utils.data import Dataset

import hashlib

def deterministic_randperm(num_patches, key, global_seed=161):
    h = int(hashlib.sha1((str(key) + str(global_seed)).encode()).hexdigest(), 16) % (2**32)
    g = torch.Generator()
    g.manual_seed(h)
    return torch.randperm(num_patches, generator=g)

def get_denormalization_transform(mean: tuple, std: tuple) -> transforms.Compose:
    """Creates a transform to de-normalize image tensors using a lambda function."""
    mean_tensor = torch.tensor(mean)
    std_tensor = torch.tensor(std)

    return transforms.Compose([
        # Reshape to (C, 1, 1) to work with image tensors (C, H, W)
        transforms.Lambda(lambda x: x * std_tensor.view(3, 1, 1)),
        transforms.Lambda(lambda x: x + mean_tensor.view(3, 1, 1)),
    ])

def get_mask_transform(img_size):
    return transforms.Compose([
        transforms.Resize(
            size=img_size,
            interpolation=transforms.InterpolationMode.NEAREST
        ),
        transforms.ToTensor(),
    ])

def get_class_label(filename: str) -> str:
    """
    Extracts the class label from a filename.
    Assumes the filename format is 'classlabel_some_other_info.png'.
    """
    return filename.split("_")[0]

def parse_encounter_id(video_id: str) -> Tuple[str, str]:
    """
    Parses a video ID like 'R105_20230201_263' into camera and date.
    
    Returns:
        A tuple (camera, date) or (None, None) if parsing fails.
    """
    if not isinstance(video_id, str):
        return None, None
    try:
        parts = video_id.split('_')
        if len(parts) >= 2:
            return parts[0], parts[1]  # (camera, date)
    except IndexError:
        pass
    return None, None

def get_balanced_individual_splits_cross_encounter(
    train_files: List[str],
    holdout_percentage: float,
    queries_per_class: int = 3
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Splits individuals into disjunct tune/holdout sets, ensuring that all
    selected query images are "cross-encounter eligible."

    An image is cross-encounter eligible if the same individual has at least one
    other image taken at a different encounter (different camera or different day).
    This guarantees a valid positive match exists when using cross-encounter filtering.

    Args:
        train_files: List of all image file paths.
        holdout_percentage: Fraction of individuals to put in the holdout set.
        queries_per_class: Max number of query images to select per individual.

    Returns:
        A tuple containing:
        - tune_query_files: List of query file paths for the tuning set.
        - tune_db_files: List of all database file paths for the tuning set.
        - holdout_query_files: List of query file paths for the holdout set.
        - holdout_db_files: List of all database file paths for the holdout set.
    """
    # 1. Group files by individual and then by encounter
    # Structure: {gorilla_id: {encounter_id: [file1, file2, ...]}}
    individuals_to_encounters = defaultdict(lambda: defaultdict(list))
    for f in train_files:
        label = get_class_label(f)
        video_id = f.split('_')[1] + "_" + f.split('_')[2] + "_" + f.split('_')[3]
        encounter_id = parse_encounter_id(video_id)
        if label and encounter_id[0] is not None:
            individuals_to_encounters[label][encounter_id].append(f)

    # 2. Get a list of all unique individuals and shuffle for a random split
    all_individuals = list(individuals_to_encounters.keys())
    random.shuffle(all_individuals)

    # 3. Split individuals into tune and holdout sets
    num_individuals = len(all_individuals)
    split_idx = int(num_individuals * holdout_percentage)

    # Edge case handling: ensure both sets are non-empty if possible
    if num_individuals > 1 and split_idx == 0:
        split_idx = 1
    if num_individuals > 1 and split_idx == num_individuals:
        split_idx = num_individuals - 1

    holdout_inds = all_individuals[:split_idx]
    tune_inds = all_individuals[split_idx:]
    
    assert set(tune_inds).isdisjoint(holdout_inds), "Individuals must be disjunct"

    # 4. Helper function to select cross-encounter eligible queries
    def _select_eligible_queries(
        individuals: List[str],
        data: Dict[str, Dict[Tuple[str, str], List[str]]]
    ) -> List[str]:
        query_files = []
        eligible_inds_count = 0
        
        for ind in individuals:
            encounters_map = data[ind]
            
            # An individual is eligible only if they appear in more than one encounter
            if len(encounters_map) < 2:
                print(f"    - Individual {ind} skipped (only {len(encounters_map)} encounter).")
                continue # Skip this individual, they cannot have a cross-encounter query
            
            eligible_inds_count += 1
            
            all_files_for_ind = [f for files in encounters_map.values() for f in files]
            if len(all_files_for_ind) <= 1:
                continue # Cannot select a query and leave one in the db
            # Sort encounters by the number of images they have, smallest first
            sorted_encounters = sorted(encounters_map.items(), key=lambda item: len(item[1]))
            
            potential_queries = []
            # Take all images from the smallest encounters first, as they make the best queries
            for encounter_id, files in sorted_encounters:
                potential_queries.extend(files)

            random.shuffle(potential_queries)
            
            # Select up to `queries_per_class`, ensuring at least one image is left in the DB overall.
            n_queries = min(queries_per_class, len(all_files_for_ind) - 1)
            query_files.extend(potential_queries[:n_queries])
                
        print(f"    - Selected queries from {eligible_inds_count}/{len(individuals)} eligible individuals.")
        return query_files

    # 5. Select queries for tune and holdout sets using the new logic
    print("Selecting queries for Tune set:")
    tune_query_files = _select_eligible_queries(tune_inds, individuals_to_encounters)
    
    print("Selecting queries for Holdout set:")
    holdout_query_files = _select_eligible_queries(holdout_inds, individuals_to_encounters)

    # 6. Create database files (all files for the individuals in each set)
    tune_db_files = [
        f for ind in tune_inds
        for encounter_files in individuals_to_encounters[ind].values()
        for f in encounter_files
    ]
    holdout_db_files = [
        f for ind in holdout_inds
        for encounter_files in individuals_to_encounters[ind].values()
        for f in encounter_files
    ]

    # 7. Final summary and assertions
    print("\n--- Split Summary ---")
    print(f"Tune Set: {len(tune_inds)} individuals, {len(tune_db_files)} DB images, {len(tune_query_files)} queries")
    print(f"Holdout Set: {len(holdout_inds)} individuals, {len(holdout_db_files)} DB images, {len(holdout_query_files)} queries")

    assert set(tune_query_files).issubset(set(tune_db_files))
    assert set(holdout_query_files).issubset(set(holdout_db_files))
    
    # It's possible for a set to have no eligible queries, so we handle this gracefully
    if not holdout_query_files and len(holdout_inds) > 0:
        print("\nWarning: Holdout set has individuals but no cross-encounter eligible queries were found.")
    if not tune_query_files and len(tune_inds) > 0:
        print("\nWarning: Tune set has individuals but no cross-encounter eligible queries were found.")
        
    return tune_query_files, tune_db_files, holdout_query_files, holdout_db_files

#TODO: make this deliberately cross video to enable more diverse queries
def get_balanced_individual_splits(
    train_files: List[str],
    holdout_percentage: float,
    queries_per_class: int = 3
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Splits individuals into disjunct tune/holdout sets while balancing image counts.
    Allows multiple query images per individual.
    
    Args:
        train_files: list of all image file paths
        holdout_percentage: fraction of individuals to put in holdout
        queries_per_class: number of query images to select per individual (if available)
    
    Returns:
        tune_query_files, tune_db_files, holdout_query_files, holdout_db_files
    """
    
    # Group files by individual
    gorillas_to_files = defaultdict(list)
    for f in train_files:
        label = get_class_label(f)
        gorillas_to_files[label].append(f)

    # Shuffle file order within each individual to avoid bias
    for files in gorillas_to_files.values():
        random.shuffle(files)

    # Get a list of all unique individuals
    individuals = list(gorillas_to_files.keys())
    random.shuffle(individuals)  # Shuffle to ensure a random split

    # Calculate the split index
    num_individuals = len(individuals)
    split_idx = int(num_individuals * holdout_percentage)

    if num_individuals > 1 and split_idx == 0:
        split_idx = 1
    if num_individuals > 1 and split_idx == num_individuals:
        split_idx = num_individuals - 1

    holdout_inds = individuals[:split_idx]
    tune_inds = individuals[split_idx:]

    assert set(tune_inds).isdisjoint(holdout_inds), "Individuals must be disjunct"

    # Select queries
    tune_query_files = []
    for ind in tune_inds:
        files = gorillas_to_files[ind]
        if len(files) > 1:
            n_queries = min(queries_per_class, len(files) - 1)  # leave at least 1 for db
            tune_query_files.extend(random.sample(files, n_queries))

    holdout_query_files = []
    for ind in holdout_inds:
        files = gorillas_to_files[ind]
        if len(files) > 1:
            n_queries = min(queries_per_class, len(files) - 1)
            holdout_query_files.extend(random.sample(files, n_queries))

    # DB = all files for that individual
    tune_db_files = [f for ind in tune_inds for f in gorillas_to_files[ind]]
    holdout_db_files = [f for ind in holdout_inds for f in gorillas_to_files[ind]]

    print(f"Tune: {len(tune_inds)} inds, {len(tune_db_files)} db imgs, {len(tune_query_files)} queries")
    print(f"Holdout: {len(holdout_inds)} inds, {len(holdout_db_files)} db imgs, {len(holdout_query_files)} queries")

    assert set(tune_query_files).issubset(tune_db_files)
    assert set(holdout_query_files).issubset(holdout_db_files)
    assert tune_query_files, "Tune query files cannot be empty"
    assert holdout_query_files, "Holdout query files cannot be empty"

    return tune_query_files, tune_db_files, holdout_query_files, holdout_db_files


def load_config(config_name: str, cli_overrides: list):
    """
    Loads base config, merges the experiment-specific override,
    and finally merges any command-line overrides.
    """
    config_dir = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/configs"

    # Load base configuration
    base_config = OmegaConf.load(os.path.join(config_dir, "base.yaml"))

    # Load experiment-specific override
    override_path = os.path.join(config_dir, "experiment", f"{config_name}.yaml")
    if not os.path.exists(override_path):
        raise FileNotFoundError(f"Experiment config file not found: {override_path}")
    override_config = OmegaConf.load(override_path)

    # Load command-line overrides passed from sbatch
    cli_config = OmegaConf.from_cli(cli_overrides)

    config = OmegaConf.merge(base_config, override_config, cli_config)
    
    return config

def get_db_path(model_checkpoint_path: str, dataset_name: str, split_name: str, db_dir: str, decision_metric: str=None, lrp_params: Dict= {}) -> str:
    checkpoint_name = os.path.splitext(os.path.basename(model_checkpoint_path))[0]

    lrp_str = "_".join(
        f"{param}_{str(value).replace('.', 'p')}"
        for param, value in lrp_params.items()
        if value is not None
    )
    db_filename = f"{checkpoint_name}_{dataset_name}_{split_name}{'_' + decision_metric if decision_metric else ''}{'_' + lrp_str if lrp_params else ''}_db.pt"

    db_path = os.path.join(db_dir, db_filename)
    return db_path

def get_hpi_colors(cfg: Dict) -> Dict:
    """
    returns hpi colors in normalized RGB format
    """
    return {
        "red": tuple(int(c) / 255 for c in cfg["plots"]["red"]),
        "orange": tuple(int(c) / 255 for c in cfg["plots"]["orange"]),
        "yellow": tuple(int(c) / 255 for c in cfg["plots"]["yellow"]),
        "gray": tuple(int(c) / 255 for c in cfg["plots"]["gray"]),
    }

if __name__ == "__main__":
    data_path = "/workspaces/vast-gorilla/gorillawatch/data3/stratified_open_split_NEW/spac23+24-body_face-squared-deduplicated/train"
    files = [f for f in os.listdir(data_path) if f.lower().endswith((".jpg", ".png"))]
    tune_query_files, all_tune_files, holdout_query_files, all_holdout_files = get_balanced_individual_splits_cross_encounter(
        train_files=files,
        holdout_percentage=0.2
    )

