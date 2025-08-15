import os
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import random
from PIL import Image
from omegaconf import OmegaConf

from dataset import GorillaReIDDataset


def get_class_label(filename: str) -> str:
    """
    Extracts the class label from a filename.
    Assumes the filename format is 'classlabel_some_other_info.png'.
    """
    return filename.split("_")[0]

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

def get_db_path(model_checkpoint_path: str, dataset: GorillaReIDDataset, split_name: str, db_dir: str) -> str:
    checkpoint_name = os.path.splitext(os.path.basename(model_checkpoint_path))[0]
    db_filename = f"{checkpoint_name}_{dataset.dataset_name}_{split_name}_db.pt"
    db_path = os.path.join(db_dir, db_filename)
    return db_path


if __name__ == "__main__":
    data_path = "/workspaces/vast-gorilla/gorillawatch/data/eval_body_squared_cleaned_open_2024_bigval/train"
    files = [f for f in os.listdir(data_path) if f.lower().endswith((".jpg", ".png"))]
    tune_query_files, all_tune_files, holdout_query_files, all_holdout_files = get_balanced_individual_splits(
        train_files=files,
        holdout_percentage=0.2
    )
