import os
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import random
from PIL import Image

def get_class_label(filename: str) -> str:
    """
    Extracts the class label from a filename.
    Assumes the filename format is 'classlabel_some_other_info.png'.
    """
    return filename.split("_")[0]

def get_balanced_individual_splits(
    train_files: List[str],
    holdout_percentage: float
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Splits individuals into disjunct tune/holdout sets while balancing image counts.
    Returns:
        - tune_query_files
        - tune_db_files
        - holdout_query_files
        - holdout_db_files
    """

    # Group files by individual
    gorillas_to_files = defaultdict(list)
    for f in train_files:
        label = get_class_label(f)
        gorillas_to_files[label].append(f)

    # Get a list of all unique individuals
    individuals = list(gorillas_to_files.keys())
    random.shuffle(individuals) # Shuffle to ensure a random split

    # Calculate the split index based on the percentage of INDIVIDUALS
    num_individuals = len(individuals)
    split_idx = int(num_individuals * holdout_percentage)

    # Ensure we have at least one individual in the holdout set if possible
    if num_individuals > 1 and split_idx == 0:
        split_idx = 1
        
    # Ensure we have at least one individual in the tune set if possible
    if num_individuals > 1 and split_idx == num_individuals:
        split_idx = num_individuals - 1

    # Split the individuals into holdout and tune sets
    holdout_inds = individuals[:split_idx]
    tune_inds = individuals[split_idx:]

    # Check sanity
    assert set(tune_inds).isdisjoint(holdout_inds), "Individuals must be disjunct"

    # Construct query + DB files
    tune_query_files = [
        random.choice(gorillas_to_files[ind])
        for ind in tune_inds
        if len(gorillas_to_files[ind]) > 1
    ]
    holdout_query_files = [
        random.choice(gorillas_to_files[ind])
        for ind in holdout_inds
        if len(gorillas_to_files[ind]) > 1
    ]

    tune_db_files = [f for ind in tune_inds for f in gorillas_to_files[ind]]
    holdout_db_files = [f for ind in holdout_inds for f in gorillas_to_files[ind]]

    print(f"Tune: {len(tune_inds)} individuals, {len(tune_db_files)} images in db, {len(tune_query_files)} images in queries")
    print(f"Holdout: {len(holdout_inds)} individuals, {len(holdout_db_files)} images in db, {len(holdout_query_files)} images in queries")

    assert len(tune_query_files) <= len(tune_db_files), "Tune query files must be a subset of DB files"
    assert len(holdout_query_files) <= len(holdout_db_files), "Holdout query files must be a subset of DB files"
    assert len(tune_query_files) != 0, "Tune query files cannot be empty"
    assert len(holdout_query_files) != 0, "Holdout query files cannot be empty"


    return tune_query_files, tune_db_files, holdout_query_files, holdout_db_files


def load_all_configs(config_dir: str):
    config = {}
    for file in Path(config_dir).glob("*.yaml"):
        key = file.stem  
        with open(file, "r") as f:
            config[key] = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    data_path = "/workspaces/vast-gorilla/gorillawatch/data/eval_body_squared_cleaned_open_2024_bigval/train"
    files = [f for f in os.listdir(data_path) if f.lower().endswith((".jpg", ".png"))]
    tune_query_files, all_tune_files, holdout_query_files, all_holdout_files = get_balanced_individual_splits(
        train_files=files,
        holdout_percentage=0.2
    )
