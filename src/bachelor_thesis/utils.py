import os
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import random

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

    # Sort individuals by number of images (desc)
    individuals = sorted(gorillas_to_files.keys(), key=lambda x: -len(gorillas_to_files[x]))

    # Initialize containers
    tune_inds, holdout_inds = [], []
    tune_count, holdout_count = 0, 0
    total_images = sum(len(files) for files in gorillas_to_files.values())
    target_holdout = int(total_images * holdout_percentage)

    # Greedy assign individuals to keep image counts balanced
    for ind in individuals:
        files = gorillas_to_files[ind]
        if holdout_count + len(files) <= target_holdout:
            holdout_inds.append(ind)
            holdout_count += len(files)
        else:
            tune_inds.append(ind)
            tune_count += len(files)

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

    print(f"Tune: {len(tune_inds)} individuals, {len(tune_db_files)} images, {len(tune_query_files)} queries")
    print(f"Holdout: {len(holdout_inds)} individuals, {len(holdout_db_files)} images, {len(holdout_query_files)} queries")

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
    data_path = "/sc/home/iven.schlegelmilch/bachelor_thesis_code/sample_images/train"
    files = [f for f in os.listdir(data_path) if f.lower().endswith((".jpg", ".png"))]
    tune_query_files, all_tune_files, holdout_query_files, all_holdout_files = get_balanced_individual_splits(
        train_files=files,
        holdout_percentage=0.2
    )
