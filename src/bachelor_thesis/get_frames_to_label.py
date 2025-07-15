import os
from collections import defaultdict
from glob import glob

def get_images_for_each_class(image_dir: str, num_images_per_class: int = 1) -> dict:
    """
    Recursively get up to `num_images_per_class` images for each class from the given directory.
    Assumes images are named in the format 'class_label_*.png'.
    Searches through all subdirectories (e.g., train, val, test).
    
    Returns:
        dict: Mapping from class label to list of image paths.
    """
    class_images = defaultdict(list)

    # Go through all png files recursively
    for filepath in glob(os.path.join(image_dir, "**", "*.png"), recursive=True):
        filename = os.path.basename(filepath)
        class_label = filename.split("_")[0]

        if len(class_images[class_label]) < num_images_per_class:
            class_images[class_label].append(filepath)

    return dict(class_images)

def main():
    model_config_path = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/configs/data.yaml"
    with open(model_config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    dataset_dir = cfg['dataset_dir']



if __name__ == "__main__":
    main()