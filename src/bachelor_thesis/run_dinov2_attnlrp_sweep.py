import torch
from lxt.efficient import monkey_patch_zennit
import pandas as pd
import yaml
import os
from PIL import Image
from pathlib import Path
import wandb


from basemodel import get_model_wrapper
from dinov2_attnlrp_sweep import (
    run_gamma_sweep, 
    evaluate_gamma_sweep, 
    print_robustness_summary, 
    visualize_robustness_analysis, 
    find_robust_hyperparameters,
    log_sweep_to_wandb  
)
from lrp_helpers import visualize_relevances
from knn_helpers import get_knn_db


def get_image_for_each_class(image_dir: str):
    """
    Get one image for each class from the given directory.
    Assumes images are named in the format 'class_label_*.png'.
    """
    class_images = {}
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            class_label = filename.split("_")[0]
            if class_label not in class_images:
                class_images[class_label] = os.path.join(image_dir, filename)
    return class_images

def load_all_configs(config_dir: str):
    config = {}
    for file in Path(config_dir).glob("*.yaml"):
        key = file.stem  
        with open(file, "r") as f:
            config[key] = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    monkey_patch_zennit(verbose=True)  # is this needed? seems to be

    SAVE_HEATMAPS = False 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODE = "knn"  # "simple" or "knn"

    model_wrapper, transforms = get_model_wrapper()

    config_dir = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/configs"
    cfg = load_all_configs(config_dir)

    run = wandb.init(
        project="Thesis-Iven", 
        entity="gorillawatch", 
        name="attnlrp_gamma_sweep",  
        config=cfg 
    )

    images_dir = cfg["data"]["dataset_dir"]

    db_embeddings, db_labels = get_knn_db(
        knn_db_dir=cfg["knn"]["db_embeddings_dir"],
        image_dir=images_dir,
        model_wrapper=model_wrapper,
        transforms=transforms,
        device=DEVICE
    )

    class_images = get_image_for_each_class(image_dir=images_dir)
    ground_truth_labels = list(class_images.keys())
    input_tensors = []
    for label in ground_truth_labels:
        img_path = class_images[label]
        img = Image.open(img_path).convert("RGB")
        input_tensor = transforms(img)
        input_tensors.append(input_tensor)
    input_tensors = torch.stack(input_tensors).to(DEVICE)

    relevances_by_gamma, violations_by_gamma = run_gamma_sweep(
        model_wrapper=model_wrapper,
        input_tensors=input_tensors,
        mode=MODE,
        db_embeddings=db_embeddings,
        db_labels=db_labels,
        ground_truth_labels=ground_truth_labels,
        k_neighbors=cfg["knn"]["k"],
        conv_gamma_values=cfg["sweep"]["conv_gammas"],
        lin_gamma_values=cfg["sweep"]["lin_gammas"]
    )
    if SAVE_HEATMAPS:
        visualize_relevances(
            relevances=relevances_by_gamma, 
            mode=MODE, 
            image_name=os.path.basename(image_path).split(".")[0], 
            dim=(len(conv_gammas), len(lin_gammas))
        )
    # eval

 
    results, all_curves_data = evaluate_gamma_sweep(
        relevances_by_gamma=relevances_by_gamma,
        violations_by_gamma=violations_by_gamma,
        input_tensors=input_tensors,
        ground_truth_labels=ground_truth_labels,
        model_wrapper=model_wrapper,
        db_embeddings=db_embeddings,
        db_labels=db_labels,
        patch_size=cfg["model"]["patch_size"],
        k_neighbors= cfg["knn"]["k"],
        plot_curves=True  # Set to True if you want individual curves
    )

    best_raw, best_norm, analysis_df = find_robust_hyperparameters(
        results=results,
        robustness_percentile=cfg["sweep"]["robustness_percentile"],  
        min_score_threshold=cfg["sweep"]["min_score_threshold"]    
    )

    aggregate_stats = print_robustness_summary(best_raw, best_norm, analysis_df, results)

    # Create visualizations
    visualize_robustness_analysis(analysis_df, results)
    
    log_sweep_to_wandb(
        results=results,
        analysis_df=analysis_df,
        all_curves_data=all_curves_data,
        best_params_raw=best_raw,
        best_params_normalized=best_norm,
        aggregate_stats=aggregate_stats,
        config=cfg
    )

    run.finish()

