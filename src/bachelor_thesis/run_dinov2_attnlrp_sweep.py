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

#TODO: use dataset class?

def get_image_path_map(image_dir: str):
    """
    Get one image for each class from the given directory.
    Assumes images are named in the format 'class_..._*.png'.
    """
    image_path_map = {}
    class_images = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            class_label = filename.split("_")[0]
            if class_label not in class_images:
                image_path_map[filename.split(".")[0]] = os.path.join(image_dir, filename)
                class_images.append(class_label)
    return image_path_map

def load_all_configs(config_dir: str):
    config = {}
    for file in Path(config_dir).glob("*.yaml"):
        key = file.stem  
        with open(file, "r") as f:
            config[key] = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    monkey_patch_zennit(verbose=True)  # is this needed? seems to be

    SAVE_HEATMAPS = True 
    LOG_TO_WANDB = True  
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODE = "knn"  # "simple" or "knn"


    model_wrapper, transforms = get_model_wrapper()

    config_dir = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/configs"
    cfg = load_all_configs(config_dir)


    images_dir = cfg["data"]["dataset_dir"]

    db_embeddings, db_filenames = get_knn_db(
        knn_db_dir=cfg["knn"]["db_embeddings_dir"],
        image_dir=images_dir,
        model_wrapper=model_wrapper,
        transforms=transforms,
        device=DEVICE
    )

    image_path_map = get_image_path_map(image_dir=images_dir)
    input_filenames = list(image_path_map.keys())
    input_tensors = []
    for filename in input_filenames:
        img_path = image_path_map[filename]
        img = Image.open(img_path).convert("RGB")
        input_tensor = transforms(img)
        input_tensors.append(input_tensor)
    input_tensors = torch.stack(input_tensors).to(DEVICE)

    relevances_by_parameters, violations_by_parameters = run_gamma_sweep(
        model_wrapper=model_wrapper,
        input_tensors=input_tensors,
        mode=MODE,
        db_embeddings=db_embeddings,
        db_filenames=db_filenames,
        input_filenames=input_filenames,
        k_neighbors=cfg["knn"]["k"],
        distance_metrics=cfg["sweep"]["distance_metrics"],
        conv_gamma_values=cfg["sweep"]["conv_gammas"],
        lin_gamma_values=cfg["sweep"]["lin_gammas"]
    )
    if SAVE_HEATMAPS:
        for input_filename in input_filenames:
            visualize_relevances(
                relevances=relevances_by_parameters[input_filename], 
                mode=MODE, 
                image_name=input_filename, 
                dim=(len(cfg["sweep"]["conv_gammas"]), len(cfg["sweep"]["lin_gammas"]))
            )
    
    # eval
    results, all_curves_data = evaluate_gamma_sweep(
        relevances_by_parameters=relevances_by_parameters,
        violations_by_parameters=violations_by_parameters,
        input_tensors=input_tensors,
        input_filenames=input_filenames,
        model_wrapper=model_wrapper,
        db_embeddings=db_embeddings,
        db_filenames=db_filenames,
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
    visualize_robustness_analysis(analysis_df)
    
    if LOG_TO_WANDB:
        run = wandb.init(
            project="Thesis-Iven", 
            entity="gorillawatch", 
            name="attnlrp_gamma_sweep",  
            config=cfg 
        )
        log_sweep_to_wandb(
            results=results,
            analysis_df=analysis_df,
            all_curves_data=all_curves_data,
            best_params_raw=best_raw,
            best_params_normalized=best_norm,
            aggregate_stats=aggregate_stats
        )

        run.finish()

