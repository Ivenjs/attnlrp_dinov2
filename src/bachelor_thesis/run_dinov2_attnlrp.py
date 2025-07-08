import torch
from lxt.efficient import monkey_patch_zennit
import pandas as pd
import yaml
import os
from PIL import Image

from basemodel import get_model_wrapper
from dinov2_attnlrp_sweep import run_gamma_sweep, evaluate_gamma_sweep, print_robustness_summary, visualize_robustness_analysis, find_robust_hyperparameters
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


if __name__ == "__main__":
    monkey_patch_zennit(verbose=True)  # is this needed? seems to be


    SAVE_HEATMAPS = False 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODE = "knn"  # "simple" or "knn"

    model_wrapper, transforms = get_model_wrapper()

    # 3. Prepare your input image
    image_path = "/workspaces/bachelor_thesis_code/sample_images/YE41_R035_20220818_091_1842_799985.png"

    model_config_path = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/configs/model_config.yaml"
    with open(model_config_path, "r") as f:
        cfg = yaml.safe_load(f)
    conv_gammas = cfg["CONV_GAMMAS"]
    lin_gammas = cfg["LIN_GAMMAS"]

    ground_truth_label = os.path.basename(image_path).split("_")[0]
    image = Image.open(image_path).convert(
        "RGB"
    )
    input_tensor = transforms(image).unsqueeze(0).to(DEVICE)

    images_dir = "/workspaces/vast-gorilla/gorillawatch/data/eval_body_squared_cleaned_open_2024_bigval/train"

    db_embeddings, db_labels = get_knn_db(
        knn_db_dir="/workspaces/bachelor_thesis_code/knn_db",
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
        k_neighbors=5,
        conv_gamma_values=conv_gammas,
        lin_gamma_values=lin_gammas
    )
    if SAVE_HEATMAPS:
        visualize_relevances(
            relevances=relevances_by_gamma, 
            mode=MODE, 
            image_name=os.path.basename(image_path).split(".")[0], 
            dim=(len(conv_gammas), len(lin_gammas))
        )
    # eval

    PATCH_SIZE = cfg["patch_size"]
 
    results = evaluate_gamma_sweep(
        relevances_by_gamma=relevances_by_gamma,
        violations_by_gamma=violations_by_gamma,
        input_tensors=input_tensors,
        ground_truth_labels=ground_truth_labels,
        model_wrapper=model_wrapper,
        db_embeddings=db_embeddings,
        db_labels=db_labels,
        patch_size=PATCH_SIZE,
        k_neighbors=5,
        plot_curves=False  # Set to True if you want individual curves
    )

    best_params, analysis_df = find_robust_hyperparameters(
        results=results,
        robustness_percentile=0.9,  # 90% of cases should be good
        min_score_threshold=0.0     # Adjust based on your score range
    )

    aggregate_stats = print_robustness_summary(best_params, analysis_df, results)
    
    # Create visualizations
    visualize_robustness_analysis(analysis_df, results)
    
    # Optionally save results
    pd.DataFrame(results).to_csv("/workspaces/bachelor_thesis_code/src/bachelor_thesis/robustness_analysis/detailed_results.csv", index=False)
    analysis_df.to_csv("/workspaces/bachelor_thesis_code/src/bachelor_thesis/robustness_analysis/robustness_analysis.csv", index=False)

    if aggregate_stats:
        aggregate_stats['by_gamma'].to_csv("/workspaces/bachelor_thesis_code/src/bachelor_thesis/robustness_analysis/gamma_statistics.csv", index=False)
        aggregate_stats['by_image'].to_csv("/workspaces/bachelor_thesis_code/src/bachelor_thesis/robustness_analysis/image_statistics.csv", index=False)
            

