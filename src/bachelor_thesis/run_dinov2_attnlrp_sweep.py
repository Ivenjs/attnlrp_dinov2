import torch
from lxt.efficient import monkey_patch_zennit
import pandas as pd
import yaml
import os
from PIL import Image
from pathlib import Path
import random
from collections import defaultdict
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from basemodel import get_model_wrapper
from dinov2_attnlrp_sweep import (
    run_gamma_sweep, 
    evaluate_gamma_sweep, 
    print_robustness_summary, 
    visualize_robustness_analysis, 
    find_robust_hyperparameters,
    log_nested_validation_to_wandb
    )
from knn_helpers import get_knn_db
from dataset import GorillaReIDDataset, custom_collate_fn
from utils import get_balanced_individual_splits, load_all_configs



if __name__ == "__main__":
    monkey_patch_zennit(verbose=True)  # is this needed? seems to be

    LOG_TO_WANDB = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VERBOSE = False  
    random.seed(27)  
    torch.manual_seed(27)  

    model_wrapper, image_transforms, _ = get_model_wrapper(device=DEVICE, finetuned=True)

    config_dir = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/configs"
    cfg = load_all_configs(config_dir)
    MODE = cfg["lrp"]["mode"]


    root_dir = cfg["data"]["dataset_dir"]
    train_dir = os.path.join(root_dir, "train")

    train_files = [f for f in os.listdir(train_dir) if f.lower().endswith((".jpg", ".png"))]

    tune_query_files, all_tune_files, holdout_query_files, all_holdout_files = get_balanced_individual_splits(
        train_files=train_files,
        holdout_percentage=cfg["sweep"]["holdout_percentage"]
    )
    
    tune_query_dataset = GorillaReIDDataset(
        image_dir=train_dir, filenames=tune_query_files, transform=image_transforms
    )
    tune_db_dataset = GorillaReIDDataset(
        image_dir=train_dir, filenames=all_tune_files, transform=image_transforms
    )

    holdout_query_dataset = GorillaReIDDataset(
        image_dir=train_dir, filenames=holdout_query_files, transform=image_transforms
    )
    holdout_db_dataset = GorillaReIDDataset(
        image_dir=train_dir, filenames=all_holdout_files, transform=image_transforms
    )

    print(f"Tune query size: {len(tune_query_dataset)}, Holdout query size: {len(holdout_query_dataset)}")
    print(f"Tune DB size: {len(tune_db_dataset)}, Holdout DB size: {len(holdout_db_dataset)}")

    # This phase generates all necessary data for the 'tune' set.
    print("\n--- RUNNING FULL SWEEP ON TUNE SET ---")
    tune_db_embeddings, tune_db_labels, tune_db_filenames = get_knn_db(
        db_dir=cfg["knn"]["db_embeddings_dir"], split_name="explainer_tune", dataset=tune_db_dataset,
        model_wrapper=model_wrapper, model_checkpoint_path=cfg["model"]["checkpoint_path"], batch_size=cfg["data"]["batch_size"], device=DEVICE
    )
    tune_dataloader = DataLoader(tune_query_dataset, batch_size=cfg["data"]["batch_size"], num_workers=4, collate_fn=custom_collate_fn,shuffle=False)

    tune_relevances, tune_violations = run_gamma_sweep(
        model_wrapper, tune_dataloader, DEVICE, MODE, tune_db_embeddings, tune_db_filenames, tune_db_labels,
        cfg["knn"]["k"], cfg["sweep"]["distance_metrics"], cfg["sweep"]["conv_gammas"], cfg["sweep"]["lin_gammas"], VERBOSE
    )
    tune_eval_dataloader = DataLoader(tune_query_dataset, batch_size=1, num_workers=4, collate_fn=custom_collate_fn)
    tune_results_list, tune_curves_list = evaluate_gamma_sweep(
        tune_relevances, tune_violations, tune_eval_dataloader, model_wrapper,
        tune_db_embeddings, tune_db_labels, tune_db_filenames, cfg["model"]["patch_size"], DEVICE,
        cfg["knn"]["k"], cfg["eval"]["patches_per_step"], VERBOSE
    )

    # --- GENERATE RESULTS FOR HOLDOUT SET ---
    # This phase generates all necessary data for the 'holdout' set.
    print("\n--- RUNNING FULL SWEEP ON HOLDOUT SET ---")
    holdout_db_embeddings, holdout_db_labels, holdout_db_filenames = get_knn_db(
        db_dir=cfg["knn"]["db_embeddings_dir"], split_name="explainer_holdout", dataset=holdout_db_dataset,
        model_wrapper=model_wrapper, model_checkpoint_path=cfg["model"]["checkpoint_path"], batch_size=cfg["data"]["batch_size"], device=DEVICE
    )

    holdout_dataloader = DataLoader(holdout_query_dataset, batch_size=cfg["data"]["batch_size"], num_workers=4, collate_fn=custom_collate_fn, shuffle=False)
    holdout_relevances, holdout_violations = run_gamma_sweep(
        model_wrapper, holdout_dataloader, DEVICE, MODE, holdout_db_embeddings, holdout_db_filenames, holdout_db_labels,
        cfg["knn"]["k"], cfg["sweep"]["distance_metrics"], cfg["sweep"]["conv_gammas"], cfg["sweep"]["lin_gammas"], VERBOSE
    )
    holdout_eval_dataloader = DataLoader(holdout_query_dataset, batch_size=1, num_workers=4, collate_fn=custom_collate_fn)
    holdout_results_list, holdout_curves_list = evaluate_gamma_sweep(
        holdout_relevances, holdout_violations, holdout_eval_dataloader, model_wrapper,
        holdout_db_embeddings, holdout_db_labels, holdout_db_filenames, cfg["model"]["patch_size"], DEVICE,
        cfg["knn"]["k"], cfg["eval"]["patches_per_step"], VERBOSE
    )

    # --- PHASE 3: SEQUENTIAL ANALYSIS & DECISION MAKING ---
    print("\n" + "="*80)
    print("--- PHASE 3: SEQUENTIAL ANALYSIS & DECISION MAKING ---")
    print("="*80)

    # Step 3a: Select BEST parameters using ONLY the TUNE set results.
    print("\nFinding best parameters on TUNE set...")
    best_params_raw_tune, best_params_norm_tune, analysis_df_tune = find_robust_hyperparameters(
        results=tune_results_list,
        robustness_percentile=cfg["sweep"]["robustness_percentile"],
        min_score_threshold=cfg["sweep"]["min_score_threshold"]
    )
    
    print_robustness_summary(best_params_raw_tune, best_params_norm_tune, analysis_df_tune)

    # Step 3b: Evaluate these BEST parameters on the HOLDOUT set for an unbiased performance estimate.
    _, _, analysis_df_holdout = find_robust_hyperparameters(
        results=holdout_results_list,
        robustness_percentile=cfg["sweep"]["robustness_percentile"],
        min_score_threshold=cfg["sweep"]["min_score_threshold"]
    )

    # Step 3c: Analyze the generalization gap.
    print("\nLooking up performance of TUNE-selected parameters on the HOLDOUT set...")
    holdout_performance_row = analysis_df_holdout[
        (analysis_df_holdout['conv_gamma'] == best_params_raw_tune['conv_gamma']) &
        (analysis_df_holdout['lin_gamma'] == best_params_raw_tune['lin_gamma']) &
        (analysis_df_holdout['distance_metric'] == best_params_raw_tune['distance_metric'])
    ].iloc[0] # Get the first (and only) row as a Series

    # Extract performance stats from this row
    tune_performance = {
        'mean_faithfulness': best_params_raw_tune['mean_score'],
        'min_faithfulness': best_params_raw_tune['min_score'],
        'std_faithfulness': best_params_raw_tune['std_score'],
    }
    holdout_performance = {
        'mean_faithfulness': holdout_performance_row['raw_mean'],
        'min_faithfulness': holdout_performance_row['raw_min'],
        'std_faithfulness': holdout_performance_row['raw_std'],
    }

    # Step 3d: Analyze the generalization gap and make the final decision.
    # robustness score is only the decision metric (which params to choose), while the performance metrics are used for the final acceptance decision.
    print("\n" + "="*80)
    print("--- FINAL VALIDATION & DECISION ---")
    print("="*80)
    print(f"Chosen Parameters: conv_gamma={best_params_raw_tune['conv_gamma']}, lin_gamma={best_params_raw_tune['lin_gamma']}, dist={best_params_raw_tune['distance_metric']}")
    print(f"Performance on TUNE set:    Mean Faithfulness = {tune_performance['mean_faithfulness']:.4f}")
    print(f"Performance on HOLDOUT set: Mean Faithfulness = {holdout_performance['mean_faithfulness']:.4f}")

    generalization_drop = (tune_performance['mean_faithfulness'] - holdout_performance['mean_faithfulness'])
    relative_drop_percent = (generalization_drop / tune_performance['mean_faithfulness']) * 100 if tune_performance['mean_faithfulness'] != 0 else 0

    print(f"\nGeneralization Drop: {relative_drop_percent:.2f}%")

    ACCEPTABLE_DROP_PERCENT = 15.0
    if relative_drop_percent > ACCEPTABLE_DROP_PERCENT:
        print(f"DECISION: REJECT. High generalization drop.")
        FINAL_DECISION = "REJECTED"
    else:
        print(f"DECISION: APPROVE. Generalization is within acceptable limits.")
        FINAL_DECISION = "APPROVED"
        
        # --- PHASE 4: REPORTING & LOGGING ---
        print("\n" + "="*80)
        print("--- PHASE 4: FINAL REPORTING ---")
        print("="*80)
        
        # The 'best_raw' for logging is the one selected from the TUNE set.
        # The 'aggregate_stats' for logging should be the performance on the HOLDOUT set.
        final_official_params = best_params_raw_tune
        final_official_performance = holdout_performance

        print("\nOfficial Approved LRP Hyperparameters:")
        print(final_official_params)
        print("\nOfficial Unbiased Performance Metrics (from Holdout set):")
        print(final_official_performance)


    if LOG_TO_WANDB:
        log_nested_validation_to_wandb (
            cfg=cfg,
            final_decision=FINAL_DECISION,
            approved_params=best_params_raw_tune,
            tune_performance=tune_performance,
            holdout_performance=holdout_performance,
            generalization_drop_percent=relative_drop_percent,
            analysis_df_tune=analysis_df_tune,
            analysis_df_holdout=analysis_df_holdout,
            tune_results_list=tune_results_list,
            holdout_results_list=holdout_results_list,
            tune_curves_list=tune_curves_list,
            holdout_curves_list=holdout_curves_list
        )
