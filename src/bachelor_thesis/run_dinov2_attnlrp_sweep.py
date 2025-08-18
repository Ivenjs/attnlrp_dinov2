from omegaconf import OmegaConf
import torch
from lxt.efficient import monkey_patch_zennit
import os
import argparse
import random
from collections import defaultdict
from torch.utils.data import DataLoader
import wandb

from basemodel import get_model_wrapper
from sweep_helpers import (
    evaluate_gamma_sweep, 
    print_robustness_summary, 
    find_robust_hyperparameters,
    log_nested_validation_to_wandb
    )
from knn_helpers import get_knn_db
from dataset import GorillaReIDDataset, custom_collate_fn
from utils import get_balanced_individual_splits, get_db_path, load_config
from lrp_helpers import generate_relevances



def main(cfg: dict):
    monkey_patch_zennit(verbose=True)  

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VERBOSE = False  
    random.seed(cfg["seed"])  
    torch.manual_seed(cfg["seed"])  


    MODE = cfg["lrp"]["mode"]
    DECISION_METRIC = MODE
    print(f"\n--- RUNNING WITH MODE: {MODE} ---")
    FINETUNED = cfg["model"]["finetuned"]    


    is_finetuned = cfg["model"]["finetuned"]
    model_type_str = "finetuned" if is_finetuned else "base"
    print(f"\n--- Running SWEEP for: {model_type_str.upper()} MODEL ---")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    run = wandb.init(
        project="Thesis-Iven", 
        entity="gorillawatch", 
        name=f"attnlrp_gamma_sweep_{model_type_str}_{MODE}",
        config=cfg_dict,
        job_type="analysis"
    )
    
    model_wrapper, image_transforms, _ = get_model_wrapper(device=DEVICE, cfg=cfg["model"])

    root_dir = cfg["data"]["dataset_dir"]
    train_dir = os.path.join(root_dir, "train")

    train_files = [f for f in os.listdir(train_dir) if f.lower().endswith((".jpg", ".png"))]

    tune_query_files, _, holdout_query_files, _ = get_balanced_individual_splits(
        train_files=train_files,
        holdout_percentage=cfg["sweep"]["holdout_percentage"],
        queries_per_class=cfg["sweep"]["queries_per_class"]
    )
    
    tune_query_dataset = GorillaReIDDataset(
        image_dir=train_dir, filenames=tune_query_files, transform=image_transforms
    )

    holdout_query_dataset = GorillaReIDDataset(
        image_dir=train_dir, filenames=holdout_query_files, transform=image_transforms
    )

    train_db_dataset = GorillaReIDDataset(
        image_dir=train_dir, filenames=train_files, transform=image_transforms
    )

    print(f"Tune query size: {len(tune_query_dataset)}, Holdout query size: {len(holdout_query_dataset)}")
    print(f"DB size: {len(train_db_dataset)}")

    # This phase generates all necessary data for the 'tune' set.
    print("\n--- RUNNING FULL SWEEP ON TUNE SET ---")

    train_db_path = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset=train_db_dataset,
        split_name="train",
        db_dir=cfg["knn"]["db_embeddings_dir"]
    )
    train_db_embeddings, train_db_labels, train_db_filenames, train_db_videos = get_knn_db(
        db_path=train_db_path,
        dataset=train_db_dataset,
        model_wrapper=model_wrapper,
        batch_size=cfg["data"]["batch_size"],
        device=DEVICE
    )

    tune_dataloader = DataLoader(tune_query_dataset, batch_size=cfg["data"]["batch_size"], num_workers=4, collate_fn=custom_collate_fn,shuffle=False)
    
    tune_relevances_all = generate_relevances(
        model_wrapper=model_wrapper,
        dataloader=tune_dataloader,
        device=DEVICE,
        conv_gamma_values=cfg["sweep"]["conv_gammas"],
        lin_gamma_values=cfg["sweep"]["lin_gammas"],
        mode=MODE,
        distance_metrics=cfg["sweep"]["distance_metrics"],
        proxy_temp_values=cfg["sweep"]["temp"],
        topk_neg_values=cfg["sweep"]["topk_neg"],
        db_embeddings=train_db_embeddings,
        db_filenames=train_db_filenames,
        db_labels=train_db_labels,
        db_video_ids=train_db_videos
    )

    tune_eval_dataloader = DataLoader(tune_query_dataset, batch_size=1, num_workers=4, collate_fn=custom_collate_fn)
    tune_results_list, tune_curves_list = evaluate_gamma_sweep(
        tune_relevances_all, tune_eval_dataloader, model_wrapper,
        train_db_embeddings, train_db_labels, train_db_filenames, train_db_videos,cfg["model"]["patch_size"], DEVICE,
        cfg["eval"]["patches_per_step"], cfg["eval"]["baseline_value"], False
    )

    # --- GENERATE RESULTS FOR HOLDOUT SET ---
    # This phase generates all necessary data for the 'holdout' set.
    print("\n--- RUNNING FULL SWEEP ON HOLDOUT SET ---")

    holdout_dataloader = DataLoader(holdout_query_dataset, batch_size=cfg["data"]["batch_size"], num_workers=4, collate_fn=custom_collate_fn, shuffle=False)
    holdout_relevances_all = generate_relevances(
        model_wrapper=model_wrapper,
        dataloader=holdout_dataloader,
        device=DEVICE,
        conv_gamma_values=cfg["sweep"]["conv_gammas"],
        lin_gamma_values=cfg["sweep"]["lin_gammas"],
        mode=MODE,
        distance_metrics=cfg["sweep"]["distance_metrics"],
        proxy_temp_values=cfg["sweep"]["temp"],
        topk_neg_values=cfg["sweep"]["topk_neg"],
        db_embeddings=train_db_embeddings,
        db_filenames=train_db_filenames,
        db_labels=train_db_labels,
        db_video_ids=train_db_videos,
    )

    holdout_eval_dataloader = DataLoader(holdout_query_dataset, batch_size=1, num_workers=4, collate_fn=custom_collate_fn)
    holdout_results_list, holdout_curves_list = evaluate_gamma_sweep(
        holdout_relevances_all, holdout_eval_dataloader, model_wrapper,
        train_db_embeddings, train_db_labels, train_db_filenames, train_db_videos, cfg["model"]["patch_size"], DEVICE,
        cfg["eval"]["patches_per_step"], cfg["eval"]["baseline_value"], False
    )

    # --- PHASE 3: SEQUENTIAL ANALYSIS & DECISION MAKING ---
    print("\n" + "="*80)
    print("--- PHASE 3: SEQUENTIAL ANALYSIS & DECISION MAKING ---")
    print("="*80)

    # Step 3a: Select BEST parameters using ONLY the TUNE set results.
    print("\nFinding best parameters on TUNE set...")
    best_params_raw_tune, analysis_df_tune, worst_params_raw_tune = find_robust_hyperparameters(
        results=tune_results_list,
        decision_metric=DECISION_METRIC
    )
    
    print_robustness_summary(best_params_raw_tune, analysis_df_tune, DECISION_METRIC)

    # Step 3b: Evaluate these BEST parameters on the HOLDOUT set for an unbiased performance estimate.
    _, analysis_df_holdout, _ = find_robust_hyperparameters(
        results=holdout_results_list,
        decision_metric=DECISION_METRIC
    )

    # Step 3c: Analyze the generalization gap.
    print(f"\nLooking up performance of TUNE-selected parameters on the HOLDOUT set for metric '{DECISION_METRIC}'...")
    holdout_performance_row = analysis_df_holdout[
        (analysis_df_holdout['conv_gamma'] == best_params_raw_tune['conv_gamma']) &
        (analysis_df_holdout['lin_gamma'] == best_params_raw_tune['lin_gamma']) &
        (analysis_df_holdout['proxy_temp'] == best_params_raw_tune['proxy_temp']) &
        (analysis_df_holdout['distance_metric'] == best_params_raw_tune['distance_metric']) &
        (analysis_df_holdout['metric_name'] == DECISION_METRIC) 
    ].iloc[0]

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
    print(f"Chosen Parameters: conv_gamma={best_params_raw_tune['conv_gamma']}, lin_gamma={best_params_raw_tune['lin_gamma']}, dist={best_params_raw_tune['distance_metric']}, proxy_temp={best_params_raw_tune['proxy_temp']}")
    print(f"Performance on TUNE set:    Mean Faithfulness = {tune_performance['mean_faithfulness']:.4f}")
    print(f"Performance on HOLDOUT set: Mean Faithfulness = {holdout_performance['mean_faithfulness']:.4f}")

    generalization_drop = (tune_performance['mean_faithfulness'] - holdout_performance['mean_faithfulness'])
    relative_drop_percent = (generalization_drop / tune_performance['mean_faithfulness']) * 100 if tune_performance['mean_faithfulness'] != 0 else 0

    print(f"\nGeneralization Drop: {relative_drop_percent:.2f}%")

    ACCEPTABLE_DROP_PERCENT = 30.0
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

        print("\nOfficial Approved LRP Hyperparameters:")
        print(best_params_raw_tune)
        print("\nOfficial Unbiased Performance Metrics (from Holdout set):")
        print(holdout_performance)


    log_nested_validation_to_wandb (
        cfg=cfg, 
        final_decision=FINAL_DECISION,
        approved_params=best_params_raw_tune,
        worst_params=worst_params_raw_tune,
        tune_performance=tune_performance,
        holdout_performance=holdout_performance,
        generalization_drop_percent=relative_drop_percent,
        analysis_df_tune=analysis_df_tune,
        analysis_df_holdout=analysis_df_holdout,
        tune_results_list=tune_results_list,
        holdout_results_list=holdout_results_list,
        tune_curves_list=tune_curves_list,
        holdout_curves_list=holdout_curves_list,
        run=run,
    )

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DINOv2 AttnLRP sweep.")
    parser.add_argument(
        "--config_name", 
        type=str, 
        required=True,
        help="The name of the experiment config (e.g., 'finetuned', 'non_finetuned')."
    )   
    
    args, unknown_args = parser.parse_known_args()

    cfg = load_config(args.config_name, unknown_args)

    main(cfg)