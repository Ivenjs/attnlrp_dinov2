from omegaconf import OmegaConf
import torch
from lxt.efficient import monkey_patch_zennit
import os
import argparse
import random
from collections import defaultdict
from torch.utils.data import DataLoader, ConcatDataset
import wandb

from basemodel import get_model_wrapper
from sweep_helpers import (
    evaluate_gamma_sweep_proxy_score, 
    evaluate_gamma_sweep_acc,
    print_robustness_summary, 
    find_robust_hyperparameters,
    log_sweep
    )
from knn_helpers import get_knn_db
from dataset import GorillaReIDDataset, custom_collate_fn
from utils import get_db_path, load_config
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

    val_dir = os.path.join(root_dir, "validation")
    val_files = [f for f in os.listdir(val_dir) if f.lower().endswith((".jpg", ".png"))]


    train_db_dataset = GorillaReIDDataset(
        image_dir=train_dir, 
        filenames=train_files, 
        transform=image_transforms, 
        k=cfg["knn"]["k"]
    )

    val_db_dataset = GorillaReIDDataset(
        image_dir=val_dir, 
        filenames=val_files,
        transform=image_transforms, 
        k=cfg["knn"]["k"]
    )


    train_query_indices = []
    val_query_indices = []

    print("Filtering eligible queries into train/val sets...")
    queries_per_class_counter = defaultdict(int)

    for idx in train_db_dataset.images_for_ce_knn:
        label = train_db_dataset.labels[idx]
        if queries_per_class_counter[label] < cfg["sweep"]["queries_per_class"]:
            train_query_indices.append(idx)
            queries_per_class_counter[label] += 1
    
    for idx in val_db_dataset.images_for_ce_knn:
        label = val_db_dataset.labels[idx]
        if queries_per_class_counter[label] < cfg["sweep"]["queries_per_class"]:
            val_query_indices.append(idx)
            queries_per_class_counter[label] += 1


    print(f"Selected {len(train_query_indices)} train queries and {len(val_query_indices)} val queries.")

    train_query_subset = torch.utils.data.Subset(train_db_dataset, train_query_indices)
    val_query_subset = torch.utils.data.Subset(val_db_dataset, val_query_indices)

    global_train_query_indices = train_query_indices

    datasets = [train_db_dataset, val_db_dataset]
    full_db_dataset = ConcatDataset(datasets)
    full_dataset_splits = "+".join([os.path.basename(d.image_dir) for d in datasets])

    full_db_path = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name=train_db_dataset.dataset_name,
        split_name=full_dataset_splits,
        bp_transforms=cfg["model"]["bp_transforms"],
        db_dir=cfg["knn"]["db_embeddings_dir"]
    )

    full_db_embeddings, full_db_labels, full_db_filenames, full_db_videos = get_knn_db(
        db_path=full_db_path,
        dataset=full_db_dataset,
        model_wrapper=model_wrapper,
        batch_size=cfg["data"]["batch_size"],
        device=DEVICE
    )

    query_dataset_offset = 0
    found = False
    for d in datasets:
        if d is val_db_dataset:
            found = True
            break
        query_dataset_offset += len(d)


    if not found:
        raise RuntimeError("Query dataset (split_dataset) not found in db_constituents.")

    global_val_query_indices = [idx + query_dataset_offset for idx in val_query_indices]

    print("\n--- RUNNING FULL SWEEP ON TRAIN SET ---")

    train_db_path = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name=train_db_dataset.dataset_name,
        split_name="train",
        bp_transforms=cfg["model"]["bp_transforms"],
        db_dir=cfg["knn"]["db_embeddings_dir"]
    )

    train_db_embeddings, train_db_labels, train_db_filenames, train_db_videos = get_knn_db(
        db_path=train_db_path,
        dataset=train_db_dataset,
        model_wrapper=model_wrapper,
        batch_size=cfg["data"]["batch_size"],
        device=DEVICE
    )


    train_dataloader = DataLoader(train_query_subset, batch_size=cfg["data"]["batch_size"], num_workers=4, collate_fn=custom_collate_fn,shuffle=False)

    print(f"sweep parameters:")
    for key, value in cfg["sweep"].items():
        print(f"  {key}: {value}")

    train_relevances_all = generate_relevances(
        model_wrapper=model_wrapper,
        dataloader=train_dataloader,
        device=DEVICE,
        conv_gamma_values=cfg["sweep"]["conv_gammas"],
        lin_gamma_values=cfg["sweep"]["lin_gammas"],
        mode=MODE,
        distance_metrics=cfg["sweep"]["distance_metrics"],
        proxy_temp_values=cfg["sweep"]["temp"],
        topk_values=cfg["sweep"]["topk"],
        db_embeddings=train_db_embeddings,
        db_filenames=train_db_filenames,
        db_labels=train_db_labels,
        db_video_ids=train_db_videos,
        cross_encounter=cfg["lrp"]["cross_encounter"]
    )

    if cfg["sweep"]["sweep_evaluation"] == "proxy_score":
        train_eval_dataloader = DataLoader(train_query_subset, batch_size=1, num_workers=4, collate_fn=custom_collate_fn)
        train_results_list, train_curves_list = evaluate_gamma_sweep_proxy_score(
            train_relevances_all, train_eval_dataloader, model_wrapper,
            train_db_embeddings, train_db_labels, train_db_filenames, train_db_videos, cfg["model"]["patch_size"], DEVICE,
            cfg["eval"]["patches_per_step"], cfg["eval"]["baseline_value"], cfg["lrp"]["cross_encounter"],False, cfg["seed"]
        )
    elif cfg["sweep"]["sweep_evaluation"] == "accuracy":
        train_results_list, train_curves_list = evaluate_gamma_sweep_acc(
            all_relevance_results=train_relevances_all,
            query_dataset=train_query_subset,
            global_query_indices=global_train_query_indices,
            model=model_wrapper,
            db_embeddings=train_db_embeddings,
            db_labels=train_db_labels,
            db_videos=train_db_videos,
            cfg=cfg,
            patch_size=cfg["model"]["patch_size"],
            patches_per_step=cfg["eval"]["patches_per_step"],
            baseline_value=cfg["eval"]["baseline_value"]
        )


    print("\n--- PHASE 2: FINDING TOP-K CANDIDATES FROM TRAIN SET ---")
    _, analysis_df_train, _ = find_robust_hyperparameters(
        results=train_results_list,
        decision_metric=DECISION_METRIC,
        sweep_evaluation=cfg["sweep"]["sweep_evaluation"]
    )

    analysis_df_train = analysis_df_train.sort_values(by='raw_mean', ascending=False).reset_index(drop=True)

    top_k = cfg["sweep"].get("top_k_for_val", 10) 
    top_k_df = analysis_df_train.head(top_k)
    
    param_cols = ['conv_gamma', 'lin_gamma', 'distance_metric', 'proxy_temp', 'topk']
    valid_param_cols = [col for col in param_cols if col in top_k_df.columns]
    
    top_k_param_combos = top_k_df[valid_param_cols].to_dict('records')

    print(f"\nIdentified Top {len(top_k_param_combos)} parameter combinations to test on the validation set:")
    for i, params in enumerate(top_k_param_combos):
        print(f"  {i+1}. {params}")

    print("\n--- PHASE 3: TARGETED SWEEP ON VAL SET ---")

    val_dataloader = DataLoader(val_query_subset, batch_size=cfg["data"]["batch_size"], num_workers=4, collate_fn=custom_collate_fn, shuffle=False)
    
    val_relevances_all = generate_relevances(
        model_wrapper=model_wrapper,
        dataloader=val_dataloader,
        device=DEVICE,
        param_combinations_list=top_k_param_combos,
        mode=MODE,
        db_embeddings=full_db_embeddings,
        db_filenames=full_db_filenames,
        db_labels=full_db_labels,
        db_video_ids=full_db_videos,
        cross_encounter=cfg["lrp"]["cross_encounter"],
        # won't be used
        conv_gamma_values=[],
        lin_gamma_values=[],
        distance_metrics=[],
        proxy_temp_values=[],
        topk_values=[]
    )

    if cfg["sweep"]["sweep_evaluation"] == "proxy_score":
        val_eval_dataloader = DataLoader(val_query_subset, batch_size=1, num_workers=4, collate_fn=custom_collate_fn)
        val_results_list, val_curves_list = evaluate_gamma_sweep_proxy_score(
            val_relevances_all, val_eval_dataloader, model_wrapper,
            full_db_embeddings, full_db_labels, full_db_filenames, full_db_videos, cfg["model"]["patch_size"], DEVICE,
            cfg["eval"]["patches_per_step"], cfg["eval"]["baseline_value"], cfg["lrp"]["cross_encounter"], False, cfg["seed"]
        )
    elif cfg["sweep"]["sweep_evaluation"] == "accuracy":
        val_results_list, val_curves_list = evaluate_gamma_sweep_acc(
            all_relevance_results=val_relevances_all,
            query_dataset=val_query_subset,
            global_query_indices=global_val_query_indices,
            model=model_wrapper,
            db_embeddings=full_db_embeddings,
            db_labels=full_db_labels,
            db_videos=full_db_videos,
            cfg=cfg,
            patch_size=cfg["model"]["patch_size"],
            patches_per_step=cfg["eval"]["patches_per_step"],
            baseline_value=cfg["eval"]["baseline_value"]
        )

    print("\n--- PHASE 4: FINAL SELECTION ON VAL SET ---")
    
    best_params_val, analysis_df_val, worst_params_val = find_robust_hyperparameters(
        results=val_results_list,
        decision_metric=DECISION_METRIC,
        sweep_evaluation=cfg["sweep"]["sweep_evaluation"]
    )

    print_robustness_summary(best_params_val, analysis_df_val, DECISION_METRIC)

    train_performance_row = analysis_df_train[
        (analysis_df_train['conv_gamma'] == best_params_val['conv_gamma']) &
        (analysis_df_train['lin_gamma'] == best_params_val['lin_gamma']) &
        (analysis_df_train['proxy_temp'] == best_params_val['proxy_temp']) &
        (analysis_df_train['distance_metric'] == best_params_val['distance_metric']) &
        (analysis_df_train['topk'] == best_params_val['topk']) &
        (analysis_df_train['metric_name'] == DECISION_METRIC)
    ].iloc[0]

    train_performance = {
        'mean_faithfulness': train_performance_row['raw_mean'],
        'min_faithfulness': train_performance_row['raw_min'],
        'std_faithfulness': train_performance_row['raw_std'],
    }
    val_performance = {
        'mean_faithfulness': best_params_val['mean_score'],
        'min_faithfulness': best_params_val['min_score'],
        'std_faithfulness': best_params_val['std_score'],
    }

    print("\n" + "="*80)
    print("--- FINAL VALIDATION & DECISION ---")
    print("="*80)
    print(f"Chosen Parameters: conv_gamma={best_params_val['conv_gamma']}, lin_gamma={best_params_val['lin_gamma']}, dist={best_params_val['distance_metric']}, proxy_temp={best_params_val['proxy_temp']}, topk={best_params_val['topk']}")
    print(f"Performance on TRAIN set: Mean Faithfulness = {train_performance['mean_faithfulness']:.4f}")
    print(f"Performance on VAL set:   Mean Faithfulness = {val_performance['mean_faithfulness']:.4f}")

    generalization_drop = (train_performance['mean_faithfulness'] - val_performance['mean_faithfulness'])
    relative_drop_percent = (generalization_drop / train_performance['mean_faithfulness']) * 100 if train_performance['mean_faithfulness'] != 0 else 0

    print(f"\nGeneralization Drop: {relative_drop_percent:.2f}%")

    if relative_drop_percent > cfg["sweep"]["acceptable_drop_percent"]:
        print(f"DECISION: REJECT. High generalization drop.")
        FINAL_DECISION = "REJECTED"
    else:
        print(f"DECISION: APPROVE. Generalization is within acceptable limits.")
        FINAL_DECISION = "APPROVED"
        
        print("\n" + "="*80)
        print("--- PHASE 4: FINAL REPORTING ---")
        print("="*80)

        print("\nOfficial Approved LRP Hyperparameters:")
        print(best_params_val)


    log_sweep (
        cfg=cfg, 
        final_decision=FINAL_DECISION,
        approved_params=best_params_val,
        worst_params=worst_params_val,
        tune_performance=train_performance,
        holdout_performance=val_performance,
        generalization_drop_percent=relative_drop_percent,
        analysis_df_tune=analysis_df_train,
        analysis_df_holdout=analysis_df_val,
        tune_results_list=train_results_list,
        holdout_results_list=val_results_list,
        tune_curves_list=train_curves_list,
        holdout_curves_list=val_curves_list,
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
