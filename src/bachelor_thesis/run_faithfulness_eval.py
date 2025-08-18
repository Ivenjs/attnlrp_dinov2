import os
import random
import subprocess
from omegaconf import OmegaConf
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils import get_db_path, get_mask_transform, load_config
from basemodel import get_model_wrapper
from dataset import GorillaReIDDataset, custom_collate_fn
from model_evaluation import evaluate_model
from dataset import PerturbedGorillaReIDDataset
from lrp_helpers import get_relevances
from knn_helpers import get_knn_db 
import argparse 
from lxt.efficient import monkey_patch_zennit

import wandb
from utils import get_hpi_colors



def main(cfg):
    # --- 1. Standard Setup ---
    monkey_patch_zennit(verbose=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODE = cfg["lrp"]["mode"]
    print(f"\n--- RUNNING WITH MODE: {MODE} ---")

    DECISION_METRIC = MODE
    model_wrapper, image_transforms, _ = get_model_wrapper(device=DEVICE, cfg=cfg["model"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    split_name = cfg["data"]["analysis_split"]
    split_dir = os.path.join(cfg["data"]["dataset_dir"], split_name)
    split_files = [f for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".png"))]
    model_type_str = "finetuned" if cfg["model"]["finetuned"] else "base"
    mask_transform = get_mask_transform(cfg["model"]["img_size"])

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project="Thesis-Iven", 
        entity="gorillawatch", # Or your entity
        name=f"attnlrp_faithfulness_analysis_{model_type_str}_{DECISION_METRIC}",
        config=cfg_dict,
        job_type="analysis"
    )

    # Create the base validation dataset
    base_dataset = GorillaReIDDataset(
        image_dir=split_dir,
        filenames=split_files,
        transform=image_transforms,
        base_mask_dir=cfg["data"]["base_mask_dir"],
        mask_transform=mask_transform,
        k=cfg["knn"]["k"], 
    )

    # --- 2. Generate or Load Relevance Maps (Expensive step, do it once!) ---
    db_path_knn = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset=base_dataset,
        split_name=split_name,
        db_dir=cfg["knn"]["db_embeddings_dir"]
    )
    db_embeddings, db_labels, db_filenames, db_video_ids = get_knn_db(
        db_path=db_path_knn,
        dataset=base_dataset,
        model_wrapper=model_wrapper,
        batch_size=cfg["data"]["batch_size"],
        device=DEVICE
    )
    
    dataloader = DataLoader(base_dataset, batch_size=cfg["data"]["batch_size"], num_workers=0,shuffle=False, collate_fn=custom_collate_fn)


    db_path_relevances = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset=base_dataset,
        split_name=split_name,
        db_dir=cfg["lrp"]["db_relevances_dir"],
        decision_metric=DECISION_METRIC
    )

    relevances_all = get_relevances(
        db_path=db_path_relevances,
        model_wrapper=model_wrapper,
        dataloader=dataloader,
        device=DEVICE,
        recompute=False,
        # All of these will be caught by **kwargs and passed to generate_relevances
        conv_gamma=cfg["lrp"]["conv_gamma"],           # Pass as single value (will be converted to list)
        lin_gamma=cfg["lrp"]["lin_gamma"],             # Pass as single value
        proxy_temp=cfg["knn"]["temp"],          # Pass as single value 
        distance_metric=cfg["knn"]["distance_metric"], #pass as single value
        mode=cfg["lrp"]["mode"],
        topk_neg=cfg["knn"]["topk_neg"],
        db_embeddings=db_embeddings,
        db_filenames=db_filenames,
        db_labels=db_labels,
        db_video_ids=db_video_ids
    )

    relevance_dict = {
            item['filename']: item['relevance'] for item in relevances_all
        }


    # --- 3. Run Perturbation Experiments ---
    hpi_colors = get_hpi_colors(cfg=cfg)
    colors = {
        'morf': hpi_colors["red"],
        'lerf': hpi_colors["yellow"],
        'random': hpi_colors["gray"],
    }

    # establish baseline
    base_accuracy = evaluate_model(
        model_wrapper=model_wrapper,
        val_dataset=base_dataset,
        cfg=cfg,
        device=DEVICE,
        db_embeddings=db_embeddings,
        db_labels=db_labels,
        db_videos=db_video_ids
    )
    print(f"Base accuracy: {base_accuracy:.4f}")

    perturbation_fractions = [0.25, 0.5, 0.75, 0.99]
    modes = ['morf', 'lerf', 'random']
    results = {mode: [base_accuracy] for mode in modes}
    

    patch_size = cfg["model"]["patch_size"] 

    for mode in modes:
        print(f"\n===== Evaluating Mode: {mode.upper()} =====")
        for frac in perturbation_fractions:
            print(f"--- Perturbation Fraction: {frac*100:.1f}% ---")
            
            # Create the perturbed dataset for this specific configuration
            perturbed_dataset = PerturbedGorillaReIDDataset(
                base_dataset=base_dataset,
                relevance_maps=relevance_dict,
                perturbation_mode=mode,
                perturbation_fraction=frac,
                patch_size=patch_size,
                baseline_value=cfg["eval"]["baseline_value"],
            )

            # Run the full, original evaluation pipeline on the perturbed data
            accuracy = evaluate_model(
                model_wrapper=model_wrapper,
                val_dataset=base_dataset,
                cfg=cfg,
                device=DEVICE,
                db_embeddings=db_embeddings,
                db_labels=db_labels,
                db_videos=db_video_ids,
                query_dataset=perturbed_dataset
            )
            print(f"Accuracy for {mode.upper()} at {frac*100:.1f}% perturbation: {accuracy:.4f}")
            results[mode].append(accuracy)

    # --- 4. Plot Results ---
    plot_fractions = [0.0] + perturbation_fractions

    plt.figure(figsize=(10, 6))
    for mode in modes:
        plt.plot(plot_fractions, 
                 results[mode], 
                 marker='o', 
                 linestyle='-', 
                 label=mode.upper(), 
                 color=colors.get(mode, 'black')
                )
    
    plt.title(f'Impact of Patch Perturbation on k-NN Re-ID Accuracy for model {cfg["model"]["finetuned"]}')
    plt.xlabel('Fraction of Patches Perturbed')
    plt.ylabel(f"Cross-Video k-NN@{cfg['knn']['k']} Accuracy")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(bottom=0)
    plt.xticks(plot_fractions, [f'{int(p*100)}%' for p in plot_fractions])

    model_name = "finetuned" if cfg["model"]["finetuned"] else "non_finetuned"

    save_path = f"knn_perturbation_impact_{model_name}_{DECISION_METRIC}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

    print("\nFinal Results:")
    print(results)

    try:
        print("\nLogging results to Weights & Biases...")
        # Log the final results dictionary. WandB will render this nicely.
        wandb.log({"perturbation_results": results})

        # Log the locally saved plot as a wandb.Image object
        wandb.log({
            "perturbation_impact_plot": wandb.Image(save_path, caption=f"Perturbation analysis for {cfg['model']['finetuned']}")
        })
        print("Successfully logged to WandB.")
    except Exception as e:
        print(f"Could not log to WandB. Please ensure wandb.init() was called. Error: {e}")
    wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run faithfulness eval.")
    parser.add_argument(
        "--config_name", 
        type=str, 
        required=True,
        help="The name of the experiment config (e.g., 'finetuned', 'non_finetuned')."
    )   
    
    args, unknown_args = parser.parse_known_args()

    cfg = load_config(args.config_name, unknown_args)

    # Also has to be run because of the caching of the relevance and mask values
    command = [
        "python", 
        "/workspaces/bachelor_thesis_code/src/bachelor_thesis/generate_masks.py", 
        "--config_name", 
        args.config_name
    ] + unknown_args

    result = subprocess.run(
        command,
        check=True
    )

    main(cfg)
