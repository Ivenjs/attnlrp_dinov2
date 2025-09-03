import os
import random
import subprocess
from omegaconf import OmegaConf
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
import argparse
from lxt.efficient import monkey_patch_zennit
import wandb

from utils import get_db_path, get_mask_transform, load_config, get_hpi_colors
from basemodel import get_model_wrapper
from dataset import GorillaReIDDataset, PerturbedGorillaReIDDataset, custom_collate_fn
from model_evaluation import evaluate_model
from lrp_helpers import get_relevances
from knn_helpers import get_knn_db


def main(cfg):
    """
    Runs a faithfulness evaluation by perturbing images based on LRP relevance maps
    and measuring the impact on k-NN Re-ID accuracy.
    """
    # --- 1. Setup ---
    monkey_patch_zennit(verbose=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODE = cfg["lrp"]["mode"]
    print(f"\n--- RUNNING FAITHFULNESS EVALUATION WITH LRP MODE: {MODE} ---")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    model_wrapper, image_transforms, _ = get_model_wrapper(device=DEVICE, cfg=cfg["model"])
    mask_transform = get_mask_transform(cfg["model"]["img_size"])

    # --- 2. WandB Initialization ---
    model_type_str = "finetuned" if cfg["model"]["finetuned"] else "base"
    run_name = f"faithfulness_eval_{model_type_str}_{MODE}"
    wandb.init(
        project="Thesis-Iven",
        entity="gorillawatch",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        job_type="analysis"
    )

    # --- 3. Prepare Datasets ---
    split_name = cfg["data"]["analysis_split"]
    val_dir = os.path.join(cfg["data"]["dataset_dir"], split_name)
    val_files = [f for f in os.listdir(val_dir) if f.lower().endswith((".jpg", ".png"))]
    
    val_dataset = GorillaReIDDataset(
        image_dir=val_dir,
        filenames=val_files,
        transform=image_transforms,
        k=cfg["knn"]["k"],
    )
    
    train_dir = os.path.join(cfg["data"]["dataset_dir"], "train")
    train_files = [f for f in os.listdir(train_dir) if f.lower().endswith((".jpg", ".png"))]
    train_dataset = GorillaReIDDataset(
        image_dir=train_dir, filenames=train_files, transform=image_transforms
    )

    # The full database (gallery) for KNN search includes both training and validation sets.
    # This database remains constant throughout the experiment.
    full_db_dataset = ConcatDataset([train_dataset, val_dataset])

    # --- 4. Create the KNN Search Database (Gallery) ---
    print("Preparing the main KNN database (gallery)...")
    db_path = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name="all_dataset",
        split_name=f"train+{split_name}",
        db_dir=cfg["knn"]["db_embeddings_dir"]
    )
    db_embeddings, db_labels, db_filenames, db_videos = get_knn_db(
        db_path=db_path,
        dataset=full_db_dataset,
        model_wrapper=model_wrapper,
        batch_size=cfg["data"]["batch_size"],
        device=DEVICE
    )

    # --- 5. Generate or Load Relevance Maps (for validation images only) ---
    print("Generating/Loading relevance maps for validation images...")
    # Note: For relevance generation, we need embeddings of the validation set *only*
    # to find nearest neighbors within that set if required by the LRP mode.
    val_db_path_knn = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name=val_dataset.dataset_name, split_name=split_name, db_dir=cfg["knn"]["db_embeddings_dir"]
    )
    val_embeddings, val_labels, val_filenames, val_video_ids = get_knn_db(
        db_path=val_db_path_knn, dataset=val_dataset, model_wrapper=model_wrapper,
        batch_size=cfg["data"]["batch_size"], device=DEVICE
    )

    val_dataloader = DataLoader(val_dataset, batch_size=cfg["data"]["batch_size"], num_workers=0, shuffle=False, collate_fn=custom_collate_fn)
    
    db_path_relevances = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name=val_dataset.dataset_name, split_name=split_name, db_dir=cfg["lrp"]["db_relevances_dir"],
        decision_metric=MODE
    )
    
    relevances_all = get_relevances(
        db_path=db_path_relevances, model_wrapper=model_wrapper, dataloader=val_dataloader,
        device=DEVICE, recompute=False, conv_gamma=cfg["lrp"]["conv_gamma"], lin_gamma=cfg["lrp"]["lin_gamma"],
        proxy_temp=cfg["knn"]["temp"], distance_metric=cfg["knn"]["distance_metric"], mode=cfg["lrp"]["mode"],
        topk_neg=cfg["knn"]["topk_neg"], db_embeddings=val_embeddings, db_filenames=val_filenames,
        db_labels=val_labels, db_video_ids=val_video_ids, cross_encounter=cfg["lrp"]["cross_encounter"]
    )
    relevance_dict = {item['filename']: item['relevance'] for item in relevances_all}

    # --- 6. Run Perturbation Experiments ---
    # First, establish the baseline accuracy on unperturbed data.
    print("\n--- Evaluating Baseline Accuracy (0% Perturbation) ---")
    base_accuracy = evaluate_model(
        model_wrapper=model_wrapper, dataset=val_dataset, cfg=cfg, device=DEVICE,
        db_embeddings=db_embeddings, db_labels=db_labels, db_videos=db_videos
    )
    print(f"Baseline Balanced Accuracy: {base_accuracy:.4f}")

    perturbation_fractions = [0.25, 0.5, 0.75, 0.99]
    perturbation_modes = ['lerf', 'random', 'morf']
    results = {mode: [base_accuracy] for mode in perturbation_modes}

    for mode in perturbation_modes:
        print(f"\n===== Evaluating Perturbation Mode: {mode.upper()} =====")
        for frac in perturbation_fractions:
            print(f"--- Perturbation Fraction: {frac*100:.1f}% ---")
            
            perturbed_dataset = PerturbedGorillaReIDDataset(
                base_dataset=val_dataset,
                relevance_maps=relevance_dict,
                perturbation_mode=mode,
                perturbation_fraction=frac,
                patch_size=cfg["model"]["patch_size"],
                seed=cfg["seed"],
                baseline_value=cfg["eval"]["baseline_value"],
            )

            # Evaluate using the perturbed images as queries against the original, full database
            accuracy = evaluate_model(
                model_wrapper=model_wrapper,
                dataset=val_dataset, # Base dataset info, not strictly used
                cfg=cfg,
                device=DEVICE,
                db_embeddings=db_embeddings, # The consistent, full database
                db_labels=db_labels,
                db_videos=db_videos,
                query_dataset=perturbed_dataset # This triggers on-the-fly embedding generation
            )
            print(f"Accuracy for {mode.upper()} at {frac*100:.1f}% perturbation: {accuracy:.4f}")
            results[mode].append(accuracy)

    # --- 7. Plot and Log Results ---
    print("\nFinal Results:", results)
    
    plot_fractions = [0.0] + perturbation_fractions
    hpi_colors = get_hpi_colors(cfg=cfg)
    colors = {'morf': hpi_colors["red"], 'lerf': hpi_colors["yellow"], 'random': hpi_colors["gray"]}

    plt.figure(figsize=(10, 6))
    for mode in perturbation_modes:
        plt.plot(plot_fractions, results[mode], marker='o', linestyle='-', label=mode.upper(), color=colors.get(mode, 'black'))

    plt.title(f'Impact of Patch Perturbation on Re-ID Accuracy ({model_type_str} model, LRP mode: {MODE})')
    plt.xlabel('Fraction of Patches Perturbed')
    plt.ylabel(f"Balanced Cross-Video k-NN@{cfg['knn']['k']} Accuracy")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(bottom=0)
    plt.xticks(plot_fractions, [f'{int(p*100)}%' for p in plot_fractions])

    save_path = f"knn_perturbation_impact_{model_type_str}_{MODE}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

    try:
        wandb.log({"perturbation_results": results})
        wandb.log({"perturbation_impact_plot": wandb.Image(save_path, caption=f"Perturbation analysis for {run_name}")})
        print("Successfully logged results and plot to WandB.")
    except Exception as e:
        print(f"Could not log to WandB. Error: {e}")
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run faithfulness evaluation for LRP.")
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="The name of the experiment config (e.g., 'finetuned', 'non_finetuned')."
    )
    args, unknown_args = parser.parse_known_args()
    cfg = load_config(args.config_name, unknown_args)

    # Ensure masks are generated before running the main script
    print("Running mask generation script first...")
    command = [
        "python",
        "/workspaces/bachelor_thesis_code/src/bachelor_thesis/generate_masks.py",
        "--config_name",
        args.config_name
    ] + unknown_args
    subprocess.run(command, check=True)

    main(cfg)