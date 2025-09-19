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

from eval_helpers import faithfulness_eval_acc
from utils import get_db_path, get_mask_transform, load_config, get_hpi_colors, parse_encounter_id
from basemodel import get_model_wrapper
from dataset import GorillaReIDDataset, custom_collate_fn
from model_evaluation import evaluate_model
from lrp_helpers import get_relevances
from knn_helpers import get_knn_db


def main(cfg):
    """
    Runs a faithfulness evaluation by perturbing images based on LRP relevance maps
    and measuring the impact on k-NN Re-ID accuracy.
    """
    monkey_patch_zennit(verbose=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODE = cfg["lrp"]["mode"]
    print(f"\n--- RUNNING FAITHFULNESS EVALUATION WITH LRP MODE: {MODE} ---")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    model_wrapper, image_transforms, _ = get_model_wrapper(device=DEVICE, cfg=cfg["model"])
    mask_transform = get_mask_transform(cfg["model"]["img_size"])

    print(f"the dtype of the model is: {next(model_wrapper.model.parameters()).dtype}")

    model_type_str = "finetuned" if cfg["model"]["finetuned"] else "base"
    run_name = f"faithfulness_eval_{model_type_str}_{MODE}"
    wandb.init(
        project="Thesis-Iven",
        entity="gorillawatch",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        job_type="analysis"
    )

    # --- Prepare Datasets ---
    dataset_dir = cfg["data"]["dataset_dir"]
    if not "zoo" in dataset_dir:
        split_name = cfg["data"]["analysis_split"]
        split_dir = os.path.join(dataset_dir, split_name)
        split_files = [f for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".png"))]
        
        split_dataset = GorillaReIDDataset(
            image_dir=split_dir,
            filenames=split_files,
            transform=image_transforms,
            base_mask_dir=cfg["data"]["base_mask_dir"],
            mask_transform=mask_transform,
            k=cfg["knn"]["k"],
        )
        
        train_dir = os.path.join(cfg["data"]["dataset_dir"], "train")
        train_files = [f for f in os.listdir(train_dir) if f.lower().endswith((".jpg", ".png"))]
        train_dataset = GorillaReIDDataset(
            image_dir=train_dir, filenames=train_files, transform=image_transforms
        )
        

        datasets = [split_dataset, train_dataset]
        full_db_dataset = ConcatDataset(datasets)
        full_dataset_splits = "+".join([os.path.basename(d.image_dir) for d in datasets])

        print(f"full dataset contains {len(full_db_dataset)} images from splits: {full_dataset_splits}")


        query_dataset_offset = 0
        found = False
        for d in datasets:
            if d is split_dataset:
                found = True
                break
            query_dataset_offset += len(d)

        print("Query dataset offset in DB:", query_dataset_offset)

        if not found:
            raise RuntimeError("Query dataset (split_dataset) not found in db_constituents.")
        
    else:
        print("Using Zoo dataset for evaluation.")
        split_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith((".jpg", ".png"))]
        split_dataset = GorillaReIDDataset(
            image_dir=dataset_dir,
            filenames=split_files,
            transform=image_transforms,
            base_mask_dir=cfg["data"]["base_mask_dir"],
            mask_transform=mask_transform,
            k=cfg["knn"]["k"],
        )

        query_dataset_offset = 0
        full_db_dataset = split_dataset
        full_dataset_splits = os.path.basename(dataset_dir)
        split_name = full_dataset_splits



    local_query_indices = split_dataset.images_for_ce_knn

    global_query_indices = [idx + query_dataset_offset for idx in local_query_indices]

    # --- Create the KNN Search Database ---
    print("Preparing the main KNN database (gallery)...")
    db_path = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name=split_dataset.dataset_name,
        split_name=full_dataset_splits,
        bp_transforms=cfg["model"]["bp_transforms"],
        db_dir=cfg["knn"]["db_embeddings_dir"]
    )
    all_db_embeddings, all_db_labels, all_db_filenames, all_db_videos = get_knn_db(
        db_path=db_path,
        dataset=full_db_dataset,
        model_wrapper=model_wrapper,
        batch_size=cfg["data"]["batch_size"],
        device=DEVICE
    )


    # --- Generate or Load Relevance Maps (for test images only) ---
    split_db_path_knn = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name=split_dataset.dataset_name, split_name=split_name, bp_transforms=cfg["model"]["bp_transforms"], db_dir=cfg["knn"]["db_embeddings_dir"]
    )
    split_embeddings, split_labels, split_filenames, split_video_ids = get_knn_db(
        db_path=split_db_path_knn, dataset=split_dataset, model_wrapper=model_wrapper,
        batch_size=cfg["data"]["batch_size"], device=DEVICE
    )

    # this loader only contains the subset of images that are used as queries for cross-encounter knn
    split_query_subset = torch.utils.data.Subset(split_dataset, local_query_indices)

    split_dataloader = DataLoader(split_query_subset, batch_size=cfg["data"]["batch_size"], num_workers=0, shuffle=False, collate_fn=custom_collate_fn)


    if cfg["lrp"]["eval_db"] == "test":
        relevance_split_name = split_name
        relevance_db_embeddings = split_embeddings
        relevance_db_labels = split_labels
        relevance_db_filenames = split_filenames
        relevance_db_videos = split_video_ids
    elif cfg["lrp"]["eval_db"] == "all":
        relevance_split_name = full_dataset_splits
        relevance_db_embeddings = all_db_embeddings
        relevance_db_labels = all_db_labels
        relevance_db_filenames = all_db_filenames
        relevance_db_videos = all_db_videos

    db_path_relevances = get_db_path(
        model_checkpoint_path=cfg["model"]["checkpoint_path"],
        dataset_name=split_dataset.dataset_name,
        split_name=relevance_split_name,
        bp_transforms=cfg["model"]["bp_transforms"], 
        db_dir=cfg["lrp"]["db_relevances_dir"],
        decision_metric=MODE,
        lrp_params={
            "conv_gamma": cfg["lrp"]["conv_gamma"],
            "lin_gamma": cfg["lrp"]["lin_gamma"],
            "proxy_temp": cfg["lrp"]["temp"],
            "topk": cfg["lrp"]["topk"],
        }
    )
    
    
    relevances_all = get_relevances(
        db_path=db_path_relevances, 
        model_wrapper=model_wrapper, 
        dataloader=split_dataloader,
        device=DEVICE, 
        recompute=False, 
        conv_gamma=cfg["lrp"]["conv_gamma"], 
        lin_gamma=cfg["lrp"]["lin_gamma"],
        proxy_temp=cfg["lrp"]["temp"], 
        distance_metric=cfg["lrp"]["distance_metric"], 
        mode=cfg["lrp"]["mode"],
        topk=cfg["lrp"]["topk"], 
        db_embeddings=relevance_db_embeddings, 
        db_filenames=relevance_db_filenames,
        db_labels=relevance_db_labels, 
        db_video_ids=relevance_db_videos, 
        cross_encounter=cfg["lrp"]["cross_encounter"]
    )
    relevance_dict = {item['filename']: item['relevance'] for item in relevances_all}

    # --- Perturbation Experiments ---
    # Cross checking unperturbed accuracy
    print("\n--- Evaluating Baseline Accuracy (0% Perturbation) ---")
    base_accuracy = evaluate_model(
        model_wrapper=model_wrapper, query_indices_in_db=global_query_indices, cfg=cfg, device=DEVICE,
        db_embeddings=all_db_embeddings, db_labels=all_db_labels, db_videos=all_db_videos
    )
    print(f"Baseline Balanced Accuracy: {base_accuracy:.4f}")

    unique_labels = sorted(list(set(all_db_labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    all_db_labels_int = torch.tensor([label_to_id[s] for s in all_db_labels], dtype=torch.long, device=DEVICE)

    db_encounters = [parse_encounter_id(v) for v in all_db_videos]
    unique_encounters = sorted(list(set(db_encounters)))
    encounter_to_id = {enc: i for i, enc in enumerate(unique_encounters)}
    all_db_encounter_ids_int = torch.tensor([encounter_to_id[s] for s in db_encounters], dtype=torch.long, device=DEVICE)

    perturbation_fractions = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1] # 0 will be included by default 


    relevances_name=os.path.basename(db_path_relevances)
    eval_results = faithfulness_eval_acc(
        relevance_maps_dict=relevance_dict,
        query_dataset=split_query_subset, 
        global_query_indices=global_query_indices,
        model=model_wrapper, 
        db_embeddings=all_db_embeddings,
        db_labels_int=all_db_labels_int,
        db_encounter_ids_int=all_db_encounter_ids_int,
        label_to_id=label_to_id,
        encounter_to_id=encounter_to_id,
        cfg=cfg,
        patch_size=cfg["model"]["patch_size"],
        patches_per_step=cfg["eval"]["patches_per_step"],
        baseline_value=cfg["eval"]["baseline_value"],
        seed=cfg["seed"],
        fractions_to_record=perturbation_fractions, # this overrides the patches_per_step, but is optional
        relevances_name=relevances_name
    )

    curves = {
        'lerf': eval_results["fraction_accuracies_lerf"],
        'morf': eval_results["fraction_accuracies_morf"],
        'random': eval_results["fraction_accuracies_random"]
    }
    print("\nEvaluation Results:")
    print(curves)

    hpi_colors = get_hpi_colors(cfg=cfg)
    colors = {'morf': hpi_colors["red"], 'lerf': hpi_colors["yellow"], 'random': hpi_colors["gray"]}


    plt.figure(figsize=(10, 6))
    for curve_name, frac_acc_dict in curves.items():
        fractions, accuracies = zip(*sorted(frac_acc_dict.items()))
        
        plt.plot(
            fractions, accuracies,
            marker='o', linestyle='-',
            label=curve_name.upper(),
            color=colors.get(curve_name, 'black')
        )

    plt.title(f'Impact of Patch Perturbation on Re-ID Accuracy ({model_type_str} model, LRP mode: {MODE})')
    plt.xlabel('Fraction of Patches Perturbed')
    plt.ylabel(f"Balanced Cross-Encounter k-NN@{cfg['knn']['k']} Accuracy")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(bottom=0)

    plt.xticks(fractions, [f'{int(f*100)}%' for f in fractions])

    save_path = f"knn_perturbation_impact_{model_type_str}_{MODE}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

    try:
        wandb.log({"perturbation_results": eval_results})
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

    command = [
        "python",
        "/workspaces/bachelor_thesis_code/src/bachelor_thesis/generate_masks.py",
        "--config_name",
        args.config_name
    ] + unknown_args
    #subprocess.run(command, check=True)

    main(cfg)