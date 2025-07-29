from mask_generator import MaskGenerator
from utils import load_all_configs
from dataset import GorillaReIDDataset, custom_collate_fn
from lxt.efficient import monkey_patch_zennit
from torchvision import transforms
from basemodel import get_model_wrapper
from knn_helpers import get_knn_db
from eval_helpers import attention_inside_mask
from tqdm import tqdm
from typing import Dict, Tuple, Any, List
from collections import defaultdict
from lrp_helpers import compute_simple_attnlrp_pass, compute_knn_attnlrp_pass, LRPConservationChecker
from dino_patcher import DINOPatcher
from basemodel import TimmWrapper
from torch.utils.data import DataLoader
from dinov2_attnlrp_sweep import run_gamma_sweep
import gc
import numpy as np
import subprocess
import os
import random
import torch
# 1) run sam to get segmentation masks of data split, if not already saved
# 2) create dataset with masks
# 3) run lrp with swept parameters on images in dataset and compute the mask score
# 3a) compare basemodel vs finetuned model on val
# 3b) compare finetuned model on train vs val (overfitting?)
# 4) save worst performing images and mask their background. How does the knn score change? can I also recompute accuracy with only these few images?


def compute_relevances(
    model_wrapper: TimmWrapper, 
    conv_gamma: float, 
    lin_gamma: float, 
    dataloader: DataLoader, 
    device: torch.device,
    db_embeddings: torch.Tensor = None, 
    db_labels: List[str] = None, 
    db_filenames: List[str] = None, 
    k_neighbors: int = 5, 
    mode: str = "knn", 
    verbose: bool = False, 
    distance_metric: str = "cosine"
) -> Tuple[Dict[float, Tuple[torch.Tensor, np.ndarray]], Dict[int, Dict[Tuple[float, float], Any]]]:
    relevance_mask_dict = defaultdict(dict)
    violations = defaultdict(dict)
    with DINOPatcher(model_wrapper), LRPConservationChecker(model_wrapper) as checker:
        for batch in tqdm(dataloader, desc="Processing batches"):
            input_batch = batch["image"].to(device)
            labels_batch = batch["label"]
            filenames_batch = batch["filename"]
            mask_batch = batch["mask"]
            for j, filename in enumerate(filenames_batch):
                # Slice the data for the j-th sample
                input_tensor_single = input_batch[j].unsqueeze(0) 
                label_single = labels_batch[j]

                if mode == "simple":
                    # Call the non-batched LRP function directly
                    relevance_single, violations_single = compute_simple_attnlrp_pass(
                        conv_gamma=conv_gamma,
                        lin_gamma=lin_gamma,
                        model_wrapper=model_wrapper,
                        input_tensor=input_tensor_single,  
                        checker=checker,
                        verbose=verbose  
                    )

                elif mode == "knn":
                    assert db_embeddings is not None, "db_embeddings must be provided for 'knn' mode."
                    assert db_filenames is not None, "db_filenames must be provided for 'knn' mode."
                    # Call the non-batched LRP function directly
                    relevance_single, violations_single = compute_knn_attnlrp_pass(
                        conv_gamma=conv_gamma,
                        lin_gamma=lin_gamma,
                        model_wrapper=model_wrapper,
                        input_tensor=input_tensor_single,
                        checker=checker,
                        query_label=label_single,
                        query_filename=filename,
                        db_embeddings=db_embeddings,
                        db_labels=db_labels,
                        db_filenames=db_filenames,
                        distance_metric=distance_metric,
                        k_neighbors=k_neighbors,
                        verbose=verbose, 
                    )

                mask_tensor_single = mask_batch[j]
            
                # Create a memory-safe NumPy copy of the mask
                mask_np_copy = None
                if mask_tensor_single is not None:
                    # 1. Move tensor to CPU
                    # 2. Convert to NumPy array
                    # 3. This implicitly copies the data, breaking the memory link.
                    mask_np_copy = mask_tensor_single.cpu().numpy()

                # Store the final, safe data
                # relevance_single is detached from graph and moved to CPU
                # mask_np_copy is a plain NumPy array with no ties to the DataLoader
                relevance_mask_dict[filename] = (relevance_single.detach().cpu(), mask_np_copy)
                violations[filename] = violations_single
    return relevance_mask_dict, violations


if __name__ == "__main__":

    result = subprocess.run(
        ["python", "/workspaces/bachelor_thesis_code/src/bachelor_thesis/generate_masks.py"],
        check=True  
    )

    monkey_patch_zennit(verbose=True)  

    LOG_TO_WANDB = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODE = "knn"  # "simple" or "knn"
    VERBOSE = False  
    random.seed(27)  
    torch.manual_seed(27)  

    model_wrapper_finetuned, image_transforms = get_model_wrapper(device=DEVICE, finetuned=True)

    config_dir = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/configs"
    cfg = load_all_configs(config_dir)

    root_dir = cfg["data"]["dataset_dir"]
    val_dir = os.path.join(root_dir, "val")

    val_files = [f for f in os.listdir(val_dir) if f.lower().endswith((".jpg", ".png"))]


    mask_transform = transforms.Compose([
        transforms.Resize(
            size=cfg["model"]["img_size"], # e.g., (518, 518)
            interpolation=transforms.InterpolationMode.NEAREST # Use NEAREST for masks!
        ),
        transforms.ToTensor(), # Converts mask to a [1, H, W] tensor of floats (0.0 or 1.0)
    ])
    # IMPORTANT: The interpolation mode for the mask must be NEAREST.
    # Using BILINEAR or BICUBIC would create intermediate values (like 0.5)
    # along the edges, blurring the mask.

    val_dataset = GorillaReIDDataset(
        image_dir=val_dir,
        filenames=val_files,
        transform=image_transforms,      # The full transform for the image
        base_mask_dir=cfg["data"]["base_mask_dir"],
        mask_transform=mask_transform  # The spatial-only transform for the mask
    )

    val_db_embeddings_finetuned, val_db_labels_finetuned, val_db_filenames_finetuned = get_knn_db(
        db_dir=cfg["knn"]["db_embeddings_dir"], split_name="val", dataset=val_dataset,
        model_wrapper=model_wrapper_finetuned, model_checkpoint_path=cfg["model"]["checkpoint_path"], batch_size=cfg["data"]["batch_size"], device=DEVICE
    )
    val_dataloader = DataLoader(val_dataset, batch_size=cfg["data"]["batch_size"], num_workers=4, collate_fn=custom_collate_fn,shuffle=False)

    conv_gamma = cfg["lrp"]["conv_gamma"]
    lin_gamma = cfg["lrp"]["lin_gamma"]
    distance_metric = cfg["knn"]["distance_metric"]
    k_neighbors = cfg["knn"]["k"]

    relevance_mask_dict_finetuned, violations = compute_relevances(
        model_wrapper=model_wrapper_finetuned,
        conv_gamma=conv_gamma,
        lin_gamma=lin_gamma,
        dataloader=val_dataloader,
        device=DEVICE,
        db_embeddings=val_db_embeddings_finetuned,
        db_labels=val_db_labels_finetuned,
        db_filenames=val_db_filenames_finetuned,
        k_neighbors=k_neighbors,
        mode=MODE,
        verbose=VERBOSE,
        distance_metric=distance_metric
    )

    mask_score_finetuned = defaultdict()
    for filename, (relevance,mask) in relevance_mask_dict_finetuned.items():
        mask_score_finetuned[filename] = attention_inside_mask(relevance, mask)


    print(f"Clearing model finetuned from GPU memory...")
    del model_wrapper_finetuned
    gc.collect() # Trigger Python's garbage collection
    torch.cuda.empty_cache() # Release cached memory back to the OS

    #TODO: New knn database for the base model!!!!
    model_wrapper_base, _ = get_model_wrapper(device=DEVICE, finetuned=False)
    val_db_embeddings_base, val_db_labels_base, val_db_filenames_base = get_knn_db(
        db_dir=cfg["knn"]["db_embeddings_dir"], split_name="val", dataset=val_dataset,
        model_wrapper=model_wrapper_base, model_checkpoint_path="/workspaces/bachelor_thesis_code/base", batch_size=cfg["data"]["batch_size"], device=DEVICE
    )

    relevance_mask_dict_base, _ = compute_relevances(
        model_wrapper=model_wrapper_base,
        conv_gamma=conv_gamma,
        lin_gamma=lin_gamma,
        dataloader=val_dataloader,
        device=DEVICE,
        db_embeddings=val_db_embeddings_base,
        db_labels=val_db_labels_base,
        db_filenames=val_db_filenames_base,
        k_neighbors=k_neighbors,
        mode=MODE,
        verbose=VERBOSE,
        distance_metric=distance_metric
    )

    mask_score_base = defaultdict()
    for filename, (relevance, mask) in relevance_mask_dict_base.items():
        mask_score_base[filename] = attention_inside_mask(relevance, mask)

    #compute and print the mean, max and mix mask scores for both models
    mean_mask_score_finetuned = torch.tensor(list(mask_score_finetuned.values())).mean()
    max_mask_score_finetuned = torch.tensor(list(mask_score_finetuned.values())).max()
    min_mask_score_finetuned = torch.tensor(list(mask_score_finetuned.values())).min()

    mean_mask_score_base = torch.tensor(list(mask_score_base.values())).mean()
    max_mask_score_base = torch.tensor(list(mask_score_base.values())).max()
    min_mask_score_base = torch.tensor(list(mask_score_base.values())).min()

    print(f"Finetuned Model - Mean Mask Score: {mean_mask_score_finetuned}, Max: {max_mask_score_finetuned}, Min: {min_mask_score_finetuned}")
    print(f"Base Model - Mean Mask Score: {mean_mask_score_base}, Max: {max_mask_score_base}, Min: {min_mask_score_base}")
