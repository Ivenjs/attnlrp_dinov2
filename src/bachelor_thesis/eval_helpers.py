from collections import defaultdict
import numpy as np
from pydash import result
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from basemodel import TimmWrapper
from typing import Any, Tuple, List, Dict, Callable
from knn_helpers import calculate_distance
from lrp_helpers import compute_knn_proxy_soft_all, compute_knn_proto_margin, compute_similarity_score, compute_knn_proxy_soft_topk
from utils import deterministic_randperm, parse_encounter_id
from model_evaluation import perform_knn_ce_evaluation


PATCH_SIZE = 14  # Size of the patches to average over

def calculate_auc(curve: torch.Tensor) -> float:
    """Calculates the Area Under the Curve using the mean, as in the paper."""
    # The paper defines the area as (1/N) * sum(f_j(x_k)).
    # This is equivalent to the mean of the curve points.
    return torch.mean(curve).item()


def _run_perturbation_experiment(
    model: torch.nn.Module, # UN-PATCHED model
    input_tensor: torch.Tensor,
    patch_order: torch.Tensor,
    perturbation_type: str,
    patch_size: int,
    score_fn: Callable,
    score_fn_kwargs: Dict[str, Any],
    patches_per_step: int = 1,
    baseline_value: str = "black",
) -> torch.Tensor:
    """
    Runs a generic perturbation experiment, tracking a given score at each step.

    Args:
        model: The model to evaluate.
        input_tensor: The input image tensor.
        patch_order: The order in which to perturb patches.
        perturbation_type: 'deletion' or 'insertion'.
        patch_size: The size of each square patch.
        score_fn: A callable function that takes an embedding and kwargs and returns a scalar score.
        score_fn_kwargs: A dictionary of keyword arguments to pass to score_fn.
        patches_per_step: Number of patches to perturb in each step.
        baseline_value: 'black' or 'mean' for the perturbation baseline.
    """

    model.eval()

    if baseline_value.lower() == "black":
            baseline_fill = torch.zeros_like(input_tensor)
    elif baseline_value.lower() == "mean":
        mean_color = input_tensor.mean(dim=[2, 3], keepdim=True)
        baseline_fill = mean_color.expand_as(input_tensor)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_value}")

    # Calculate the initial, unperturbed k-NN proxy score
    with torch.no_grad():
        initial_embedding = model(input_tensor)
        result = score_fn(initial_embedding, **score_fn_kwargs)
        if isinstance(result, tuple):
            initial_score = result[0] #similarity returns reference embedding for easy lookup
        else:
            initial_score = result

    num_patches = len(patch_order)
    h, w = input_tensor.shape[-2:]
    num_patches_w = w // patch_size


    output_scores = [initial_score.item()]

    if perturbation_type == 'deletion':
        perturbed_tensor = input_tensor.clone()
    else: # insertion
        perturbed_tensor = baseline_fill.clone()


    patches_processed_so_far = 0
    
    pbar = tqdm(total=num_patches, desc=f"{perturbation_type.capitalize()} Eval (Granular)")

    while patches_processed_so_far < num_patches:
        start_idx = patches_processed_so_far
        end_idx = min(start_idx + patches_per_step, num_patches)
        patches_to_process = patch_order[start_idx:end_idx]

        for patch_idx in patches_to_process:
            row = (patch_idx // num_patches_w) * patch_size
            col = (patch_idx % num_patches_w) * patch_size

            if perturbation_type == 'deletion':
                perturbed_tensor[..., row:row+patch_size, col:col+patch_size] = \
                    baseline_fill[..., row:row+patch_size, col:col+patch_size]
            else: # insertion
                original_patch = input_tensor[..., row:row+patch_size, col:col+patch_size]
                perturbed_tensor[..., row:row+patch_size, col:col+patch_size] = original_patch

        # After perturbing the chunk, run the model and get the score
        with torch.no_grad():
            current_embedding = model(perturbed_tensor)
            result = score_fn(current_embedding, **score_fn_kwargs)
            if isinstance(result, tuple):
                score = result[0] #similarity returns reference embedding for easy lookup
            else:
                score = result
            output_scores.append(score.item())

        # Update progress
        num_in_chunk = end_idx - start_idx
        patches_processed_so_far += num_in_chunk
        pbar.update(num_in_chunk)
    
    pbar.close()

    # Convert the list of scores to a tensor for calculations
    return torch.tensor(output_scores, device=input_tensor.device)


def srg_eval(
    relevance_map: torch.Tensor,
    input_tensor: torch.Tensor,
    model: TimmWrapper, # UN-PATCHED model
    mode: str,
    patch_size: int,
    patches_per_step: int,
    input_filename: str,
    baseline_value: str = "black",
    plot_curves: bool = False,
    seed=161,
    **kwargs
) -> Dict:
    """
    Calculates the Faithfulness Score (LeRF_AUC - MoRF_AUC) for a given explanation mode.

    The score function used for perturbation is determined by the `mode`.
    All mode-specific arguments (e.g., db_embeddings, query_label) must be passed via **kwargs.
    """
    # relevance should be shape (1,1,H,W)
    patch_relevance = F.avg_pool2d(relevance_map, kernel_size=patch_size, stride=patch_size)
    patch_relevance_flat = patch_relevance.flatten()

    lerf_order = torch.argsort(patch_relevance_flat, descending=False)
    morf_order = torch.argsort(patch_relevance_flat, descending=True)

    if mode == "soft_knn_margin_all":
        score_fn = compute_knn_proxy_soft_all
        score_fn_kwargs = {
            "query_label": kwargs["query_label"],
            "query_filename": kwargs["query_filename"],
            "query_video_id": kwargs["query_video_id"],
            "db_embeddings": kwargs["db_embeddings"],
            "db_labels": kwargs["db_labels"],
            "db_filenames": kwargs["db_filenames"],
            "db_video_ids": kwargs["db_video_ids"],
            "distance_metric": kwargs["distance_metric"],
            "temp": kwargs["proxy_temp"],
            "cross_encounter": kwargs["cross_encounter"]
        }
    elif mode == "soft_knn_margin_topk":
        score_fn = compute_knn_proxy_soft_topk
        score_fn_kwargs = {
            "query_label": kwargs["query_label"],
            "query_filename": kwargs["query_filename"],
            "query_video_id": kwargs["query_video_id"],
            "db_embeddings": kwargs["db_embeddings"],
            "db_labels": kwargs["db_labels"],
            "db_filenames": kwargs["db_filenames"],
            "db_video_ids": kwargs["db_video_ids"],
            "distance_metric": kwargs["distance_metric"],
            "temp": kwargs["proxy_temp"],
            "topk": kwargs.get("topk", 5),
            "cross_encounter": kwargs["cross_encounter"]
        }
    elif mode == "proto_margin":
        score_fn = compute_knn_proto_margin
        score_fn_kwargs = {
            "query_label": kwargs["query_label"],
            "query_filename": kwargs["query_filename"],
            "query_video_id": kwargs["query_video_id"],
            "db_embeddings": kwargs["db_embeddings"],
            "db_labels": kwargs["db_labels"],
            "db_filenames": kwargs["db_filenames"],
            "db_video_ids": kwargs["db_video_ids"],
            "distance_metric": kwargs["distance_metric"],
            "temp": kwargs["proxy_temp"],
            "topk_neg": kwargs.get("topk", 50),
            "cross_encounter": kwargs["cross_encounter"]
        }
    elif mode == "similarity":
        score_fn = compute_similarity_score
        score_fn_kwargs = {
            "query_label": kwargs["query_label"],
            "query_filename": kwargs["query_filename"],
            "query_video_id": kwargs["query_video_id"],
            "db_embeddings": kwargs["db_embeddings"],
            "db_labels": kwargs["db_labels"],
            "db_filenames": kwargs["db_filenames"],
            "db_video_ids": kwargs["db_video_ids"],
            "cross_encounter": kwargs["cross_encounter"],
            "reference_embedding": kwargs["reference_embedding"]
        }
    else:
        raise ValueError(f"Unsupported evaluation mode: '{mode}'")

    perturb_args = {
        "patch_size": patch_size,
        "score_fn": score_fn,
        "score_fn_kwargs": score_fn_kwargs,
        "patches_per_step": patches_per_step,
        "baseline_value": baseline_value
    }

    order = deterministic_randperm(len(morf_order), input_filename, seed)

    morf_curve = _run_perturbation_experiment(model, input_tensor, morf_order, 'deletion', **perturb_args)
    lerf_curve = _run_perturbation_experiment(model, input_tensor, lerf_order, 'deletion', **perturb_args)
    random_curve = _run_perturbation_experiment(
        model, input_tensor, order, 'deletion', **perturb_args
    )

    auc_morf = calculate_auc(morf_curve)
    auc_lerf = calculate_auc(lerf_curve)
    auc_random = calculate_auc(random_curve)

    faithfulness_score = auc_lerf - auc_morf
    morf_vs_random = auc_random - auc_morf
    lerf_vs_random = auc_lerf - auc_random

    print(f"--- SRG Results for Mode: '{mode}' ---")
    print(f"Area under LeRF curve: {auc_lerf:.4f}")
    print(f"Area under MoRF curve: {auc_morf:.4f}")
    print(f"Faithfulness Score (LeRF - MoRF): {faithfulness_score:.4f}")
    print(f"Improvement vs. Random (Random - MoRF): {morf_vs_random:.4f}")
    print(f"Improvement vs. Random (LeRF - Random): {lerf_vs_random:.4f}")
    print("-" * 20)
    
    results = {
        'faithfulness_score': faithfulness_score,
        'morf_vs_random': morf_vs_random,
        'lerf_vs_random': lerf_vs_random,
        'auc_morf': auc_morf,
        'auc_lerf': auc_lerf,
        'auc_random': auc_random,
        'morf_curve': morf_curve.cpu().numpy(),
        'lerf_curve': lerf_curve.cpu().numpy(),
        'random_curve': random_curve.cpu().numpy()
    }
    

    

    if plot_curves:
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        num_eval_steps = len(results['morf_curve'])
        x_axis = np.linspace(0, 100, num_eval_steps)
        
        ax.plot(x_axis, results['morf_curve'], label=f"MoRF (AUC={results['auc_morf']:.3f})", color='red')
        ax.plot(x_axis, results['lerf_curve'], label=f"LeRF (AUC={results['auc_lerf']:.3f})", color='blue')
        ax.plot(x_axis, results['random_curve'], label=f"Random (AUC={results['auc_random']:.3f})", color='gray', linestyle='--')
        ax.set_title(f"Perturbation Curves (Mode: {mode.title()})")
        ax.set_xlabel('% of Patches Removed')
        ax.set_ylabel(f"{mode.replace('_', ' ').title()} Score")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.suptitle(f'Faithfulness Evaluation for {kwargs.get("query_filename", "Unknown Image")}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show() # Or save to a file
        plt.close(fig)

    return {mode: results}

def _run_downstream_perturbation_sweep(
    model: torch.nn.Module,
    query_images_tensor: torch.Tensor,
    # ### MODIFIED ###: Now accepts pre-processed query metadata
    query_labels_int: torch.Tensor,
    query_encounter_ids_int: torch.Tensor,
    query_original_indices: torch.Tensor,
    all_patch_orders: torch.Tensor,
    # ### MODIFIED ###: Now accepts pre-processed DB metadata and config
    db_embeddings: torch.Tensor,
    db_labels_int: torch.Tensor,
    db_encounter_ids_int: torch.Tensor,
    cfg: Dict[str, Any], # Pass the config dict for knn params
    patch_size: int,
    patches_per_step: int,
    baseline_value: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Runs a perturbation sweep and evaluates downstream k-NN accuracy at each step.
    This version is adapted to use the specific `perform_knn_ce_evaluation` function.
    """
    model.eval()
    num_patches = all_patch_orders.shape[1]
    
    # ### MODIFIED ###: Collect all common arguments for the evaluation function
    eval_kwargs = {
        "db_embeddings": db_embeddings,
        "db_labels_int": db_labels_int,
        "db_encounter_ids_int": db_encounter_ids_int,
        "k": cfg["knn"]["k"],
        "batch_size": cfg["data"]["batch_size"],
        "device": device,
        "distance_metric": cfg["knn"]["distance_metric"],
        "query_labels_int": query_labels_int,
        "query_encounter_ids_int": query_encounter_ids_int,
        "query_original_indices": query_original_indices,
    }

    # --- 1. Get Baseline (0% perturbation) Accuracy ---
    with torch.no_grad():
        original_query_embeddings = []
        # Use a simple TensorDataset for efficient batching
        temp_dataset = torch.utils.data.TensorDataset(query_images_tensor)
        temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=cfg["data"]["batch_size"])
        for batch in temp_loader:
             original_query_embeddings.append(model(batch[0].to(device)))
        original_query_embeddings = torch.cat(original_query_embeddings)

    print("  Calculating baseline accuracy (0% perturbation)...")
    baseline_accuracy = perform_knn_ce_evaluation(
        query_embeddings=original_query_embeddings,
        **eval_kwargs
    )
    
    accuracy_curve = [baseline_accuracy]
    
    # --- 2. Iteratively Perturb and Evaluate ---
    perturbed_images = query_images_tensor.clone().to(device)
    pbar = tqdm(total=num_patches, desc="Running Downstream Perturbation Sweep", leave=False)
    
    patches_processed_so_far = 0
    while patches_processed_so_far < num_patches:
        start_idx = patches_processed_so_far
        end_idx = min(start_idx + patches_per_step, num_patches)
        
        perturbed_images = apply_perturbation_to_batch(
            perturbed_images, all_patch_orders, start_idx, end_idx, patch_size, baseline_value
        )
        
        with torch.no_grad():
            current_query_embeddings = []
            temp_dataset = torch.utils.data.TensorDataset(perturbed_images)
            temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=cfg["data"]["batch_size"])
            for batch in temp_loader:
                current_query_embeddings.append(model(batch[0]))
            current_query_embeddings = torch.cat(current_query_embeddings)

        current_accuracy = perform_knn_ce_evaluation(
            query_embeddings=current_query_embeddings,
            **eval_kwargs
        )
        accuracy_curve.append(current_accuracy)
        
        num_in_chunk = end_idx - start_idx
        patches_processed_so_far += num_in_chunk
        pbar.update(num_in_chunk)
        
    pbar.close()
    
    return torch.tensor(accuracy_curve, device=device)

def evaluate_faithfulness_downstream(
    relevance_maps_dict: Dict[str, torch.Tensor],
    query_dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    # ### MODIFIED ###: Accepts pre-processed data and config
    db_embeddings: torch.Tensor,
    db_labels_int: torch.Tensor,
    db_encounter_ids_int: torch.Tensor,
    label_to_id: Dict[str, int],
    encounter_to_id: Dict[str, int],
    cfg: Dict[str, Any],
    patch_size: int,
    patches_per_step: int,
    baseline_value: str = "black",
    plot_curves: bool = True,
    seed=161,
) -> Dict:
    device = db_embeddings.device
    model.to(device)

    # --- 1. Prepare ALL Query Data and Relevance Orders ---
    print("Pre-computing perturbation orders and preparing query metadata...")
    query_images_list, query_labels_list, query_videos_list = [], [], []
    query_original_indices_list = [] # ### MODIFIED ###: Collect original indices
    morf_orders, lerf_orders, random_orders = [], [], []

    # Filter query_dataset to only include images with relevance maps
    valid_indices = [
        i for i, sample in enumerate(query_dataset) 
        if sample['filename'] in relevance_maps_dict
    ]
    query_subset = torch.utils.data.Subset(query_dataset, valid_indices)

    for sample in tqdm(query_subset, desc="Preparing Queries", leave=False):
        filename_no_ext = sample['filename']
        
        query_images_list.append(sample['image'])
        query_labels_list.append(sample['label'])
        query_videos_list.append(sample['video'])
        query_original_indices_list.append(sample['original_index']) # ### MODIFIED ###

        relevance_map = relevance_maps_dict[filename_no_ext]
        patch_relevance = F.avg_pool2d(relevance_map, kernel_size=patch_size, stride=patch_size)
        patch_relevance_flat = patch_relevance.flatten()

        morf_orders.append(torch.argsort(patch_relevance_flat, descending=True))
        lerf_orders.append(torch.argsort(patch_relevance_flat, descending=False))
        random_orders.append(deterministic_randperm(len(patch_relevance_flat), filename_no_ext, seed))

    query_images_tensor = torch.stack(query_images_list)
    morf_orders_tensor = torch.stack(morf_orders).to(device)
    lerf_orders_tensor = torch.stack(lerf_orders).to(device)
    random_orders_tensor = torch.stack(random_orders).to(device)
    
    # ### MODIFIED ###: Convert query metadata to integer tensors
    query_labels_int = torch.tensor([label_to_id[s] for s in query_labels_list], dtype=torch.long, device=device)
    query_encounters = [parse_encounter_id(v) for v in query_videos_list]
    query_encounter_ids_int = torch.tensor([encounter_to_id[s] for s in query_encounters], dtype=torch.long, device=device)
    query_original_indices_tensor = torch.tensor(query_original_indices_list, dtype=torch.long, device=device)

    print(f"Prepared {len(query_images_list)} images for evaluation.")

    sweep_args = {
        "model": model,
        "query_images_tensor": query_images_tensor,
        "query_labels_int": query_labels_int,
        "query_encounter_ids_int": query_encounter_ids_int,
        "query_original_indices": query_original_indices_tensor,
        "db_embeddings": db_embeddings,
        "db_labels_int": db_labels_int,
        "db_encounter_ids_int": db_encounter_ids_int,
        "cfg": cfg,
        "patch_size": patch_size,
        "patches_per_step": patches_per_step,
        "baseline_value": baseline_value,
        "device": device,
    }

    morf_curve = _run_downstream_perturbation_sweep(all_patch_orders=morf_orders_tensor, **sweep_args)
    lerf_curve = _run_downstream_perturbation_sweep(all_patch_orders=lerf_orders_tensor, **sweep_args)
    random_curve = _run_downstream_perturbation_sweep(all_patch_orders=random_orders_tensor, **sweep_args)
    
    # --- 3. Calculate Final Metrics ---
    # The paper's AUC definition is the mean. Here, for accuracy curves,
    # a proper trapezoidal AUC is more standard.
    # Let's stick to your definition for consistency.
    auc_morf = calculate_auc(morf_curve)
    auc_lerf = calculate_auc(lerf_curve)
    auc_random = calculate_auc(random_curve)

    # Note: For accuracy, MoRF should drop fastest (lowest AUC), LeRF slowest (highest AUC)
    # The "faithfulness" score could be defined as AUC(LeRF) - AUC(MoRF)
    faithfulness_score = auc_lerf - auc_morf
    morf_vs_random = auc_random - auc_morf
    lerf_vs_random = auc_lerf - auc_random

    print("\n--- Downstream Faithfulness Results ---")
    print(f"Area under LeRF curve: {auc_lerf:.4f}")
    print(f"Area under MoRF curve: {auc_morf:.4f}")
    print(f"Faithfulness Score (LeRF - MoRF): {faithfulness_score:.4f}")
    print(f"Improvement vs. Random (Random - MoRF): {morf_vs_random:.4f}")
    print(f"Improvement vs. Random (LeRF - Random): {lerf_vs_random:.4f}")
    print("-" * 20)

    results = {
        'faithfulness_score': faithfulness_score,
        'morf_vs_random': morf_vs_random,
        'lerf_vs_random': lerf_vs_random,
        'auc_morf': auc_morf,
        'auc_lerf': auc_lerf,
        'auc_random': auc_random,
        'morf_curve': morf_curve.cpu().numpy(),
        'lerf_curve': lerf_curve.cpu().numpy(),
        'random_curve': random_curve.cpu().numpy()
    }

    if plot_curves:
        pass
    return results
    #return {"downstream_accuracy": results}
    
def run_downstream_sweep_and_log(
    all_relevance_results: List[Dict], # The same input as before
    query_dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    db_embeddings: torch.Tensor,
    db_labels: List[str],
    db_videos: List[str],
    cfg: Dict[str, Any],
    patch_size: int,
    patches_per_step: int,
    baseline_value: str = "black",
    plot_curves: bool = False, # Set to False to avoid plotting in a loop
    seed=161
) -> tuple[list[dict], list[dict]]:
    """
    Evaluates downstream faithfulness for multiple sets of relevance maps (e.g., a gamma sweep).
    
    This function groups relevance maps by their generation parameters, runs the batched
    downstream evaluation for each group, and formats the results into flat and long-form
    dataframes for easy analysis and logging.

    Returns:
        - A pandas DataFrame with summary statistics (AUCs, faithfulness) for each param set.
        - A pandas DataFrame with the raw curve data (step, score) for each param set.
    """
    device = db_embeddings.device
    unique_labels = sorted(list(set(db_labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    db_labels_int = torch.tensor([label_to_id[s] for s in db_labels], dtype=torch.long, device=device)

    db_encounters = [parse_encounter_id(v) for v in db_videos]
    unique_encounters = sorted(list(set(db_encounters)))
    encounter_to_id = {enc: i for i, enc in enumerate(unique_encounters)}
    db_encounter_ids_int = torch.tensor([encounter_to_id[s] for s in db_encounters], dtype=torch.long, device=device)

    summary_results_list = []
    all_curves_data_list = []

    # --- 1. Group relevance maps by their generation parameters ---
    # The key will be a tuple of sorted parameter items to ensure uniqueness
    # The value will be a dictionary of {filename: relevance_map}
    relevances_by_params = defaultdict(dict)
    
    for res in all_relevance_results:
        # Create a stable, hashable key from the parameters dictionary
        params_key = tuple(sorted(res['params'].items()))
        filename = res['filename'] # Assuming filename now has no extension
        relevances_by_params[params_key][filename] = res['relevance']

    print(f"Found {len(relevances_by_params)} unique parameter combinations to evaluate.")

    # --- 2. Iterate over each parameter set and run the full downstream evaluation ---
    for params_key, relevance_maps_dict in tqdm(relevances_by_params.items(), desc="Evaluating Parameter Sets"):
        params_dict = dict(params_key)
        print(f"\n--- Evaluating parameters: {params_dict} ---")

        # Run the full downstream evaluation for this set of relevance maps
        srg_results = evaluate_faithfulness_downstream(
            relevance_maps_dict=relevance_maps_dict,
            query_dataset=query_dataset,
            model=model,
            db_embeddings=db_embeddings,
            # ### MODIFIED ###: Pass all the pre-computed data
            db_labels_int=db_labels_int,
            db_encounter_ids_int=db_encounter_ids_int,
            label_to_id=label_to_id,
            encounter_to_id=encounter_to_id,
            cfg=cfg,
            patch_size=patch_size,
            patches_per_step=patches_per_step,
            baseline_value=baseline_value,
            plot_curves=plot_curves,
            seed=seed,
        )

        # --- 3. Format and store the results for this parameter set ---
        
        # a) Create the flat summary row
        summary_row = {
            **params_dict,  # Unpack conv_gamma, lin_gamma, etc.
            "metric_name": cfg["lrp"]["mode"],
            "faithfulness_score": srg_results["faithfulness_score"],
            "morf_vs_random": srg_results["morf_vs_random"],
            "lerf_vs_random": srg_results["lerf_vs_random"],
            "auc_morf": srg_results["auc_morf"],
            "auc_lerf": srg_results["auc_lerf"],
            "auc_random": srg_results["auc_random"],
        }
        summary_results_list.append(summary_row)

        # b) Create the long-form curve data
        all_curve_sets = [
            ("morf_raw", srg_results["morf_curve"]),
            ("lerf_raw", srg_results["lerf_curve"]),
            ("random_raw", srg_results["random_curve"])
        ]

        for curve_label, curve in all_curve_sets:
            # The x-axis represents the percentage of patches perturbed
            num_steps = len(curve)
            x_axis = np.linspace(0, 100, num_steps)

            for step_idx, score in enumerate(curve):
                curve_point_row = {
                    **params_dict,
                    "metric_name": cfg["lrp"]["mode"],
                    "curve_label": curve_label,
                    "step": step_idx,
                    "percent_perturbed": x_axis[step_idx],
                    "score": score
                }
                all_curves_data_list.append(curve_point_row)
    
    return summary_results_list, all_curves_data_list

def apply_perturbation_to_batch(
    image_batch: torch.Tensor,
    patch_orders: torch.Tensor,
    step_start_idx: int,
    step_end_idx: int,
    patch_size: int,
    baseline_value: str = "mean",
) -> torch.Tensor:
    """
    Applies perturbation to a batch of images based on pre-computed patch orders.

    Args:
        image_batch: The tensor of images to perturb [N, C, H, W].
        patch_orders: The full perturbation order for each image [N, num_patches].
        step_start_idx: The starting index of patches to perturb in this step.
        step_end_idx: The ending index of patches to perturb in this step.
        patch_size: The size of each square patch.
        baseline_value: The baseline to use for perturbation.
    """
    if baseline_value.lower() == "black":
        # Create baseline once, then broadcast
        baseline_fill_patch = torch.zeros(
            (1, image_batch.shape[1], patch_size, patch_size),
            device=image_batch.device,
            dtype=image_batch.dtype
        )
    elif baseline_value.lower() == "mean":
        # 'mean' baseline is image-specific. We handle this inside the loop.
        pass
    else:
        raise ValueError(f"Unknown baseline type: {baseline_value}")

    h, w = image_batch.shape[-2:]
    num_patches_w = w // patch_size
    
    patches_to_perturb_this_step = patch_orders[:, step_start_idx:step_end_idx]

    for i in range(image_batch.shape[0]):  
        if baseline_value.lower() == "mean":
             mean_color = image_batch[i].mean(dim=[1, 2], keepdim=True)
             baseline_fill_patch = mean_color.expand(
                 -1, patch_size, patch_size
             ).unsqueeze(0)


        for patch_idx in patches_to_perturb_this_step[i]:
            patch_idx = patch_idx.item()
            row = (patch_idx // num_patches_w) * patch_size
            col = (patch_idx % num_patches_w) * patch_size
            
            image_batch[i, :, row:row+patch_size, col:col+patch_size] = baseline_fill_patch

    return image_batch

def attention_inside_mask(relevance: torch.Tensor, mask: np.ndarray) -> Tuple[float, float, float]:
    """
    Args:
        mask (np.ndarray): Boolean or binary array of shape (1 ,H, W).
        relevance (torch.Tensor): Tensor of shape (1, 1, H, W) with relevance scores.

    Returns:
        Tuple[float, float, float]:
            (total_fraction, positive_fraction, negative_fraction)
    """
    target_size = relevance.shape[-2:]  # e.g., (518, 518)
    mask_tensor = torch.from_numpy(mask).to(relevance.device)

    if mask_tensor.shape[-2:] != target_size:
        # F.interpolate needs a 4D input (N, C, H, W), so we add a temporary batch dimension.
        # We must also convert to float for the interpolation operation.
        # 'nearest' mode is crucial for masks to avoid creating non-binary values.
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0).float(),
            size=target_size,
            mode='nearest'
        ).squeeze(0) # Remove the temporary batch dimension.

    # From this point on, mask_tensor is guaranteed to have the same H, W as relevance.

    relevance_squeezed = relevance.squeeze(0).squeeze(0)  # from (1, 1, H, W) -> (H, W)
    boolean_mask = mask_tensor.squeeze(0).bool() # from (1, H, W) -> (H, W)

    relevance_inside = relevance_squeezed[boolean_mask]

    total_abs_relevance = relevance_squeezed.abs().sum()
    inside_abs_relevance = relevance_inside.abs().sum()

    positive_relevance = relevance_squeezed.clamp(min=0)
    negative_relevance = relevance_squeezed.clamp(max=0).abs()

    pos_inside = positive_relevance[boolean_mask].sum()
    pos_total = positive_relevance.sum()

    neg_inside = negative_relevance[boolean_mask].sum()
    neg_total = negative_relevance.sum()

    # Calculate fractions, guarding against division by zero
    total_frac = (inside_abs_relevance / total_abs_relevance).item() if total_abs_relevance > 0 else 0.0
    pos_frac = (pos_inside / pos_total).item() if pos_total > 0 else 0.0
    neg_frac = (neg_inside / neg_total).item() if neg_total > 0 else 0.0

    return total_frac, pos_frac, neg_frac

def get_query_performance_metrics(
    query_embedding: torch.Tensor,
    query_label: str,
    query_filename: str,
    db_embeddings: torch.Tensor,
    db_labels: list[str],
    db_filenames: list[str],
    distance_metric: str = "cosine",
    k: int = 5 
) -> dict:
    """
    Computes multiple performance metrics for a single query against a database.
    """
    device = query_embedding.device
    db_labels_np = np.array(db_labels)
    db_filenames_np = np.array(db_filenames)

    with torch.no_grad():
        distances = calculate_distance(query_embedding, db_embeddings, distance_metric)
        try:
            query_idx = db_filenames.index(query_filename)
            distances[query_idx] = torch.inf
        except ValueError:
            pass

        sorted_indices = torch.argsort(distances)
        sorted_labels = db_labels_np[sorted_indices.cpu().numpy()]

    match_indices = np.where(sorted_labels == query_label)[0]
    rank = match_indices[0] + 1 if len(match_indices) > 0 else -1

    gt_positive_mask = (db_labels_np == query_label) & (db_filenames_np != query_filename)
    gt_positive_indices = np.where(gt_positive_mask)[0]
    
    gt_similarity = -1.0
    if gt_positive_indices.size > 0:
        min_gt_distance = torch.min(distances[gt_positive_indices])
        gt_similarity = 1.0 - min_gt_distance.item()

    recall_at_k = 0.0
    num_gt_positives = gt_positive_indices.size
    if num_gt_positives > 0:
        # Get the indices of the top-k results
        top_k_indices = sorted_indices[:k].cpu().numpy()
        
        # Count how many of the top-k results are ground-truth positives
        # Using sets is efficient for this intersection
        top_k_set = set(top_k_indices)
        gt_positives_set = set(gt_positive_indices)
        
        num_correct_in_top_k = len(top_k_set.intersection(gt_positives_set))
        
        recall_at_k = num_correct_in_top_k / num_gt_positives

    return {
        'rank': rank,
        'gt_similarity': gt_similarity,
        'recall_at_k': recall_at_k
    }

