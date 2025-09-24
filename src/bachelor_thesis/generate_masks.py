from mask_generator import MaskGenerator
from tqdm import tqdm
import os
import torch
from utils import load_config
from typing import List, Dict, Optional, Any
from PIL import Image
import pandas as pd
import numpy as np
import argparse 

import db_utils
import coco_json_utils

def prepare_segmentation_masks(
    image_dir: str,
    filenames: List[str],
    mask_dir: str,
    mask_generator: MaskGenerator,
    cfg: Dict, 
    generate_masks_from: str = "cropped",
    batch_size: int = 32,
    verbose: bool = True,
    coco_data: Optional[Dict[str, Any]] = None,
    dataset_root_dir: Optional[str] = None
):
    if verbose:
        print(f"Checking for and generating masks in: {mask_dir}")
        
    os.makedirs(mask_dir, exist_ok=True)
    
    
    generation_jobs = []
    data_to_process_df = pd.DataFrame()

    if generate_masks_from == "cropped":
        for filename in filenames:
            mask_path = os.path.join(mask_dir, os.path.basename(filename))
            if not os.path.exists(mask_path):
                img_path = os.path.join(image_dir, os.path.basename(filename))
                generation_jobs.append({'img_path': img_path, 'mask_path': mask_path})
    
    elif generate_masks_from == "db":
        file_info_df = db_utils.gather_file_info(filenames)
        data_to_process_df = db_utils.fetch_bounding_boxes(
            file_info_df,
            db_schema=cfg['db']['db_schema'],
            feature_type=cfg['db']['feature_type']
        )

    elif generate_masks_from == "coco_json":
        if coco_data is None or dataset_root_dir is None:
            raise ValueError("coco_data and dataset_root_dir must be provided for 'coco_json' mode.")
        data_to_process_df = coco_json_utils.fetch_bounding_boxes_from_json(
            filenames, coco_data, dataset_root_dir
        )

    if not data_to_process_df.empty:
        if data_to_process_df.empty:
            print("No data could be fetched for the given files.")
            return

        for _, row in data_to_process_df.iterrows():
            # Use original_filename which is consistent across both utils
            mask_filename = row['original_filename']
            mask_path = os.path.join(mask_dir, mask_filename)
            if not os.path.exists(mask_path):
                row_dict = row.to_dict()
                row_dict["mask_path"] = mask_path
                generation_jobs.append(row_dict)

    if not generation_jobs:
        if verbose:
            print("All masks already exist. No generation needed.")
        return

    if verbose:
        print(f"Found {len(generation_jobs)} missing masks. Generating in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(generation_jobs), batch_size), desc="Generating Mask Batches"):
        batch_jobs = generation_jobs[i : i + batch_size]
        
        try:
            if generate_masks_from == "cropped":
                # This part is unchanged
                image_batch_pil = [Image.open(job['img_path']).convert("RGB") for job in batch_jobs]
                generated_masks_np = mask_generator.generate_masks_from_crops_batch(image_batch_pil)
                
                for job, mask_np in zip(batch_jobs, generated_masks_np):
                    mask_pil = Image.fromarray(mask_np * 255)
                    mask_pil.save(job['mask_path'])

            elif generate_masks_from in ["db", "coco_json"]:
                batch_df = pd.DataFrame(batch_jobs)
                
                full_image_batch = []
                box_prompt_batch = []
                jobs_with_crop_info = []

                for video_path, group in batch_df.groupby("video_path"):
                    frame_numbers = group["frame_nr"].tolist()
                    extracted_frames = db_utils.extract_frames_batch(video_path, frame_numbers)

                    for _, job_row in group.iterrows():
                        frame_nr = job_row["frame_nr"]
                        if frame_nr in extracted_frames:
                            frame_rgb = extracted_frames[frame_nr]
                            h_img, w_img, _ = frame_rgb.shape

                            if generate_masks_from == "db":
                                # DB gives normalized center x, y, w, h
                                orig_x0 = (job_row["x"] - job_row["w"] / 2) * w_img
                                orig_y0 = (job_row["y"] - job_row["h"] / 2) * h_img
                                orig_x1 = (job_row["x"] + job_row["w"] / 2) * w_img
                                orig_y1 = (job_row["y"] + job_row["h"] / 2) * h_img
                                center_x = job_row["x"] * w_img
                                center_y = job_row["y"] * h_img
                                box_w = job_row["w"] * w_img
                                box_h = job_row["h"] * h_img
                            
                            elif generate_masks_from == "coco_json":
                                # JSON gives absolute [x_min, y_min, width, height]
                                x_min, y_min, box_w, box_h = job_row["bbox"]
                                orig_x0 = x_min
                                orig_y0 = y_min
                                orig_x1 = x_min + box_w
                                orig_y1 = y_min + box_h
                                center_x = x_min + box_w / 2
                                center_y = y_min + box_h / 2

                            box_for_sam_prompt = np.array([orig_x0, orig_y0, orig_x1, orig_y1])
                            
                            side = max(box_w, box_h)
                            half_side = side / 2
                            
                            ideal_x0 = center_x - half_side
                            ideal_y0 = center_y - half_side
                            ideal_x1 = center_x + half_side
                            ideal_y1 = center_y + half_side

                            full_image_batch.append(frame_rgb)
                            box_prompt_batch.append(box_for_sam_prompt)
                            jobs_with_crop_info.append({
                                'job_info': job_row.to_dict(),
                                'ideal_box_float': np.array([ideal_x0, ideal_y0, ideal_x1, ideal_y1])
                            })
                            
                if not full_image_batch:
                    print("Warning: No valid frames could be loaded for this batch.")
                    continue

                generated_full_frame_masks_np = mask_generator.generate_masks_from_boxes_batch(full_image_batch, box_prompt_batch)

                # Crop with explicit padding and save each mask 
                for job_data, full_mask_np in zip(jobs_with_crop_info, generated_full_frame_masks_np):
                    h_img, w_img = full_mask_np.shape
                    ideal_box_float = job_data['ideal_box_float']


                    side = int(np.ceil(max(ideal_box_float[2] - ideal_box_float[0], ideal_box_float[3] - ideal_box_float[1])))

                    final_mask_np = np.zeros((side, side), dtype=np.uint8)

                    x0, y0, x1, y1 = ideal_box_float.astype(int)

                    src_x_start = max(x0, 0)
                    src_y_start = max(y0, 0)
                    src_x_end = min(x1, w_img)
                    src_y_end = min(y1, h_img)


                    slice_w = src_x_end - src_x_start
                    slice_h = src_y_end - src_y_start

                    # If there's nothing to copy, just save the blank mask.
                    if slice_w <= 0 or slice_h <= 0:
                        mask_pil = Image.fromarray(final_mask_np * 255)
                        mask_pil.save(job_data['job_info']['mask_path'])
                        continue


                    dest_x_start = max(0, -x0)
                    dest_y_start = max(0, -y0)
                    
                    dest_x_end = min(dest_x_start + slice_w, side)
                    dest_y_end = min(dest_y_start + slice_h, side)

                    final_copy_w = dest_x_end - dest_x_start
                    final_copy_h = dest_y_end - dest_y_start

                    final_mask_np[dest_y_start:dest_y_end, dest_x_start:dest_x_end] = \
                        full_mask_np[src_y_start:src_y_start + final_copy_h, src_x_start:src_x_start + final_copy_w]

                    mask_pil = Image.fromarray(final_mask_np * 255)
                    mask_pil.save(job_data['job_info']['mask_path'])

        except Exception as e:
            print(f"Warning: Failed to process a batch. Error: {e}")

    if verbose:
        print("Mask generation complete.")


if __name__ == "__main__":
    print("--- Starting Mask Generation Subprocess ---")
    parser = argparse.ArgumentParser(description="Run DINOv2 AttnLRP experiment.")
    parser.add_argument(
        "--config_name", 
        type=str, 
        required=True,
        help="The name of the experiment config (e.g., 'finetuned', 'non_finetuned')."
    )
    args, unknown_args = parser.parse_known_args()

    cfg = load_config(args.config_name, unknown_args)

    root_dir = cfg["data"]["dataset_dir"]
    dataset_name = os.path.basename(root_dir)

    if not "zoo" in root_dir:
        GENERATION_MODE = "db"  # db or "cropped"
    else:
        GENERATION_MODE = "coco_json"

    video_directory = None
    coco_data_preprocessed = None

    if GENERATION_MODE == "db":
        if 'db_schema' not in cfg['db'] or 'feature_type' not in cfg['db']:
            raise KeyError("Config file must contain 'db_schema' and 'feature_type' under the 'db' key for 'db' mode.")
    
    elif GENERATION_MODE == "coco_json":
        if "coco_json_path" not in cfg["data"] or "zoo_video_dir" not in cfg["data"]:
            raise KeyError("For 'coco_json' mode, config must contain 'coco_json_path' and 'video_dir'.")
        
        json_path = cfg["data"]["coco_json_path"]
        video_directory = cfg["data"]["zoo_video_dir"]
        coco_data_preprocessed = coco_json_utils.load_and_preprocess_coco_json(json_path)



    sam_checkpoint_path = "/workspaces/bachelor_thesis_code/sam2.1_l_finetuned.pt"#"/workspaces/vast-gorilla/gorillawatch/model_checkpoints/sam2.1_hiera_large.pt"
    config_dir = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/configs"
    mask_generator = MaskGenerator(
        model_checkpoint_path=sam_checkpoint_path,
        model_config_dir=config_dir
    )

    target_mask_dir = os.path.join(cfg["data"]["base_mask_dir"], dataset_name)

    dirs_to_process = []
    standard_splits = ["train", "validation", "test"]
    
    for split_name in standard_splits:
        potential_dir = os.path.join(root_dir, split_name)
        if os.path.isdir(potential_dir):
            dirs_to_process.append(potential_dir)

    # If no standard splits were found, fall back to using the root directory itself
    if not dirs_to_process:
        if any(f.lower().endswith((".jpg", ".png")) for f in os.listdir(root_dir)):
            print(f"No standard splits ('train', 'validation', 'test') found in '{root_dir}'.")
            print("Treating the root directory as a single dataset split.")
            dirs_to_process.append(root_dir)
        else:
            print(f"Warning: No standard splits found and no images found in the root directory '{root_dir}'. Nothing to process.")

    for split_dir in dirs_to_process:
        if not split_dir == root_dir:
            split_name = os.path.basename(split_dir)
        else:
            split_name = "" #becae data is directly in root dir then

        split_files = [f for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".png"))]
        if not split_files:
            print(f"!!! No image files found in '{split_dir}' – skipping.")
            continue

        print(f"\n-> Generating masks for '{split_name}' split using '{GENERATION_MODE}' mode...")
        
        output_mask_dir = os.path.join(target_mask_dir, split_name)
        print(f"the target mask dir is {output_mask_dir}")
        print(f"the split dir is {split_dir}")
        
        prepare_segmentation_masks(
            image_dir=split_dir,
            filenames=split_files,
            mask_dir=output_mask_dir,
            mask_generator=mask_generator,
            cfg=cfg,
            generate_masks_from=GENERATION_MODE,
            batch_size=cfg["data"]["batch_size"], #128,
            coco_data=coco_data_preprocessed,
            dataset_root_dir=video_directory 
        )

    print("--- Mask Generation Subprocess Complete ---")