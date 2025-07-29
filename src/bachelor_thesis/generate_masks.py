from mask_generator import MaskGenerator
from tqdm import tqdm
import os
import torch
from utils import load_all_configs
from typing import List, Dict
from PIL import Image
import pandas as pd
import numpy as np

# Import the new utility functions
import db_utils

def prepare_segmentation_masks(
    image_dir: str,
    filenames: List[str],
    mask_dir: str,
    mask_generator: MaskGenerator,
    cfg: Dict, # Pass the config dictionary for DB settings
    generate_masks_from: str = "cropped",
    batch_size: int = 32,
    verbose: bool = True,
):
    if verbose:
        print(f"Checking for and generating masks in: {mask_dir}")
        
    os.makedirs(mask_dir, exist_ok=True)
    
    # 1. Identify which masks need to be generated
    generation_jobs = []
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
        if data_to_process_df.empty:
            print("No data could be fetched from the database for the given files.")
            return

        for _, row in data_to_process_df.iterrows():
            mask_path = os.path.join(mask_dir, row['original_filename'])
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
    
    # 2. Process the generation jobs in batches
    for i in tqdm(range(0, len(generation_jobs), batch_size), desc="Generating Mask Batches"):
        batch_jobs = generation_jobs[i : i + batch_size]
        
        try:
            if generate_masks_from == "cropped":
                # Load a batch of pre-cropped images
                image_batch_pil = [Image.open(job['img_path']).convert("RGB") for job in batch_jobs]
                generated_masks_np = mask_generator.generate_masks_from_crops_batch(image_batch_pil)
                
                # Save each mask
                for job, mask_np in zip(batch_jobs, generated_masks_np):
                    mask_pil = Image.fromarray(mask_np * 255)
                    mask_pil.save(job['mask_path'])

            elif generate_masks_from == "db":
                batch_df = pd.DataFrame(batch_jobs)
                
                # Prepare data for SAM: full frames and box prompts
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

                            # Calculate tight box for SAM prompt (unchanged) 
                            orig_x0 = (job_row["x"] - job_row["w"] / 2) * w_img
                            orig_y0 = (job_row["y"] - job_row["h"] / 2) * h_img
                            orig_x1 = (job_row["x"] + job_row["w"] / 2) * w_img
                            orig_y1 = (job_row["y"] + job_row["h"] / 2) * h_img
                            box_for_sam_prompt = np.array([orig_x0, orig_y0, orig_x1, orig_y1])

                            center_x = job_row["x"] * w_img
                            center_y = job_row["y"] * h_img
                            side = max(job_row["w"] * w_img, job_row["h"] * h_img)
                            half_side = side / 2
                            
                            # These coordinates can be negative or larger than the image dimensions
                            ideal_x0 = center_x - half_side
                            ideal_y0 = center_y - half_side
                            ideal_x1 = center_x + half_side
                            ideal_y1 = center_y + half_side

                            full_image_batch.append(frame_rgb)
                            box_prompt_batch.append(box_for_sam_prompt)
                            jobs_with_crop_info.append({
                                'job_info': job_row,
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

                    x0, y0, x1, y1 = ideal_box_float.astype(int)
                    
                    canvas_side_w = x1 - x0
                    canvas_side_h = y1 - y0
                    

                    final_mask_np = np.zeros((canvas_side_h, canvas_side_w), dtype=np.uint8)

                    src_x_start = max(x0, 0)
                    src_y_start = max(y0, 0)
                    src_x_end = min(x1, w_img)
                    src_y_end = min(y1, h_img)

                    dest_x_start = max(0, -x0)
                    dest_y_start = max(0, -y0)
                    
                    copy_w = src_x_end - src_x_start
                    copy_h = src_y_end - src_y_start

                    if copy_w > 0 and copy_h > 0:
                        dest_x_end = dest_x_start + copy_w
                        dest_y_end = dest_y_start + copy_h

                        final_mask_np[dest_y_start:dest_y_end, dest_x_start:dest_x_end] = \
                            full_mask_np[src_y_start:src_y_end, src_x_start:src_x_end]

                    mask_pil = Image.fromarray(final_mask_np * 255)
                    mask_pil.save(job_data['job_info']['mask_path'])

        except Exception as e:
            print(f"Warning: Failed to process a batch. Error: {e}")

    if verbose:
        print("Mask generation complete.")

if __name__ == "__main__":
    print("--- Starting Mask Generation Subprocess ---")
    config_dir = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/configs"
    cfg = load_all_configs(config_dir)

    # --- Add new DB configurations to your config files ---
    # Example for your config yaml:
    # data:
    #   db_schema: "public"
    #   feature_type: "body"
    #   ... other keys
    #
    # Ensure these keys exist in your `cfg` dictionary
    if 'db_schema' not in cfg['db'] or 'feature_type' not in cfg['db']:
        raise KeyError("Config file must contain 'db_schema' and 'feature_type' under the 'db' key for 'db' mode.")

    root_dir = cfg["data"]["dataset_dir"]
    dataset_name = os.path.basename(root_dir)

    sam_checkpoint_path = "/workspaces/vast-gorilla/gorillawatch/model_checkpoints/sam2.1_hiera_large.pt"
    mask_generator = MaskGenerator(
        model_checkpoint_path=sam_checkpoint_path,
        model_config_dir=config_dir
    )

    target_mask_dir = os.path.join(cfg["data"]["base_mask_dir"], dataset_name)
    
    # CHOOSE YOUR MODE HERE
    GENERATION_MODE = "db" # db or "cropped"

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            print(f"!!! Skipping '{split}' – directory not found.")
            continue

        split_files = [f for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".png"))]
        if not split_files:
            print(f"!!! No image files found in '{split}' – skipping.")
            continue

        print(f"-> Generating masks for '{split}' split using '{GENERATION_MODE}' mode...")
        prepare_segmentation_masks(
            image_dir=split_dir,
            filenames=split_files,
            mask_dir=os.path.join(target_mask_dir, split),
            mask_generator=mask_generator,
            cfg=cfg,
            generate_masks_from=GENERATION_MODE,
            batch_size=16 
        )

    print("--- Mask Generation Subprocess Complete ---")