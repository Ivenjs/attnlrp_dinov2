import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import hydra
from hydra import initialize, compose
import numpy as np
import pandas as pd
import torch
from db_connect import get_db_connection
from decord import VideoReader, cpu
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm

# --- Configuration ---
# Path to save the output crops and overlays
OUTPUT_PATH = "/workspaces/bachelor_thesis_code/seg_test_out"
# Path to the SAM model checkpoint
MODEL_CHECKPOINT = "/workspaces/vast-gorilla/gorillawatch/model_checkpoints/sam2.1_hiera_large.pt"
# Path to the directory containing model config YAMLs
MODEL_CFG_PATH = "configs"

# Processing parameters
DB_SCHEMA = "public"
FEATURE_TYPE = "body"  # 'body', 'body_face', 'face_45' or 'face_90'
BATCH_SIZE = 64        # Number of detections to process in one model forward pass
MAX_WORKERS = 64       # For saving images in parallel
DB_QUERY_BATCH_SIZE = 1000 # How many records to fetch from the DB at once

# --- Model and Device Setup ---
def setup_model_and_device():
    """Initializes the device (GPU/CPU) and the SAM model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    script_dir = Path(__file__).parent.resolve()
    configs_path = script_dir / "configs"
    
    # Ensure the configs directory exists
    if not configs_path.exists():
        raise FileNotFoundError(f"Configs directory not found at: {configs_path}")
    
    # The SAM2 import initializes hydra, which can cause issues with paths.
    # This resets hydra to ensure it finds our configs correctly.
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # Use initialize_config_dir with absolute path instead of initialize_config_module
    hydra.initialize_config_dir(config_dir=str(configs_path), version_base="1.2")

    sam2_model = build_sam2("sam2.1_hiera_l.yaml", MODEL_CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor, device

def crop_and_resize_mask(full_mask, box, target_size=(518, 518)):
    """
    Crops a high-resolution mask using a bounding box and resizes it to a target size.

    Args:
        full_mask (np.ndarray): The high-resolution binary mask from SAM.
        box (np.ndarray): The bounding box [x0, y0, x1, y1].
        target_size (tuple): The target (width, height) for the output mask.

    Returns:
        np.ndarray: The resized, floating-point mask of shape target_size.
    """
    # Ensure box coordinates are integers and within the mask's bounds
    h_mask, w_mask = full_mask.shape
    x0, y0, x1, y1 = [int(c) for c in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w_mask, x1), min(h_mask, y1)

    # Handle cases where the box is invalid or has zero area
    if x1 <= x0 or y1 <= y0:
        return np.zeros(target_size, dtype=np.float32)

    # Crop the mask to the bounding box
    cropped_mask = full_mask[y0:y1, x0:x1]

    # Resize the cropped mask. INTER_AREA is best for downsampling.
    # We convert to float32 so the output is a float array representing coverage.
    resized_mask = cv2.resize(
        cropped_mask.astype(np.float32),
        target_size,
        interpolation=cv2.INTER_AREA
    )
    
    return resized_mask

def crop_masked_region(image_rgb, mask, save_path):
    """Crops the masked region from an image and saves it as a PNG with a transparent background."""
    mask = mask.astype(np.uint8)
    if mask.shape[:2] != image_rgb.shape[:2]:
        mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    y_indices, x_indices = np.where(mask != 0)
    if len(y_indices) == 0:
        return # Skip empty masks
    
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    cropped_img = image_rgb[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]

    # Create 4-channel RGBA image
    rgba = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = cropped_mask * 255  # Use mask for the alpha channel
    
    cv2.imwrite(save_path, rgba)

def overlay_mask_on_image(image_rgb, mask, box, save_path, color=(30, 144, 255), alpha=0.6):
    """Overlays a colored mask and bounding box on an image and saves it."""
    overlay = image_rgb.copy()
    
    # Create a colored version of the mask
    colored_mask = np.zeros_like(overlay, dtype=np.uint8)
    colored_mask[mask == 1] = color
    
    # Blend the overlay
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)

    # Draw the bounding box
    x0, y0, x1, y1 = [int(c) for c in box]
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2) # Green box
    
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# --- Data Fetching and Batching ---
def gather_file_info(image_dir):
    """Walks the input directory and parses frame_nr, tracking_id, and split from filenames."""
    records = []
    image_dir = Path(image_dir).resolve()

    for dirpath, _, filenames in os.walk(image_dir):
        dirpath = Path(dirpath).resolve()
        
        # "" if we're in root dir, else use subfolder name as split
        split = "" if dirpath == image_dir else dirpath.name

        for filename in filenames:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            try:
                parts = Path(filename).stem.split('_')
                frame_nr = int(parts[-2])
                tracking_id = int(parts[-1])
                records.append({
                    "frame_nr": frame_nr,
                    "tracking_id": tracking_id,
                    "split": split
                })
            except (ValueError, IndexError):
                print(f"Warning: Could not parse filename '{filename}'. Skipping.")
                
    return pd.DataFrame(records)

def fetch_bounding_boxes(file_df):
    """Fetches bounding box data from the database in batches for the given file info."""
    print(f"Fetching bounding box data for {len(file_df)} detections from the database...")
    all_bbox_data = []
    params_to_query = list(zip(file_df['frame_nr'], file_df['tracking_id']))

    with get_db_connection(schema=DB_SCHEMA) as cursor:
        for i in tqdm(range(0, len(params_to_query), DB_QUERY_BATCH_SIZE), desc="Querying DB"):
            batch_params = params_to_query[i:i + DB_QUERY_BATCH_SIZE]
            if not batch_params:
                continue

            tuple_placeholders = ', '.join(['(%s, %s)'] * len(batch_params))
            flat_params = [item for sublist in batch_params for item in sublist]

            query = f"""
                SELECT 
                    tff.video_id, v.absolute_path,
                    tff.frame_nr, tff.tracking_id,
                    tff.bbox_x_center_n, tff.bbox_y_center_n,
                    tff.bbox_width_n, tff.bbox_height_n
                FROM tracking_frame_feature tff
                JOIN video v ON tff.video_id = v.video_id
                WHERE (tff.frame_nr, tff.tracking_id) IN ({tuple_placeholders})
                AND tff.feature_type = %s
            """

            cursor.execute(query, flat_params + [FEATURE_TYPE])
            
            results = cursor.fetchall()
            for row in results:
                all_bbox_data.append(dict(zip([
                    'video_id', 'video_path', 'frame_nr', 'tracking_id',
                    'x', 'y', 'w', 'h'
                ], row)))
    
    bbox_df = pd.DataFrame(all_bbox_data)
    # Merge back with original data to get the 'split' column
    merged_df = pd.merge(file_df, bbox_df, on=['frame_nr', 'tracking_id'])
    
    # Fix video paths 
    merged_df['video_path'] = merged_df['video_path'].str.replace("gorillatracker/video_data", "vast-gorilla", regex=False)
    
    return merged_df

def extract_frames_batch(video_path, frame_numbers):
    """Efficiently extracts a batch of frames from a single video file."""
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}. Skipping frames.")
        return {}
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        frames = vr.get_batch(frame_numbers).asnumpy()
        return {num: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for num, frame in zip(frame_numbers, frames)}
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return {}

def process_batch(predictor, batch_df):
    """Processes a DataFrame batch: extracts frames, runs segmentation, and returns results."""
    # Group by video to minimize file opening/reading
    grouped_by_video = batch_df.groupby('video_path')
    
    img_batch, boxes_batch, metadata_batch = [], [], []

    for video_path, group in grouped_by_video:
        frame_numbers = sorted(group['frame_nr'].unique())
        extracted_frames = extract_frames_batch(video_path, frame_numbers)

        for _, row in group.iterrows():
            frame_nr = row['frame_nr']
            if frame_nr not in extracted_frames:
                continue

            frame = extracted_frames[frame_nr]
            h_img, w_img, _ = frame.shape

            # Convert normalized YOLO format [x_center, y_center, w, h] to pixel coordinates [x0, y0, x1, y1]
            x0 = (row['x'] - row['w'] / 2) * w_img
            y0 = (row['y'] - row['h'] / 2) * h_img
            x1 = (row['x'] + row['w'] / 2) * w_img
            y1 = (row['y'] + row['h'] / 2) * h_img
            
            img_batch.append(frame)
            boxes_batch.append(np.array([x0, y0, x1, y1]))
            # This is the key: keep metadata aligned with each item in the batch
            metadata_batch.append(row.to_dict())

    if not img_batch:
        return None

    predictor.set_image_batch(img_batch)
    masks, scores, _ = predictor.predict_batch(None, None, box_batch=boxes_batch, multimask_output=False)
    
    return {"masks": masks, "scores": scores, "metadata": metadata_batch, "images": img_batch, "boxes": boxes_batch}

def save_batch_results(results, resized_masks_list=None, dino_input_size=(518, 518)): #TODO: read this dino_input size from somewhere else
    """Saves masks and overlays from a processed batch to disk using a thread pool."""
    if not results:
        return

    crop_tasks, overlay_tasks = [], []
    
    for i in range(len(results["masks"])):
        meta = results["metadata"][i]
        
        # This correctly handles multiple gorillas per frame because each has unique metadata
        file_name = f"{meta['video_id']}_{meta['frame_nr']}_{meta['tracking_id']}.png"
        
        crop_path = Path(OUTPUT_PATH) / meta['split'] / 'crops' / file_name
        overlay_path = Path(OUTPUT_PATH) / meta['split'] / 'overlays' / file_name

        crop_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        
        mask = results["masks"][i].squeeze(0)
        box = results["boxes"][i]
        image = results["images"][i]
        
        if resized_masks_list is not None:
            resized_mask = crop_and_resize_mask(mask, box, target_size=dino_input_size)
            resized_masks_list.append(resized_mask)
        
        crop_tasks.append((image, mask, str(crop_path)))
        overlay_tasks.append((image, mask, box, str(overlay_path)))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(lambda p: crop_masked_region(*p), crop_tasks), total=len(crop_tasks), desc="Saving crops", leave=False))
        list(tqdm(executor.map(lambda p: overlay_mask_on_image(*p), overlay_tasks), total=len(overlay_tasks), desc="Saving overlays", leave=False))


def segment_images(path_to_images=None):
    """Main function to run the entire segmentation pipeline."""
    start_time = time.time()
    
    # Define DINOv2 input size here for clarity
    DINOV2_INPUT_SIZE = (518, 518)

    if path_to_images:
        INPUT_IMAGE_DIR = path_to_images
    else:
        INPUT_IMAGE_DIR = "/workspaces/bachelor_thesis_code/seg_test_in"

    predictor, _ = setup_model_and_device()
    
    file_info_df = gather_file_info(INPUT_IMAGE_DIR)
    if file_info_df.empty:
        print("No image files found to process. Exiting.")
        return [], [], [] # Return empty lists
        
    data_to_process_df = fetch_bounding_boxes(file_info_df)
    if data_to_process_df.empty:
        print("No corresponding bounding boxes found in the database. Exiting.")
        return [], [], [] # Return empty lists

    print(f"\nFound {len(data_to_process_df)} detections to process. Starting batch processing...")

    all_full_res_masks = []
    all_original_images = []
    all_resized_masks = []

    total_batches = (len(data_to_process_df) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(0, len(data_to_process_df), BATCH_SIZE), total=total_batches, desc="Overall Progress"):
        batch_df = data_to_process_df.iloc[i : i + BATCH_SIZE]
        
        inference_start = time.time()
        batch_results = process_batch(predictor, batch_df)
        inference_time = time.time() - inference_start
        
        # Accumulate results if the batch was successful
        if batch_results:
            all_full_res_masks.extend(batch_results["masks"])
            all_original_images.extend(batch_results["images"])
            
            saving_start = time.time()
            # Pass the list to be filled with resized masks
            save_batch_results(
                batch_results, 
                resized_masks_list=all_resized_masks, 
                dino_input_size=DINOV2_INPUT_SIZE
            )
            saving_time = time.time() - saving_start
            tqdm.write(f"Batch {i//BATCH_SIZE + 1}/{total_batches} | Inference: {inference_time:.2f}s, Saving: {saving_time:.2f}s")
        
    total_time = time.time() - start_time
    print("\n--- Processing Complete ---")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Processed {len(data_to_process_df)} detections.")
    print(f"Results saved to: {OUTPUT_PATH}")
    
    # --- MODIFICATION: Return the accumulated lists ---
    return all_full_res_masks, all_original_images, all_resized_masks


if __name__ == "__main__":
    segment_images()