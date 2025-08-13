import os
import shutil
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
from decord import VideoReader, cpu
from psycopg2.extras import execute_values
from tqdm import tqdm

# Assuming these utilities are in your project
from utils import load_config 
from db_connect import get_db_connection

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
FEATURE_TYPE = "body"
DB_SCHEMA = "public"
VIDEO_PATH_REPLACE_TUPLE = ("gorillatracker/video_data", "vast-gorilla")

# --- Helper Functions ---

def get_sampled_images_per_class(base_dir: Path, num_images_per_class: int = 1) -> dict[str, list[Path]]:
    """
    Scans specified subdirectories (train, val, test) and samples images for each class.
    """
    class_images = defaultdict(list)
    splits_to_scan = ["train", "val", "test"] # Explicitly define the subdirectories

    for split in splits_to_scan:
        split_dir = base_dir / split
        if not split_dir.is_dir():
            logging.warning(f"Split directory not found, skipping: {split_dir}")
            continue
        
        logging.info(f"Scanning for images in {split_dir}...")
        # Use rglob on the split_dir in case there are nested folders within it.
        # If images are guaranteed to be at the top level, you can use .glob("*.png")
        for filepath in split_dir.rglob("*.png"):
            try:
                class_label = filepath.name.split("_")[0]
                if len(class_images[class_label]) < num_images_per_class:
                    class_images[class_label].append(filepath)
            except IndexError:
                logging.warning(f"Could not parse class from filename: {filepath.name}")
    
    found_count = sum(len(paths) for paths in class_images.values())
    logging.info(f"Sampled {found_count} images across {len(class_images)} classes from all splits.")
    return dict(class_images)

def parse_image_info_to_df(image_paths: list[Path]) -> pd.DataFrame:
    """Parses frame_nr and tracking_id from a list of Path objects into a DataFrame."""
    records = []
    for img_path in image_paths:
        try:
            parts = img_path.stem.split("_")
            records.append({
                "source_path": img_path,
                "frame_nr": int(parts[-2]),
                "tracking_id": int(parts[-1]),
            })
        except (ValueError, IndexError):
            logging.warning(f"Could not parse metadata from filename '{img_path.name}'. Skipping.")
    return pd.DataFrame(records)

def fetch_video_paths_for_df(file_df: pd.DataFrame, db_schema: str, feature_type: str) -> pd.DataFrame:
    """Enriches a DataFrame with video paths from the database using a robust batch query."""
    if file_df.empty:
        return file_df

    logging.info(f"Querying database for video paths of {len(file_df)} images...")
    
    data_to_query = [(row.frame_nr, row.tracking_id, feature_type) for row in file_df.itertuples()]

    try:
        with get_db_connection(schema=db_schema) as cursor:
            query = """
                WITH v(frame_nr, tracking_id, feature_type) AS (
                    VALUES %s
                )
                SELECT
                    v.frame_nr,
                    v.tracking_id,
                    t.absolute_path AS video_path
                FROM tracking_frame_feature AS tff
                JOIN video AS t ON tff.video_id = t.video_id
                JOIN v ON tff.frame_nr = v.frame_nr
                     AND tff.tracking_id = v.tracking_id
                     AND tff.feature_type = v.feature_type
            """
            execute_values(cursor, query, data_to_query, page_size=500)
            results = cursor.fetchall()

            if not results:
                logging.warning("No matching video paths found in the database.")
                return pd.DataFrame(columns=file_df.columns.tolist() + ["video_path"])

            db_df = pd.DataFrame(results, columns=["frame_nr", "tracking_id", "video_path"])
            enriched_df = pd.merge(file_df, db_df, on=["frame_nr", "tracking_id"], how="left")

            if VIDEO_PATH_REPLACE_TUPLE:
                enriched_df["video_path"] = enriched_df["video_path"].str.replace(
                    VIDEO_PATH_REPLACE_TUPLE[0], VIDEO_PATH_REPLACE_TUPLE[1], regex=False
                )
            return enriched_df

    except Exception as e:
        logging.critical(f"Database query failed: {e}")
        return pd.DataFrame(columns=file_df.columns.tolist() + ["video_path"])


def extract_and_save_whole_frames(df_with_paths: pd.DataFrame, output_dir: Path):
    """Iterates through videos, extracts frames in batches, and saves them."""
    if 'video_path' not in df_with_paths.columns or df_with_paths['video_path'].isnull().all():
        logging.error("No video paths found in the DataFrame. Cannot extract frames.")
        return

    # Group by video to process each video only once
    for video_path, group in tqdm(df_with_paths.groupby("video_path"), desc="Extracting full frames"):
        if pd.isna(video_path):
            logging.warning(f"Skipping {len(group)} images with no associated video path.")
            continue

        frame_requests = group[["frame_nr", "source_path"]].to_dict('records')
        frame_numbers = [req["frame_nr"] for req in frame_requests]
        
        logging.info(f"Extracting {len(frame_numbers)} frames from {video_path}")
        
        # Extract frames in one batch
        try:
            with VideoReader(video_path, ctx=cpu(0)) as vr:
                extracted_frames = vr.get_batch(frame_numbers).asnumpy()
                frame_map = {num: frame for num, frame in zip(frame_numbers, extracted_frames)}
        except Exception as e:
            logging.error(f"Error reading video {video_path}: {e}. Skipping this video.")
            continue

        # Save the extracted frames
        for req in frame_requests:
            frame_num = req["frame_nr"]
            if frame_num in frame_map:
                output_name = req["source_path"].name
                save_path = output_dir / output_name
                cv2.imwrite(str(save_path), frame_map[frame_num])
            else:
                logging.warning(f"Failed to extract frame {frame_num} from {video_path}")

# --- Main Orchestrator ---

def main():
    """Main script execution."""
    # Use your config loader
    cfg = load_config("finetuned", [])
    
    # --- Configuration ---
    # Use Path objects for modern, OS-agnostic path handling
    dataset_dir = Path(cfg["data"]['dataset_dir'])
    save_dir = Path("/workspaces/vast-gorilla/gorillawatch/iven_thesis/frames_to_label")
    num_images_per_class = 3

    cropped_images_dir = save_dir / "sampled_cropped_images"
    whole_images_dir = save_dir / "corresponding_whole_images"
    cropped_images_dir.mkdir(parents=True, exist_ok=True)
    whole_images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample N cropped images for each class from the dataset directory
    logging.info(f"Scanning for images in {dataset_dir}...")
    class_images_map = get_sampled_images_per_class(dataset_dir, num_images_per_class)
    
    # Flatten the list of all sampled image paths
    all_sampled_paths = [path for paths in class_images_map.values() for path in paths]

    if not all_sampled_paths:
        logging.info("No images were sampled. Exiting.")
        return

    # 2. Copy the sampled cropped images to the output directory
    logging.info(f"Copying {len(all_sampled_paths)} sampled cropped images to {cropped_images_dir}...")
    for img_path in tqdm(all_sampled_paths, desc="Copying cropped images"):
        shutil.copy(img_path, cropped_images_dir / img_path.name)
    
    # 3. Parse filenames to get frame_nr and tracking_id
    file_info_df = parse_image_info_to_df(all_sampled_paths)

    # 4. Fetch the corresponding video paths from the database
    df_with_paths = fetch_video_paths_for_df(file_info_df, DB_SCHEMA, FEATURE_TYPE)

    # 5. Extract and save the full frames
    extract_and_save_whole_frames(df_with_paths, whole_images_dir)

    logging.info("Processing complete.")
    logging.info(f"Sampled cropped images are in: {cropped_images_dir}")
    logging.info(f"Corresponding full frames are in: {whole_images_dir}")


if __name__ == "__main__":
    main()