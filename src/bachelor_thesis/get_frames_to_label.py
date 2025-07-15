import os
import shutil
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml
from decord import VideoReader, cpu
from psycopg2.extras import execute_values

# Your corrected db_connect file
from db_connect import get_db_connection

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
FEATURE_TYPE = "body"
DB_SCHEMA = "public" # This can be passed to get_db_connection
VIDEO_PATH_REPLACE_TUPLE = ("gorillatracker/video_data", "vast-gorilla")

# --- Dataclass for clear data structure ---
@dataclass
class ImageMetadata:
    source_path: Path
    output_name: str
    frame_nr: int
    tracking_id: int

# --- Helper Functions (mostly unchanged, but using Path and logging) ---

def get_cropped_images_for_each_class(image_dir: Path, num_images_per_class: int = 1) -> dict[str, list[Path]]:
    class_images = defaultdict(list)
    for filepath in image_dir.rglob("*.png"):
        try:
            class_label = filepath.name.split("_")[0]
            if len(class_images[class_label]) < num_images_per_class:
                class_images[class_label].append(filepath)
        except IndexError:
            logging.warning(f"Could not parse class from filename: {filepath.name}")
    return dict(class_images)

def extract_frames_batch(video_path: str, frame_numbers: list[int]) -> dict[int, object]:
    if not os.path.exists(video_path):
        logging.warning(f"Video file not found: {video_path}. Skipping frames.")
        return {}
    try:
        with VideoReader(video_path, ctx=cpu(0)) as vr:
            frames = vr.get_batch(frame_numbers).asnumpy()
            return {num: frame for num, frame in zip(frame_numbers, frames)}
    except Exception as e:
        logging.error(f"Error reading video {video_path}: {e}")
        return {}

def _parse_image_info(img_path: Path) -> ImageMetadata | None:
    try:
        parts = img_path.stem.split("_")
        return ImageMetadata(
            source_path=img_path,
            output_name=img_path.name,
            frame_nr=int(parts[-2]),
            tracking_id=int(parts[-1]),
        )
    except (IndexError, ValueError) as e:
        logging.warning(f"Could not parse metadata from filename '{img_path.name}': {e}")
        return None

# --- BATCH QUERY FUNCTION (MODIFIED FOR PSYCOPG2) ---

def _fetch_video_paths_in_batch(cursor, all_metadata: list[ImageMetadata]) -> dict[tuple[int, int], str]:
    """
    Fetches video paths for a list of identifiers using psycopg2's execute_values.
    """
    if not all_metadata:
        return {}

    # Create a list of (frame_nr, tracking_id) tuples for the query
    identifiers = [(meta.frame_nr, meta.tracking_id) for meta in all_metadata]
    
    # This query joins the main table against a temporary table of values
    # created by execute_values. This is the standard, efficient pattern for psycopg2.
    query = f"""
        SELECT 
            v.frame_nr,
            v.tracking_id,
            t.absolute_path
        FROM tracking_frame_feature AS tff
        JOIN video AS t ON tff.video_id = t.video_id
        JOIN (VALUES %s) AS v(frame_nr, tracking_id) 
            ON tff.frame_nr = v.frame_nr AND tff.tracking_id = v.tracking_id
        WHERE tff.feature_type = %s
    """
    
    logging.info(f"Executing batch query for {len(identifiers)} identifiers...")
    # execute_values(cursor, sql_template, data_tuples, template_for_data)
    execute_values(cursor, query, identifiers, template=None, page_size=100)
    
    results = {}
    for frame_nr, tracking_id, video_path in cursor.fetchall():
        if VIDEO_PATH_REPLACE_TUPLE:
            video_path = video_path.replace(VIDEO_PATH_REPLACE_TUPLE[0], VIDEO_PATH_REPLACE_TUPLE[1])
        results[(frame_nr, tracking_id)] = video_path
        
    logging.info(f"Found video paths for {len(results)} identifiers.")
    return results

def _extract_and_save_whole_frames(videos_to_process: dict[str, list[tuple[int, str]]], whole_images_dir: Path):
    """Iterates through videos, extracts frames in batches, and saves them."""
    logging.info(f"Processing {len(videos_to_process)} unique videos...")
    for video_path, frame_requests in videos_to_process.items():
        frame_numbers = [req[0] for req in frame_requests]
        output_names = {req[0]: req[1] for req in frame_requests}
        
        logging.info(f"Extracting {len(frame_numbers)} frames from {video_path}...")
        extracted_frames = extract_frames_batch(video_path, frame_numbers)
        
        for frame_num, frame_data in extracted_frames.items():
            if frame_data is not None:
                output_name = output_names[frame_num]
                frame_save_path = whole_images_dir / f"{Path(output_name).stem}.png"
                cv2.imwrite(str(frame_save_path), frame_data)
            else:
                logging.warning(f"Failed to extract frame {frame_num} from {video_path}")

# --- Main Orchestrator ---

def main():
    """Main script execution."""
    config_path = Path("/workspaces/bachelor_thesis_code/src/bachelor_thesis/configs/data.yaml")
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    
    dataset_dir = Path(cfg['dataset_dir'])
    save_dir = Path("/workspaces/vast-gorilla/gorillawatch/iven_thesis/frames_to_label")
    num_images_per_class = 1

    cropped_images_dir = save_dir / "cropped_images"
    whole_images_dir = save_dir / "whole_images"
    cropped_images_dir.mkdir(parents=True, exist_ok=True)
    whole_images_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Scanning for images in {dataset_dir}...")
    class_images = get_cropped_images_for_each_class(dataset_dir, num_images_per_class)
    
    all_metadata = []
    for images in class_images.values():
        for img_path in images:
            shutil.copy(img_path, cropped_images_dir / img_path.name)
            metadata = _parse_image_info(img_path)
            if metadata:
                all_metadata.append(metadata)

    if not all_metadata:
        logging.info("No valid images found to process.")
        return

    video_path_map = {}
    try:
        # The 'with' statement now correctly manages the connection and cursor!
        with get_db_connection(schema=DB_SCHEMA) as cursor:
            video_path_map = _fetch_video_paths_in_batch(cursor, all_metadata)
    except Exception as e:
        # The error from the DB will be caught here
        logging.critical(f"Could not complete database operations. Aborting. Error: {e}")
        return

    videos_to_process = defaultdict(list)
    for meta in all_metadata:
        key = (meta.frame_nr, meta.tracking_id)
        if key in video_path_map:
            video_path = video_path_map[key]
            videos_to_process[video_path].append((meta.frame_nr, meta.output_name))
        else:
            logging.warning(f"No DB entry found for {meta.output_name} (frame: {meta.frame_nr}, track: {meta.tracking_id})")

    _extract_and_save_whole_frames(videos_to_process, whole_images_dir)

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()