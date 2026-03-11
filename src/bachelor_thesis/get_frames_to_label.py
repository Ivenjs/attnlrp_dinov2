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
from typing import List, Tuple, Set, Dict
from utils import load_config 
from db_connect import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FEATURE_TYPE = "body"
DB_SCHEMA = "public"
VIDEO_PATH_REPLACE_TUPLE = ("gorillatracker/video_data", "vast-gorilla")


def _choose_next_index_maxmin(frames: List[Tuple[int, Path]], picked_idx: Set[int]) -> int | None:
    """
    Selects the index of the next frame from 'frames' such that the 
    minimum distance (in terms of frame_nr) to already selected frames is maximized.

    frames: [(frame_nr, path)] – must be sorted by frame_nr.
    picked_idx: indices that have already been selected.

    Returns: Index within 'frames' or None if no frames are available.
    """
    n = len(frames)
    remaining = [i for i in range(n) if i not in picked_idx]
    if not remaining:
        return None
    if not picked_idx:
        return n // 2 

    picked_frames = [frames[i][0] for i in picked_idx]
    best_i, best_dist = None, -1
    for i in remaining:
        f = frames[i][0]
        d = min(abs(f - pf) for pf in picked_frames)
        if d > best_dist:
            best_dist = d
            best_i = i
    return best_i


def get_sampled_images_per_class(base_dir: Path, num_images_per_class: int = 1) -> Dict[str, List[Path]]:
    """
    Scans specified subdirectories (train, validation, test) and samples images for each class.

    Ziele (in dieser Reihenfolge):
    1) Priorisiere Diversität über verschiedene Videos (Round-Robin über Videos).
    2) Innerhalb eines Videos picke Frames mit maximaler Distanz (greedy max-min nach frame_nr).

    Erwartetes Filename-Schema:
        <class_label>_..._<video_id>_<frame_nr>_<...>.png
    wobei:
        class_label = parts[0]
        video_id    = int(parts[-3])
        frame_nr    = int(parts[-2])
    """
    if num_images_per_class <= 0:
        raise ValueError("num_images_per_class must be >= 1")

    splits_to_scan = ["train", "validation", "test"]

    all_class_video_frames: Dict[str, Dict[int, List[Tuple[int, Path]]]] = defaultdict(lambda: defaultdict(list))

    for split in splits_to_scan:
        split_dir = base_dir / split
        if not split_dir.is_dir():
            logging.warning(f"Split directory not found, skipping: {split_dir}")
            continue

        logging.info(f"Scanning for images in {split_dir}...")
        for filepath in split_dir.rglob("*.png"):
            try:
                parts = filepath.stem.split("_")
                class_label = parts[0]
                video_id = int(parts[-3])
                frame_nr = int(parts[-2])
                all_class_video_frames[class_label][video_id].append((frame_nr, filepath))
            except (IndexError, ValueError):
                logging.warning(f"Could not parse class/video/frame from filename: {filepath.name}")

    class_images: Dict[str, List[Path]] = defaultdict(list)

    for class_label, videos in all_class_video_frames.items():
        for vid in videos:
            videos[vid].sort(key=lambda x: x[0])

        total_available = sum(len(frames) for frames in videos.values())
        if total_available < num_images_per_class:
            logging.warning(
                f"Class '{class_label}' has only {total_available} images, "
                f"requested {num_images_per_class}."
            )
            class_images[class_label].extend([p for frames in videos.values() for (_, p) in frames])
            continue

        videos_picked_idx: Dict[int, Set[int]] = {vid: set() for vid in videos}
        picked_count = 0

        video_order = sorted(videos.keys())

        while picked_count < num_images_per_class:
            progress = False
            for vid in video_order:
                if picked_count >= num_images_per_class:
                    break
                frames = videos[vid]
                idx = _choose_next_index_maxmin(frames, videos_picked_idx[vid])
                if idx is None:
                    continue  
                videos_picked_idx[vid].add(idx)
                class_images[class_label].append(frames[idx][1])
                picked_count += 1
                progress = True
            if not progress:
                break

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
    if file_df.empty:
        return file_df

    logging.info(f"Querying database for video paths of {len(file_df)} images...")

    data_to_query = [(int(row.frame_nr), int(row.tracking_id), feature_type) for row in file_df.itertuples()]
    all_results = []

    try:
        with get_db_connection(schema=db_schema) as cursor:
            query = """
                WITH v(frame_nr, tracking_id, feature_type) AS (VALUES %s)
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

            BATCH = 1000
            for i in range(0, len(data_to_query), BATCH):
                chunk = data_to_query[i:i+BATCH]
                execute_values(cursor, query, chunk, page_size=len(chunk))
                all_results.extend(cursor.fetchall())

        if not all_results:
            logging.warning("No matching video paths found in the database.")
            return pd.DataFrame(columns=file_df.columns.tolist() + ["video_path"])

        db_df = pd.DataFrame(all_results, columns=["frame_nr", "tracking_id", "video_path"])
        enriched_df = pd.merge(file_df, db_df, on=["frame_nr", "tracking_id"], how="left")

        if VIDEO_PATH_REPLACE_TUPLE:
            enriched_df["video_path"] = enriched_df["video_path"].str.replace(
                VIDEO_PATH_REPLACE_TUPLE[0], VIDEO_PATH_REPLACE_TUPLE[1], regex=False
            )

        matched = enriched_df["video_path"].notna().sum()
        logging.info(f"DB match rate: {matched}/{len(enriched_df)} ({matched/len(enriched_df):.1%})")

        return enriched_df

    except Exception as e:
        logging.critical(f"Database query failed: {e}")
        return pd.DataFrame(columns=file_df.columns.tolist() + ["video_path"])


def extract_and_save_whole_frames(df_with_paths: pd.DataFrame, output_dir: Path):
    """Iterates through videos, extracts frames in batches, and saves them."""
    if 'video_path' not in df_with_paths.columns or df_with_paths['video_path'].isnull().all():
        logging.error("No video paths found in the DataFrame. Cannot extract frames.")
        return

    for video_path, group in tqdm(df_with_paths.groupby("video_path"), desc="Extracting full frames"):
        if pd.isna(video_path):
            logging.warning(f"Skipping {len(group)} images with no associated video path.")
            continue

        if os.path.exists(video_path):
            logging.info(f"video path {video_path} exists")
        else:
            logging.info(f"video path {video_path} does NOT exist")

        frame_requests = group[["frame_nr", "source_path"]].to_dict('records')
        frame_numbers = [req["frame_nr"] for req in frame_requests]
        
        logging.info(f"Extracting {len(frame_numbers)} frames from {video_path}")
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            extracted_frames = vr.get_batch(frame_numbers).asnumpy()
            frame_map = {num: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for num, frame in zip(frame_numbers, extracted_frames)}
        except Exception as e:
            logging.error(f"Error reading video {video_path}: {e}. Skipping this video.")
            continue

        for req in frame_requests:
            frame_num = req["frame_nr"]
            if frame_num in frame_map:
                output_name = req["source_path"].name
                save_path = output_dir / output_name
                cv2.imwrite(str(save_path), frame_map[frame_num])
            else:
                logging.warning(f"Failed to extract frame {frame_num} from {video_path}")


def main():
    """Main script execution."""
    cfg = load_config("finetuned", [])
    
    dataset_dir = Path(cfg["data"]['dataset_dir'])
    save_dir = Path("/workspaces/vast-gorilla/gorillawatch/iven_thesis/frames_to_label")
    num_images_per_class = 7

    cropped_images_dir = save_dir / "sampled_cropped_images"
    whole_images_dir = save_dir / "corresponding_whole_images"
    cropped_images_dir.mkdir(parents=True, exist_ok=True)
    whole_images_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Scanning for images in {dataset_dir}...")
    class_images_map = get_sampled_images_per_class(dataset_dir, num_images_per_class)
    
    all_sampled_paths = [path for paths in class_images_map.values() for path in paths]

    if not all_sampled_paths:
        logging.info("No images were sampled. Exiting.")
        return

    logging.info(f"Copying {len(all_sampled_paths)} sampled cropped images to {cropped_images_dir}...")
    for img_path in tqdm(all_sampled_paths, desc="Copying cropped images"):
        shutil.copy(img_path, cropped_images_dir / img_path.name)
    
    file_info_df = parse_image_info_to_df(all_sampled_paths)

    df_with_paths = fetch_video_paths_for_df(file_info_df, DB_SCHEMA, FEATURE_TYPE)

    extract_and_save_whole_frames(df_with_paths, whole_images_dir)

    logging.info("Processing complete.")
    logging.info(f"Sampled cropped images are in: {cropped_images_dir}")
    logging.info(f"Corresponding full frames are in: {whole_images_dir}")


if __name__ == "__main__":
    main()
