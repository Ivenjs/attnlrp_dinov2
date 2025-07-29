
import os
import logging
import pandas as pd
from pathlib import Path
from decord import VideoReader, cpu
import cv2
import numpy as np
from db_connect import get_db_connection
from psycopg2.extras import execute_values

def gather_file_info(filenames: list[str]) -> pd.DataFrame:
    """Parses frame_nr and tracking_id from a list of filenames."""
    records = []
    for filename in filenames:
        try:
            # Assuming filename format is like '..._{frame_nr}_{tracking_id}.png'
            base_name = Path(filename).stem
            parts = base_name.split("_")
            frame_nr = int(parts[-2])
            tracking_id = int(parts[-1])
            records.append({
                "original_filename": filename,
                "frame_nr": frame_nr,
                "tracking_id": tracking_id
            })
        except (ValueError, IndexError):
            logging.warning(f"Could not parse filename '{filename}'. Skipping.")

    return pd.DataFrame(records)


def fetch_bounding_boxes(file_df: pd.DataFrame, db_schema: str, feature_type: str = "body") -> pd.DataFrame:
    """
    Fetches bounding box data from the database for the detections in the DataFrame.

    Args:
        file_df (pd.DataFrame): DataFrame with 'frame_nr' and 'tracking_id' columns.
        db_schema (str): The database schema to use.
        feature_type (str): The feature type for the bounding box (e.g., 'body').

    Returns:
        pd.DataFrame: Merged DataFrame with video path and bbox coordinates.
    """
    if file_df.empty:
        return pd.DataFrame()

    logging.info(f"Fetching bounding box data for {len(file_df)} detections...")

    params_to_query = list(zip(file_df["frame_nr"], file_df["tracking_id"]))

    try:
        # NOTE: Using a 'with' statement for the cursor assumes your get_db_connection handles
        # connection and cursor closing properly.
        with get_db_connection(schema=db_schema) as cursor:
            query = f"""
                SELECT
                    v.frame_nr,
                    v.tracking_id,
                    t.absolute_path AS video_path,
                    tff.bbox_x_center_n AS x,
                    tff.bbox_y_center_n AS y,
                    tff.bbox_width_n AS w,
                    tff.bbox_height_n AS h
                FROM tracking_frame_feature AS tff
                JOIN video AS t ON tff.video_id = t.video_id
                JOIN (VALUES %s) AS v(frame_nr, tracking_id)
                    ON tff.frame_nr = v.frame_nr AND tff.tracking_id = v.tracking_id
                WHERE tff.feature_type = '{feature_type}'
            """
            results = execute_values(
                cursor,
                query,
                params_to_query,
                template='(%s, %s)', # This is the crucial addition
                page_size=500,
                fetch=True
            )

            if not results:
                logging.warning("No matching bounding boxes found in the database.")
                return pd.DataFrame()

            columns = ["frame_nr", "tracking_id", "video_path", "x", "y", "w", "h"]
            bbox_df = pd.DataFrame(results, columns=columns)
            
            # Merge with original filenames
            merged_df = pd.merge(file_df, bbox_df, on=["frame_nr", "tracking_id"], how="inner")

            merged_df["video_path"] = merged_df["video_path"].str.replace(
                "gorillatracker/video_data", "vast-gorilla", regex=False
            )
            
            return merged_df

    except Exception as e:
        logging.error(f"Failed to fetch bounding boxes from database: {e}")
        return pd.DataFrame(columns=file_df.columns.tolist() + ["video_path", "x", "y", "w", "h"])


def extract_frames_batch(video_path: str, frame_numbers: list[int]) -> dict:
    """Efficiently extracts a batch of frames from a single video file."""
    if not os.path.exists(video_path):
        logging.warning(f"Video file not found: {video_path}. Skipping frames.")
        return {}
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        # Ensure frame numbers are sorted for efficient access
        unique_frames = sorted(list(set(frame_numbers)))
        frames = vr.get_batch(unique_frames).asnumpy()
        # Return frames in RGB format
        return {num: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for num, frame in zip(unique_frames, frames)}
    except Exception as e:
        logging.error(f"Error reading video {video_path}: {e}")
        return {}