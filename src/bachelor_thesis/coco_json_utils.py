import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

def _normalize_name(name: str) -> str:
    """Standardizes gorilla names for matching."""
    return name.lower().replace("'", "").replace("-", "")

def load_and_preprocess_coco_json(json_path: str) -> Dict[tuple, Any]:
    """
    Loads a COCO-style JSON and builds a robust lookup map by iterating through 
    annotations first, ensuring no body data is lost.

    The key is a tuple: (video_number, frame_number, normalized_gorilla_name)
    
    Args:
        json_path (str): Path to the dataset.json file.

    Returns:
        Dict[tuple, Any]: A dictionary mapping the composite key to the full
                          image and 'body' annotation data dictionary.
    """
    print(f"Loading and preprocessing COCO JSON from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    category_id_to_norm_name = {
        cat['id']: _normalize_name(cat['name']) for cat in data['categories']
    }

    image_id_to_info = {img['id']: img for img in data['images']}
    
    composite_key_to_data = {}
    
    body_annotation_count = 0
    for ann in data['annotations']:
        if ann.get('annotation_type') != 'body':
            continue

        body_annotation_count += 1
        image_id = ann.get('image_id')
        category_id = ann.get('category_id')

        if image_id not in image_id_to_info:
            print(f"Warning: Body annotation with ID {ann['id']} points to a non-existent image_id {image_id}. Skipping.")
            continue
            
        if category_id not in category_id_to_norm_name:
            print(f"Warning: Annotation with ID {ann['id']} has an invalid category_id {category_id}. Skipping.")
            continue

        image_info = image_id_to_info[image_id]
        
        video_filename = image_info.get('video_filename', '')
        video_nr_match = re.search(r'\d+', video_filename)
        if not video_nr_match:
            continue
        video_nr = int(video_nr_match.group(0))

        frame_nr = image_info.get('frame_number')
        if frame_nr is None:
            continue

        norm_name = category_id_to_norm_name[category_id]
        
        key = (video_nr, frame_nr, norm_name)
        
        if key in composite_key_to_data:
            pass 
        else:
            combined_data = {**image_info, **ann}
            composite_key_to_data[key] = combined_data

    print(f"Iterated through {body_annotation_count} annotations of type 'body'.")
    print(f"Successfully created a lookup map with {len(composite_key_to_data)} unique BODY entries.")
    return composite_key_to_data


def fetch_bounding_boxes_from_json(
    filenames: List[str],
    composite_key_map: Dict[tuple, Any],
    dataset_root_dir: str,
    video_subdir: Optional[str] = None
) -> pd.DataFrame:
    """
    Gathers bounding box and video info by parsing disk filenames and looking them
    up in the pre-processed composite key map.
    (This function does not need changes as it consumes the already-filtered map).
    """
    records = []
    for fname in filenames:
        try:
            parts = Path(fname).stem.split('_')
            
            gorilla_name_raw = parts[0]
            video_nr = int(parts[2])
            frame_nr = int(parts[4])
            
            norm_name = _normalize_name(gorilla_name_raw)
            lookup_key = (video_nr, frame_nr, norm_name)
            
            if lookup_key in composite_key_map:
                data_record = composite_key_map[lookup_key]
                
                video_base_path = Path(dataset_root_dir)
                if video_subdir:
                    video_base_path = video_base_path / video_subdir
                video_full_path = video_base_path / data_record['video_filename']
                
                records.append({
                    'original_filename': fname,
                    'video_path': str(video_full_path),
                    'frame_nr': data_record['frame_number'],
                    'bbox': data_record['bbox'],
                })

        
        except (IndexError, ValueError):
            print(f"Warning: Could not parse filename '{fname}'. It does not match expected format. Skipping.")

    print(f"Fetched bounding box data for {len(records)} out of {len(filenames)} filenames.")
            
    return pd.DataFrame(records)