from collections import defaultdict
from dataset import GorillaReIDDataset, custom_collate_fn
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from basemodel import get_model_wrapper
import torch

from utils import load_config
if __name__ == "__main__":

    cfg = load_config("finetuned", [""])
    root_dir = cfg["data"]["dataset_dir"]
    train_dir = os.path.join(root_dir, "train")
    
    DEVICE = torch.device("cpu")
    train_files = [f for f in os.listdir(train_dir) if f.lower().endswith((".jpg", ".png"))]
    _, image_transforms, _ = get_model_wrapper(device=DEVICE, cfg=cfg["model"])


    train_dataset = GorillaReIDDataset(
        image_dir=train_dir, filenames=train_files, transform=image_transforms
    )

    dataloader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    print(f"Generating embeddings for {len(dataloader.dataset)} images...")
        
    all_embeddings_list = []
    all_labels = []
    all_filenames = []
    all_videos = []

    for batch in tqdm(dataloader, desc="Extracting data"):
        images = batch["image"]
        labels = batch["label"]
        filenames = batch["filename"]
        videos = batch["video"]
        
        all_labels.extend(labels)
        all_filenames.extend(filenames)
        all_videos.extend(videos)

    #video from filename = f.split('_')[1] + "_" + f.split('_')[2] + f.split('_')[3]
    #check for each label how many distict videos there are
    label_video_count = defaultdict(set)
    for label, video in zip(all_labels, all_videos):
        label_video_count[label].add(video)

    #sort ascending
    for label, videos in sorted(label_video_count.items(), key=lambda x: len(x[1])):
        print(f"Label {label} has {len(videos)} distinct videos.")
