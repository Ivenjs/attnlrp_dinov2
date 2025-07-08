from basemodel import TimmWrapper
from torchvision import transforms
import os
from PIL import Image
import torch
from tqdm import tqdm
import torch.nn.functional as F
from typing import Tuple


import yaml

from basemodel import get_model_wrapper


#TODO: rather use the transforms from the model wrapper
TRANSFORM = transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

def fill_knn_db(
    image_dir: str, model: TimmWrapper, output_dir: str, model_checkpoint: str, transform: transforms.Compose = TRANSFORM
) -> Tuple[torch.Tensor, list]:
    device = model.device

    all_embeddings = []
    labels = []

    for file in tqdm(os.listdir(image_dir), desc="Loading images for KNN DB"):
        if not file.endswith(".png"):
            continue

        image_path = os.path.join(image_dir, file)
        image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(image)  # should return [1, D] or [D]
        
        # Ensure embedding is 1D
        embedding = embedding.squeeze()
        if embedding.dim() != 1:
            raise ValueError(f"Expected 1D embedding, got shape {embedding.shape}")

        all_embeddings.append(embedding.cpu())
        labels.append(file.split("_")[0])

    stacked = torch.stack(all_embeddings).to(device)

    dataset = {
        "embeddings": stacked.cpu(),  # Save on CPU
        "labels": labels
    }

    out_path = f"{output_dir}/{model_checkpoint.split('/')[-1].split('.')[0]}_{image_dir.split('/')[-1]}.pt"
    torch.save(dataset, out_path)

    return stacked, labels

def get_knn_db(knn_db_dir: str, image_dir: str, model_wrapper: TimmWrapper, transforms: transforms.Compose, device: torch.device) -> Tuple[torch.Tensor, list]:
    db_embeddings = []
    db_labels = []

    model_config_path = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/configs/model_config.yaml"
    with open(model_config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    checkpoint_name = os.path.basename(cfg["checkpoint_path"])
    checkpoint_base = os.path.splitext(checkpoint_name)[0]

    files_in_dir = os.listdir(knn_db_dir)
    matching_checkpoints = [f for f in files_in_dir if checkpoint_base in f]
    if matching_checkpoints:
        print(f"KNN database for checkpoint {checkpoint_base} already exists. Loading the KNN database...")
        dataset = torch.load(os.path.join(knn_db_dir, matching_checkpoints[0]))
        db_embeddings = dataset["embeddings"].to(device)
        db_labels = dataset["labels"]
    else:
        print(f"KNN database for checkpoint {checkpoint_base} does not exist. Filling the KNN database...")
        db_embeddings, db_labels = fill_knn_db(
            image_dir=image_dir,
            model=model_wrapper,
            output_dir=knn_db_dir,
            model_checkpoint=checkpoint_name,
            transform=transforms
        )
    return db_embeddings, db_labels


# TODO use the GPU enhanced versions from the model_evaluation.py (gorillawatch repo)
def calculate_distance(db_embeddings: torch.Tensor, test_embedding: torch.Tensor, device: torch.device) -> torch.Tensor:
    distance = torch.norm(db_embeddings - test_embedding, dim=1)
    return distance

def knn(
    embedding: torch.Tensor,
    db_embeddings: torch.Tensor,
    db_labels: list,
    device: torch.device,
    k: int = 5,
    metric: str = "euclidean",
) -> str:
    if metric == "euclidean":
        distances = calculate_distance(db_embeddings, embedding, device)
    else:
        normalized_db_embeddings = F.normalize(db_embeddings, p=2, dim=1)
        normalized_test_embedding = F.normalize(embedding, p=2, dim=0)
        cosine_similarity = torch.matmul(normalized_db_embeddings, normalized_test_embedding.transpose(0, 1)).squeeze()
        distances = 1 - cosine_similarity

    top_k_indices = torch.topk(distances, k, largest=False).indices
    top_k_labels = [db_labels[i] for i in top_k_indices]
    label = max(set(top_k_labels), key=top_k_labels.count)
    return label


def compute_knn_proxy_score(
    query_embedding: torch.Tensor,
    db_embeddings: torch.Tensor,
    db_labels: list,
    ground_truth_label: str,
    k: int = 5,
    metric: str = "cosine" # Cosine similarity is generally better for high-dim embeddings
) -> torch.Tensor:
    """
    Computes a differentiable proxy score for a k-NN classifier's decision.

    The score is defined as: S = mean(sim_friends) - mean(sim_foes)
    This creates a differentiable objective that LRP can backpropagate through.

    Args:
        query_embedding (torch.Tensor): The (1, D) embedding of the input image.
                                        This tensor MUST have requires_grad=True.
        db_embeddings (torch.Tensor): The (N, D) embeddings in the k-NN database.
        db_labels (list): A list of N labels corresponding to the db_embeddings.
        ground_truth_label (str): The correct label for the query image.
        k (int): The number of nearest neighbors to consider.
        metric (str): The similarity metric to use, 'cosine' or 'euclidean'.

    Returns:
        torch.Tensor: A single scalar tensor representing the proxy score,
                      with its computation graph attached to the query_embedding.
    """
    # We must use a detached version of the query for the distance calculation
    # to find the neighbors. This is because topk is not nicely differentiable
    # and we only need the *identities* of the neighbors, not their gradient path.

    #TODO: make sure that I exclude the input image in friends and foes!!!

    with torch.no_grad():
        if metric == "euclidean":
            similarities = -calculate_distance(db_embeddings, query_embedding.detach(), query_embedding.device)
        else: # Default to cosine
            norm_db = F.normalize(db_embeddings, p=2, dim=1)
            norm_query = F.normalize(query_embedding.detach(), p=2, dim=1)
            similarities = F.linear(norm_db, norm_query).squeeze()

        top_k_indices = torch.topk(similarities, k, largest=True).indices


    query_index = db_labels.index(ground_truth_label)
    friends_indices = []
    foes_indices = []
    for idx_tensor in top_k_indices:
        idx = idx_tensor.item()  
        if db_labels[idx] == ground_truth_label:
            friends_indices.append(idx)
        else:
            foes_indices.append(idx)

    # Calculate the differentiable score S using the original query_embedding ---
    # Now we re-calculate similarities for friends and foes, but this time
    # with the original `query_embedding` that tracks gradients.

    # Handle the "friends"
    if not friends_indices:
        # If no friends are found in top-k, the score is highly negative.
        # We can define this as a large penalty. Let's use the average similarity
        # to all foes as the negative score. If no foes either, score is 0.
        if not foes_indices: return torch.tensor(0.0, device=query_embedding.device)
        foe_embeddings = db_embeddings[foes_indices]
        sim_foes = F.cosine_similarity(query_embedding, foe_embeddings).mean()
        return -sim_foes

    friend_embeddings = db_embeddings[friends_indices]
    sim_friends = F.cosine_similarity(query_embedding, friend_embeddings).mean()

    # Handle the "foes"
    if not foes_indices:
        # If all neighbors are friends, the goal is just to maximize similarity to them.
        return sim_friends
        
    foe_embeddings = db_embeddings[foes_indices]
    sim_foes = F.cosine_similarity(query_embedding, foe_embeddings).mean()
    
    score = sim_friends - sim_foes
    return score

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config_path = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/configs/model_config.yaml"
    with open(model_config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    model_wrapper, transforms = get_model_wrapper()

    output_dir = "/workspaces/bachelor_thesis_code/knn_db"
    # Fill the KNN database
    db_embeddings, labels = fill_knn_db(
        image_dir="/workspaces/vast-gorilla/gorillawatch/data/eval_body_squared_cleaned_open_2024_bigval/train",
        model=model_wrapper,
        output_dir=output_dir,
        model_checkpoint=os.path.basename(cfg["checkpoint_path"]),
        transform=transforms
    )

    # Initialize the KNN database
    """db_embeddings, db_labels = load_knn_db(output_dir, device)

    # Example query embedding
    query_embedding = torch.randn(1, 2048, requires_grad=True).to(device)  # Example embedding

    # Compute KNN label
    label = knn(query_embedding, db_embeddings, db_labels, device)
    print(f"KNN predicted label: {label}")

    # Compute proxy score
    ground_truth_label = "cat"  # Example ground truth label
    proxy_score = compute_knn_proxy_score(query_embedding, db_embeddings, db_labels, ground_truth_label)
    print(f"Proxy score: {proxy_score.item()}")"""