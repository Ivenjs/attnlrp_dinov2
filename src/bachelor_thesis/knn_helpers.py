from basemodel import TimmWrapper
from torchvision import transforms
import os
from PIL import Image
import torch
from tqdm import tqdm
import torch.nn.functional as F


#TODO: rather use the transforms from the model wrapper
TRANSFORM = transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

def fill_knn_db(
    image_dir: str, model: TimmWrapper,  device: torch.device, output_dir: str, model_checkpoint: str, transform: transforms.Compose = TRANSFORM
):
    embeddings = torch.Tensor([]).to(device)
    labels = []
    for file in tqdm(os.listdir(image_dir), desc="Loading images for KNN DB"):
        if file.endswith(".png"):
            image_path = os.path.join(image_dir, file)
            image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(image)
            embeddings = torch.cat((embeddings, embedding), dim=0)
            labels.append(file.split("_")[0])
    
    dataset = {"embeddings": embeddings.cpu(), "labels": labels}
    torch.save(
        dataset,
        f"{output_dir}/{model_checkpoint.split('/')[-1].split('.')[0]}_{image_dir.split('/')[-1]}.pt",
    )
    
def init_knn_db(knn_db_path: str, device: torch.device) -> tuple[torch.Tensor, list]:
    dataset = torch.load(knn_db_path)
    return dataset["embeddings"].to(device), dataset["labels"]


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
    if not query_embedding.requires_grad:
        raise ValueError("query_embedding must have requires_grad=True to compute gradients.")
    # We must use a detached version of the query for the distance calculation
    # to find the neighbors. This is because topk is not nicely differentiable
    # and we only need the *identities* of the neighbors, not their gradient path.
    with torch.no_grad():
        if metric == "euclidean":
            similarities = -calculate_distance(db_embeddings, query_embedding.detach(), query_embedding.device)
        else: # Default to cosine
            norm_db = F.normalize(db_embeddings, p=2, dim=1)
            norm_query = F.normalize(query_embedding.detach(), p=2, dim=1)
            similarities = F.linear(norm_db, norm_query).squeeze()

        top_k_indices = torch.topk(similarities, k, largest=True).indices

    friends_indices = []
    foes_indices = []
    for idx in top_k_indices:
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
