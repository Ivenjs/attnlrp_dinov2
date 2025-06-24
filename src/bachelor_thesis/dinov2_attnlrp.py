import itertools
import os

import timm
import torch
import zennit.rules as z_rules
from lxt.efficient import monkey_patch_zennit
from PIL import Image
from torchvision import transforms
from zennit.composites import LayerMapComposite
from zennit.image import imgify

monkey_patch_zennit(verbose=True)  # is this needed?

import datetime
import types

from basemodel import load_finetuned_timm_wrapper
# Import our custom patching function
from patch_dino import dino_batchnorm1d_forward, patch_dinov2_for_lrp

# 1. Load the DinoV2 model using timm
CHECKPOINT_PATH = (
    "/workspaces/vast-gorilla/gorillawatch/data/models/max_checkpoints/saved_checkpoints/giantbodybest74ens82.pth"
)
IMG_SIZE = 518
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BACKBONE = "vit_giant_patch14_dinov2.lvd142m"
EMBEDDING_DIM = 256  # The output size you trained for
SAVE_HEATMAPS = True  # Set to False if you don't want to save the heatmaps


def get_relevances(save_heatmaps: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
    model_wrapper, weights = load_finetuned_timm_wrapper(
        checkpoint_path=CHECKPOINT_PATH,
        backbone_name=BACKBONE,
        embedding_size=EMBEDDING_DIM,
        image_size=IMG_SIZE,
        device=DEVICE,
    )

    # 2. Apply our custom patches for LRP rules
    model_wrapper.model = patch_dinov2_for_lrp(
        model_wrapper.model, "cp_lrp"
    )  # for vision transformer models, use "cp_lrp"

    # patch custom BatchNorm1d forward method
    for module in model_wrapper.embedding_layer.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.forward = types.MethodType(dino_batchnorm1d_forward, module)
            print(f"  - Patched {module}")

    # Note: If our dino head has other custom non-linearities (e.g., ReLU),
    # you could patch them here too with identity_rule_implicit.
    # For a standard ReLU, this is good practice:
    #   if isinstance(module, torch.nn.ReLU):
    #       ...

    # 3. Prepare your input image
    image = Image.open("/workspaces/bachelor_thesis_code/seg_test_in/NN00_R019_20221212_281_378_287939.png").convert(
        "RGB"
    )
    input_tensor = weights(image).unsqueeze(0).to(DEVICE)

    heatmaps = []
    relevances = []  # Store the relevance maps for each gamma combination
    # 4. Use zennit to apply the gamma-rule to Linear layers
    # This is the same as the standard ViT example and works in tandem with our patches.
    for conv_gamma, lin_gamma in itertools.product([0.1, 0.25, 1.0], [0.0, 0.05, 0.1, 0.25]):
        input_tensor.grad = None  # Reset gradients
        print(f"\n--- Running LRP with Gamma (Conv2d: {conv_gamma}, Linear: {lin_gamma}) ---")

        # zennit will handle the gamma rule for both nn.Linear and nn.Conv2d layers.
        zennit_comp = LayerMapComposite(
            [
                (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
                (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
            ]
        )

        # The zennit `register` call will NOT override our manually patched forward methods.
        # It only adds its own forward/backward hooks, which is exactly what we want.
        zennit_comp.register(model_wrapper)  # Register on the full wrapper

        # Forward pass
        output = model_wrapper(input_tensor.requires_grad_())

        # Get the top prediction
        # TODO: This is a simplification
        # We want to explain why the input image (the "query") is considered
        # similar to its nearest neighbors of the correct class and dissimilar to its
        # nearest neighbors of incorrect classes. We can formulate this as a
        # pseudo-classification problem.
        most_active_feature_idx = torch.argmax(output, dim=1).item()
        print(f"Explaining the most active feature dimension: {most_active_feature_idx}")

        # Backward pass to compute LRP relevances
        output[0, most_active_feature_idx].backward()

        # Get relevance map
        relevance = input_tensor * input_tensor.grad
        relevances.append(relevance)
        # Clean up hooks for the next iteration
        zennit_comp.remove()

        # Visualize or save the relevance map
        heatmap = relevance.sum(1)

        denom = abs(heatmap).max()
        if denom == 0 or torch.isnan(denom):
            print("Warning: Zero or NaN max in heatmap. Skipping normalization.")
            # fill heatmap with ones
            # heatmap = torch.ones_like(heatmap)
            heatmap = torch.zeros_like(heatmap)
        else:
            heatmap = heatmap / denom

        heatmaps.append(heatmap[0].detach().cpu().numpy())

    if save_heatmaps:
        save_path = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/heatmaps"
        os.makedirs(save_path, exist_ok=True)
        current_dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        imgify(heatmaps, vmin=-1, vmax=1, grid=(3, 5)).save(f"{save_path}/dinov2_heatmap{current_dt}.png")

    return relevances, heatmaps


if __name__ == "__main__":
    get_relevances(save_heatmaps=SAVE_HEATMAPS)
