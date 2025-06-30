import itertools
import os

import torch
import zennit.rules as z_rules
from lxt.efficient import monkey_patch_zennit
from PIL import Image
from zennit.composites import LayerMapComposite
from zennit.image import imgify

monkey_patch_zennit(verbose=True)  # is this needed? seems to be

import datetime
import types

from basemodel import load_finetuned_timm_wrapper
# Import our custom patching function
from dino_patcher import dino_batchnorm1d_forward, DINOPatcher
from lrp_helpers import ConservationChecker

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

    # register relevance chcker
    checkers = {}
    handles = {}
    for name, module in model_wrapper.model.named_modules():
        """# Only attach to modules that have parameters or perform computation
        if list(module.children()): # Skip container modules like the top-level Sequential
            continue
        if not any(p.requires_grad for p in module.parameters()) and not isinstance(module, (nn.ReLU, nn.Flatten)):
            continue # Skip layers with no params like ReLU, Flatten if you want"""
            
        checker = ConservationChecker(f"{name} ({module.__class__.__name__})")
        handle = module.register_backward_hook(checker.hook)
        checkers[name] = checker
        handles[name] = handle


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
        
        #conservation rules violations?
        check_for_relevance_violations(checkers)
        
        
        # Clean up hooks for the next iteration
        zennit_comp.remove()

        # Visualize or save the relevance map
        heatmap = relevance.sum(1)

        denom = abs(heatmap).max()
        
        #TODO:for blue squared: you forgot to cast the model to bfloat16 and to set bfloat16 in bitsandbytes (:
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
        
    #TODO: remove conservation checker and maybe also the lrp patches from the model?
    for handle in handles.values():
        handle.remove()
    print("Removed all conservation checker hooks.")
    return relevances, heatmaps

def check_for_relevance_violations(checkers: dict[str, ConservationChecker]):
    batch_violations = {}
    for name, checker in checkers.items():
        if checker.rin is None or checker.rout is None:
            continue
        
        diff = checker.rin - checker.rout
        print(f"Layer: {checker.module_name:<30} | R_in: {checker.rin:>12.6f}, R_out: {checker.rout:>12.6f}, Diff: {diff:>12.6f}")
        if not torch.isclose(torch.tensor(checker.rin), torch.tensor(checker.rout)):
            batch_violations[name] = diff
            
    if not batch_violations:
        print("✅ All checked layers are conservative.")
    else:
        print("🔥 Found conservation violations in the following layers:")
        for name, diff in batch_violations.items():
            print(f"  - {name}: Difference = {diff}")
            
if __name__ == "__main__":
    get_relevances(save_heatmaps=SAVE_HEATMAPS)
    

