import torch
from lxt.efficient import monkey_patch_zennit
from PIL import Image
import pandas as pd

from basemodel import load_finetuned_timm_wrapper
from dinov2_attnlrp_sweep import run_gamma_sweep
from lrp_helpers import srg

monkey_patch_zennit(verbose=True)  # is this needed? seems to be

# SETUP, maybe move this to a config file
CHECKPOINT_PATH = (
    "/workspaces/vast-gorilla/gorillawatch/data/models/max_checkpoints/saved_checkpoints/giantbodybest74ens82.pth"
)
IMG_SIZE = 518
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BACKBONE = "vit_giant_patch14_dinov2.lvd142m"
EMBEDDING_DIM = 256  # The output size you trained for
SAVE_HEATMAPS = True  # Set to False if you don't want to save the heatmaps
PATCH_SIZE = 14  # The patch size used in the DINOv2 model


model_wrapper, weights = load_finetuned_timm_wrapper(
    checkpoint_path=CHECKPOINT_PATH,
    backbone_name=BACKBONE,
    embedding_size=EMBEDDING_DIM,
    image_size=IMG_SIZE,
    device=DEVICE,
)


# 3. Prepare your input image
image = Image.open("/workspaces/bachelor_thesis_code/seg_test_in/NN00_R019_20221212_281_378_287939.png").convert(
    "RGB"
)
input_tensor = weights(image).unsqueeze(0).to(DEVICE)


relevances_by_gamma, violations_by_gamma = run_gamma_sweep(
    model_wrapper=model_wrapper,
    input_tensor=input_tensor
)

# eval
results = []
for relevance_map, conv_gamma, lin_gamma, violations in relevances_by_gamma:
    print("Evaluating generated relevance map with SRG/∆A_F...")
    faithfulness_score = srg(
        relevance_map=relevance_map,
        input_tensor=input_tensor,
        model=model_wrapper, # The original, un-patched model
        target_class_idx=-100000000,  
        patch_size=PATCH_SIZE,
    )
    
    results.append({
        "conv_gamma": conv_gamma,
        "lin_gamma": lin_gamma,
        "faithfulness_score": faithfulness_score
    })
        
if not results:
    print("No results were generated.")
else:
    # Use pandas for easy sorting and display
    results_df = pd.DataFrame(results)
    
    print("\n\n--- Hyperparameter Search Complete ---")
    print("Full Results:")
    print(results_df.to_string())

    # Find the best result
    best_result = results_df.loc[results_df['faithfulness_score'].idxmax()]
    
    print("\n--- Best Hyperparameters ---")
    print(f"Convolutional Gamma: {best_result['conv_gamma']}")
    print(f"Linear Gamma:        {best_result['lin_gamma']}")
    print(f"Highest Faithfulness Score (∆A_F): {best_result['faithfulness_score']:.4f}")

    # You can now re-run LRP with these optimal parameters to generate
    # the final, "best" heatmap for visualization.

