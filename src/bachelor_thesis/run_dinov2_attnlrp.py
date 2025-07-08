import torch
from lxt.efficient import monkey_patch_zennit
import pandas as pd
import yaml
import os
from PIL import Image

from basemodel import get_model_wrapper
from dinov2_attnlrp_sweep import run_gamma_sweep
from lrp_helpers import visualize_relevances
from eval_helpers import srg

monkey_patch_zennit(verbose=True)  # is this needed? seems to be


SAVE_HEATMAPS = True 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODE = "knn"  # "simple" or "knn"

model_wrapper, transforms = get_model_wrapper()

# 3. Prepare your input image
image_path = "/workspaces/bachelor_thesis_code/sample_images/YE41_R035_20220818_091_1842_799985.png"

model_config_path = "/workspaces/bachelor_thesis_code/src/bachelor_thesis/configs/model_config.yaml"
with open(model_config_path, "r") as f:
    cfg = yaml.safe_load(f)
conv_gammas = cfg["CONV_GAMMAS"]
lin_gammas = cfg["LIN_GAMMAS"]


relevances_by_gamma, violations_by_gamma = run_gamma_sweep(
    model_wrapper=model_wrapper,
    transforms=transforms,
    input_image_path=image_path,
    device=DEVICE,
    mode=MODE,
    conv_gamma_values=conv_gammas,
    lin_gamma_values=lin_gammas
)
if SAVE_HEATMAPS:
    visualize_relevances(
        relevances=relevances_by_gamma, 
        mode=MODE, 
        image_name=os.path.basename(image_path).split(".")[0], 
        dim=(len(conv_gammas), len(lin_gammas))
    )
# eval
"""results = []
for relevance_map, conv_gamma, lin_gamma, violations in relevances_by_gamma:
    print("Evaluating generated relevance map with SRG/∆A_F...")
    faithfulness_score = srg_knn(
        relevance_map=relevance_map,
        input_tensor=input_tensor,
        model=model_wrapper, # The original, un-patched model
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
    print(f"Highest Faithfulness Score (∆A_F): {best_result['faithfulness_score']:.4f}")"""

