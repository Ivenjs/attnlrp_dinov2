import open_clip
import torch
import torch.nn as nn
from basemodel import load_dino_with_transforms

DINO_MODEL_PATH = "/workspaces/vast-gorilla/gorillawatch/models/ViT-giant-400epochs.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dino, _ = load_dino_with_transforms(checkpoint_path=DINO_MODEL_PATH, model_name="dinov2_vitg14", device=DEVICE)

# Print the first block to see the component class names
print("--- DINOv2 Architecture (Corrected Investigation) ---")
first_block = model_dino.blocks[0]
print("First Transformer Block:")
print(first_block)
print("\n--- Key Component Types ---")
print(f"Attention Class: {type(first_block.attn)}")
print(f"LayerNorm Class: {type(first_block.norm1)}")
print(f"MLP Block Class: {type(first_block.mlp)}")


model_clip, _, _ = open_clip.create_model_and_transforms("ViT-g-14", pretrained="laion2b_s34b_b88k")

print("\n--- OpenCLIP Architecture (Corrected Investigation) ---")
# Path to the first block in the visual transformer
first_block_clip = model_clip.visual.transformer.resblocks[0]

print("Attention Class:", type(first_block_clip.attn))
print("LayerNorm Class:", type(first_block_clip.ln_1))

# Instead of trying to access a non-existent '.act', print the whole MLP container
print("\nMLP Block Structure:")
print(first_block_clip.mlp)

# Now, we can programmatically find the activation layer by iterating through the container
activation_layer = None
for layer in first_block_clip.mlp:
    if isinstance(layer, nn.GELU):  # Or whatever activation it uses
        activation_layer = layer
        break

if activation_layer:
    print(f"\nFound MLP Activation Class: {type(activation_layer)}")
else:
    print("\nCould not find a GELU activation layer in the MLP.")
