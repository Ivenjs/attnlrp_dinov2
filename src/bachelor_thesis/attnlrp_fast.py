import torch
import itertools
from PIL import Image
from torchvision.models import vision_transformer

from zennit.image import imgify
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules

from lxt.efficient import monkey_patch, monkey_patch_zennit

from basemodel import load_timm_model_with_transforms, load_dino_with_transforms, get_vit_imagenet, load_finetuned_timm_wrapper


#TODO: This is stupid, because i monkey patch the vision_transformer I end up not even using, since I use DINOv2
monkey_patch(vision_transformer, verbose=True)
monkey_patch_zennit(verbose=True)

CHECKPOINT_PATH = "/workspaces/vast-gorilla/gorillawatch/models/ViT-giant-400epochs.pt"
IMG_SIZE = 518
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BACKBONE = "vit_giant_patch14_dinov2.lvd142m"
EMBEDDING_DIM = 256 # The output size you trained for


# load fine-tuned model and its weights
#model, weigths = load_timm_model_with_transforms(
#    model_path=MODEL_PATH,
#    img_size=IMG_SIZE
#)
#model, weights = get_vit_imagenet(device=DEVICE)
"""model, weights = load_dino_with_transforms(
    checkpoint_path=CHECKPOINT_PATH, # Use your finetuned checkpoint here
    model_name=BACKBONE, # Specify the giant model
    device=DEVICE
)"""

gorilla_model, weights = load_finetuned_timm_wrapper(
    checkpoint_path=CHECKPOINT_PATH,
    backbone_name=BACKBONE,
    embedding_size=EMBEDDING_DIM,
    device=DEVICE,
)

# Load and preprocess the input image
image = Image.open("/workspaces/bachelor_thesis_code/src/bachelor_thesis/image.png").convert("RGB")
#input_tensor = weights.transforms()(image).unsqueeze(0).to("cuda")
input_tensor = weights(image).unsqueeze(0).to(DEVICE)


# Store the generated heatmaps
heatmaps = []

# Experiment with different gamma values for Conv2d and Linear layers
# Gamma is a hyperparameter in LRP that controls how much positive vs. negative
# contributions are considered in the explanation
for conv_gamma, lin_gamma in itertools.product([0.1, 0.25, 100], [0, 0.01, 0.05, 0.1, 1]):
    input_tensor.grad = None  # Reset gradients
    print("Gamma Conv2d:", conv_gamma, "Gamma Linear:", lin_gamma)

    # Define rules for the Conv2d and Linear layers using 'zennit'
    # LayerMapComposite maps specific layer types to specific LRP rule implementations
    zennit_comp = LayerMapComposite([
        (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
        (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
    ])

    # Register the composite rules with the model
    zennit_comp.register(model)

    # Forward pass with gradient tracking enabled
    y = model(input_tensor.requires_grad_())

    """# Get the top 5 predictions
    _, top5_classes = torch.topk(y, 5, dim=1)
    top5_classes = top5_classes.squeeze(0).tolist()

    # Get the class labels
    labels = weights.meta["categories"]
    top5_labels = [labels[class_idx] for class_idx in top5_classes]

    # Print the top 5 predictions and their labels
    for i, class_idx in enumerate(top5_classes):
        print(f'Top {i+1} predicted class: {class_idx}, label: {top5_labels[i]}')

    # Backward pass for the highest probability class
    # This initiates the LRP computation through the network
    y[0, top5_classes[0]].backward()"""
    # 2. Create a single scalar target by summing all feature activations
    # This represents the model's "total output energy" or "overall focus".
    target_scalar = y.sum()

    print(f"Aggregated target scalar value: {target_scalar.item():.4f}")

    # 3. Perform a SINGLE backward pass from this aggregated target
    target_scalar.backward()

    # 4. Calculate the heatmap as before. It now represents the
    # combined influence of all features.
    heatmap = (input_tensor * input_tensor.grad).sum(1)

    # 5. Normalize and store
    heatmap = heatmap / abs(heatmap).max()
    heatmaps.append(heatmap[0].detach().cpu().numpy())

# Visualize all heatmaps in a grid (3×5) and save to a file
# vmin and vmax control the color mapping range
imgify(heatmaps, vmin=-1, vmax=1, grid=(3, 5)).save('vit_heatmap_fast.png')