import torch
import itertools
from PIL import Image
from torchvision.models import vision_transformer

from zennit.image import imgify
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules

from lxt.efficient import monkey_patch, monkey_patch_zennit

from basemodel import load_timm_model_with_transforms

monkey_patch(vision_transformer, verbose=True)
monkey_patch_zennit(verbose=True)

MODEL_PATH = "/workspaces/vast-gorilla/gorillawatch/models/max/body_72.pth"
IMG_SIZE = 512

# load fine-tuned model and its weights
model, weigths = load_timm_model_with_transforms(
    model_path=MODEL_PATH,
    img_size=IMG_SIZE
)

# Load and preprocess the input image
image = Image.open("/workspaces/bachelor_thesis_code/src/bachelor_thesis/image.png").convert("RGB")
input_tensor = weights.transforms()(image).unsqueeze(0).to("cuda")

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

    # Get the top 5 predictions
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
    y[0, top5_classes[0]].backward()

    # Remove the registered composite to prevent interference in future iterations
    zennit_comp.remove()

    # Calculate the relevance by computing Gradient * Input
    # This is the final step of LRP to get the pixel-wise explanation
    heatmap = (input_tensor * input_tensor.grad).sum(1)

    # Normalize relevance between [-1, 1] for plotting
    heatmap = heatmap / abs(heatmap).max()

    # Store the normalized heatmap
    heatmaps.append(heatmap[0].detach().cpu().numpy())

# Visualize all heatmaps in a grid (3×5) and save to a file
# vmin and vmax control the color mapping range
imgify(heatmaps, vmin=-1, vmax=1, grid=(3, 5)).save('vit_heatmap.png')