import torch
import itertools
from PIL import Image

# Your model loading function
from basemodel import load_dino_with_transforms, load_finetuned_timm_wrapper

# Zennit imports
from zennit.image import imgify
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules

# LXT and Python operator imports
import operator
import torch.nn as nn
import torch.nn.functional as F
from lxt.explicit.core import Composite
import lxt.explicit.functional as lf
import lxt.explicit.modules as lm
import lxt.explicit.rules as rules

from lxt.explicit.modules import INIT_MODULE_MAPPING

def initialize_dino_attention(original, replacement):
    """
    Custom initializer to transfer weights from a DINOv2 MemEffAttention
    module to an lxt MultiheadAttention_CP module.
    
    This replaces the default `initialize_MHA` which is only compatible
    with `torch.nn.MultiheadAttention`.
    """
    print(f"Running CUSTOM initializer for {type(original).__name__}...")
    
    # Instantiate the replacement module
    replacement = replacement()

    # DINOv2's MemEffAttention stores Q, K, and V weights concatenated
    # in a single 'qkv' linear layer. We need to split them.
    embed_dim = original.qkv.in_features
    
    # Split the concatenated qkv weights and biases into three parts
    q_proj_weight, k_proj_weight, v_proj_weight = original.qkv.weight.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = original.qkv.bias.chunk(3, dim=0)

    # Assign the weights to the corresponding attributes in the LXT replacement module
    replacement.q_proj_weight = q_proj_weight
    replacement.k_proj_weight = k_proj_weight
    replacement.v_proj.weight = v_proj_weight
    
    replacement.bias_q = q_proj_bias
    replacement.bias_k = k_proj_bias
    replacement.v_proj.bias = v_proj_bias

    # Copy the output projection weights and biases
    replacement.out_proj.weight = original.proj.weight
    replacement.out_proj.bias = original.proj.bias

    # Copy metadata required by the LXT attention forward pass
    replacement.embed_dim = embed_dim
    replacement.num_heads = original.num_heads
    replacement.head_dim = embed_dim // original.num_heads
    replacement.batch_first = True # DINOv2 ViT is batch-first

    return replacement

# --- Override the default LXT initializer for our specific use case ---
# We tell lxt: "When you need to initialize a MultiheadAttention_CP,
# use our new custom function, not your default one."
INIT_MODULE_MAPPING[lm.MultiheadAttention_CP] = initialize_dino_attention

# --- Setup ---
CHECKPOINT_PATH = "/workspaces/vast-gorilla/gorillawatch/data/models/max_checkpoints/saved_checkpoints/giantbodybest74ens82.pth"
IMG_SIZE = 518
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BACKBONE = "vit_giant_patch14_dinov2.lvd142m"
EMBEDDING_DIM = 256 # The output size you trained for



model_wrapper, weights = load_finetuned_timm_wrapper(
    checkpoint_path=CHECKPOINT_PATH,
    backbone_name=BACKBONE,
    embedding_size=EMBEDDING_DIM,
    image_size=IMG_SIZE,
    device=DEVICE,
)

# 2. Programmatically get the specific class types from YOUR loaded model
DINO_ATTENTION_CLASS = type(model_wrapper.model.blocks[0].attn)
DINO_MLP_CLASS = type(model_wrapper.model.blocks[0].mlp)
DINO_LAYERSCALE_CLASS = type(model_wrapper.model.blocks[0].ls1)

# 3. Define the custom LRP composite tailored specifically for DINOv2
dinov2_attnlrp = Composite({
    # --- DINOv2 Specific Module Mappings ---
    # For DINOv2's custom `MemEffAttention` module. We replace it with the generic
    # LRP-aware ViT attention module provided by lxt.
    DINO_ATTENTION_CLASS: lm.MultiheadAttention_CP,
    
    # For DINOv2's `SwiGLUFFNFused` MLP. Since there's no custom LRP module for this,
    # we apply the general-purpose EpsilonRule, treating it as a single non-linear block.
    DINO_MLP_CLASS: rules.EpsilonRule,
    
    # For the standard `nn.LayerNorm` used by DINOv2. We replace it with the purpose-built
    # LRP-aware LayerNorm module from lxt.
    nn.LayerNorm: lm.LayerNormEpsilon,
    
    # For DINOv2's `LayerScale` module. It's a simple scaling, so the IdentityRule,
    # which passes relevance through unchanged, is the correct choice.
    DINO_LAYERSCALE_CLASS: rules.IdentityRule,
    
    # For DINOv2's `DropPath` (which is `nn.Identity` at inference).
    # The IdentityRule correctly passes relevance through.
    nn.Identity: rules.IdentityRule,

    # --- Rules for LRP-replacement modules themselves (from lm.MultiheadAttention_CP) ---
    # These are needed to configure the *internals* of the replacement attention module.
    lm.LinearInProjection: rules.EpsilonRule,
    lm.LinearOutProjection: rules.EpsilonRule,
    
    # --- Generic Functional Rules (to be replaced by the tracer) ---
    # For residual connections (hidden_states = residual + hidden_states)
    operator.add: lf.add2,
    # For any other matrix multiplications found by the tracer
    operator.matmul: lf.matmul,
    # For any other normalization functions found by the tracer
    F.normalize: lf.normalize,
})

# 4. Trace the model using the CORRECT dummy input format
dummy_tensor = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

# The key 'x' corresponds to the name of the input argument in the
# DINOv2 model's forward method (def forward(self, x): ...)
dummy_inputs_dict = {'x': dummy_tensor}

print("\nTracing the DINOv2 model with our custom AttnLRP composite...")
# Pass the dictionary, not the raw tensor
traced_model = dinov2_attnlrp.register(
    model, 
    dummy_inputs=dummy_inputs_dict, 
    verbose=True
)
print("Tracing complete.")

# Load and preprocess the real input image
image = Image.open("/workspaces/bachelor_thesis_code/src/bachelor_thesis/image.png").convert("RGB")
input_tensor = weights(image).unsqueeze(0).to(DEVICE)

heatmaps = []

# 5. Run the LRP loop with the fine-tuning zennit composite
for conv_gamma, lin_gamma in itertools.product([0.1, 0.25, 100], [0, 0.01, 0.05, 0.1, 1]):
    input_tensor.grad = None
    print("\nGamma Conv2d:", conv_gamma, "Gamma Linear/MLP:", lin_gamma)

    # This zennit composite fine-tunes the rules set by the main dinov2_attnlrp tracer
    zennit_comp = LayerMapComposite([
        (nn.Conv2d, z_rules.Gamma(conv_gamma)),
        (nn.Linear, z_rules.Gamma(lin_gamma)),
        # Also apply the gamma rule to the entire MLP block
        (DINO_MLP_CLASS, z_rules.Gamma(lin_gamma)),
    ])
    zennit_comp.register(traced_model)

    y = traced_model(input_tensor.requires_grad_())
    target_scalar = y.sum()
    target_scalar.backward()
    zennit_comp.remove()

    heatmap = (input_tensor * input_tensor.grad).sum(1)
    heatmap = heatmap / abs(heatmap).max()
    heatmaps.append(heatmap[0].detach().cpu().numpy())

# Visualize the final heatmaps
imgify(heatmaps, vmin=-1, vmax=1, grid=(3, 5)).save('vit_heatmap_fast.png')