import types

import torch
import torch.nn as nn
from lxt.efficient.rules import divide_gradient, identity_rule_implicit, stop_gradient
from timm.layers.layer_scale import LayerScale
from timm.layers.mlp import GluMlp
from timm.models.vision_transformer import Attention 

"""
--- DINOv2 Image Backbone ---

1) Patch Embedding (`model.patch_embed`)

(proj) Conv2d:
Conv2d(3, 1536, kernel_size=(14, 14), stride=(14, 14)) 

(norm) Identity:
Identity() 


2) Transformer Blocks (`model.blocks`)

The model has 40 identical blocks. Showing Block[0] as a representative example:

--- DINO Transformer Block[0] Structure ---

--- norm1 ---
LayerNorm((1536,), eps=1e-06, elementwise_affine=True)

--- attn ---
Attention(
  (qkv): Linear(in_features=1536, out_features=4608, bias=True)
  (q_norm): Identity()
  (k_norm): Identity()
  (attn_drop): Dropout(p=0.0, inplace=False)
  (proj): Linear(in_features=1536, out_features=1536, bias=True)
  (proj_drop): Dropout(p=0.0, inplace=False)
)

--- ls1 ---
LayerScale()

--- drop_path1 ---
Identity()

--- norm2 ---
LayerNorm((1536,), eps=1e-06, elementwise_affine=True)

--- mlp ---
GluMlp(
  (fc1): Linear(in_features=1536, out_features=8192, bias=True)
  (act): SiLU()
  (drop1): Dropout(p=0.0, inplace=False)
  (norm): Identity()
  (fc2): Linear(in_features=4096, out_features=1536, bias=True)
  (drop2): Dropout(p=0.0, inplace=False)
)

--- ls2 ---
LayerScale()

--- drop_path2 ---
Identity()


3) Final Head (`model.head`)

norm: LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
fc_norm: Identity()
head: Identity()

============================================================

--- Separate Feature Embedding Layer ---

Linear(in_features=1536, out_features=256, bias=True)

============================================================
"""

class DINOPatcher:
    # Use cp_lrp for ViTs
    def __init__(self, model_wrapper, attention_mode="cp_lrp"):
        self.wrapper = model_wrapper
        self.attention_mode = attention_mode
        self.original_forwards = {}

    def __enter__(self):
        if self.attention_mode == "attn_lrp":
            attn_patch_fn = dino_attention_forward
        else:
            attn_patch_fn = dino_attention_forward_cp

        for name, module in self.wrapper.model.named_modules():
            key = f"model.{name}" 

            if isinstance(module, Attention):
                self.original_forwards[key] = module.forward
                module.forward = types.MethodType(attn_patch_fn, module)

            elif isinstance(module, nn.LayerNorm):
                self.original_forwards[key] = module.forward
                module.forward = types.MethodType(dino_layernorm_forward, module)

            elif isinstance(module, GluMlp):
                self.original_forwards[key] = module.forward
                module.forward = types.MethodType(dino_glumlp_forward, module)

            elif isinstance(module, LayerScale):
                self.original_forwards[key] = module.forward
                module.forward = types.MethodType(dino_layerscale_forward, module)

        return self.wrapper 

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.original_forwards:
            return

        for name, module in self.wrapper.named_modules():
            if name in self.original_forwards:
                module.forward = self.original_forwards[name]
        


# -------------------------------------------------------------------
# Forward Pass for timm Attention
# -------------------------------------------------------------------
"""
    https://github.com/rachtibat/LRP-eXplains-Transformers/blob/main/lxt/explicit/modules.py#L87:
    Implementing the CP-LRP (Conservative Propagation - LRP) rule for attention i.e. we don't let relevance flow through the softmax, but only through the value path. 
    This method *only works well in Vision Transformers* because here the advanced AttnLRP rules, which do use the softmax, have similar performance to CP-LRP rules. 
    The issue with AttnLRP is that using the softmax introduces gradient shattering, which requires applying the z-plus LRP rule. 
    This makes AttnLRP slightly less efficient and, based on our limited experiments, the small performance gain is not worthwhile in Vision Transformers.
    However, in Large Language Models, applying AttnLRP on the softmax is substantially better than CP-LRP and does not require the less efficient z-plus rule.
    Therefore, we choose the more efficient CP-LRP for attention and use AttnLRP for other parts of the ViT.

    Please refer to Section A.2.3. 'Tackling Noise in Vision Transformers' of the the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'.
"""
def dino_attention_forward_cp(self, x: torch.Tensor) -> torch.Tensor:
    """
    Custom forward pass for timm.models.vision_transformer.Attention
    with CP-LRP rules applied (stop_gradient on q and k).
    """
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    # --- CP-LRP Rule: Stop gradient flow to q and k ---
    q = stop_gradient(q)
    k = stop_gradient(k)
    # ----------------------------------------------------

    # The rest of the forward pass is standard
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        x = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
    else:  
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, v)

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def dino_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Custom forward pass for timm.models.vision_transformer.Attention
    with LRP rules applied.
    """
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        q = divide_gradient(q, 2)
        k = divide_gradient(k, 2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = divide_gradient(x, 2)
    else:  
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = divide_gradient(attn, 2)  # <-- Uniform Rule
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, v)
        x = divide_gradient(x, 2)  # <-- Uniform Rule

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


# -------------------------------------------------------------------
# Custom Forward Pass for Gated MLP (GluMlp)
# -------------------------------------------------------------------
def dino_glumlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Custom forward pass for timm.models.vision_transformer.GluMlp
    with LRP rules applied for gated activations.
    """
    x_gate, x_value = self.fc1(x).chunk(2, dim=-1)

    # --- LRP Rule 1: Identity rule on activation ---
    gate = identity_rule_implicit(self.act, x_gate)
    # ---------------------------------------------

    gate = self.drop1(gate)

    # --- LRP Rule 2: Uniform rule on element-wise multiplication ---
    weighted_value = gate * x_value
    weighted_value = divide_gradient(weighted_value, 2)
    # --------------------------------------------------------------

    weighted_value = self.norm(weighted_value)

    x = self.fc2(weighted_value)
    x = self.drop2(x)
    return x


# -------------------------------------------------------------------
# Custom Forward Pass for LayerScale
# -------------------------------------------------------------------
def dino_layerscale_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Custom forward pass for timm.models.vision_transformer.LayerScale
    with LRP rule applied for element-wise multiplication.
    """
    # The original forward is `x * self.gamma`
    # --- LRP Rule: Uniform rule on element-wise multiplication ---
    # Split relevance equally between the input x and the scaling factor gamma
    return divide_gradient(x * self.gamma, 2)
    # --------------------------------------------------------------


# -------------------------------------------------------------------
# Custom Forward Pass for LayerNorm
# -------------------------------------------------------------------
def dino_layernorm_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Custom forward pass for nn.LayerNorm with LRP rule (stop_gradient).
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False)

    # --- LRP Rule: Stop gradient on variance ---
    x = (x - mean) / stop_gradient(torch.sqrt(var + self.eps))
    # ------------------------------------------

    if self.weight is not None:
        x *= self.weight
    if self.bias is not None:
        x += self.bias
    return x

