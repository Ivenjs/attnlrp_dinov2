import types

import torch
import torch.nn as nn
from lxt.efficient.rules import divide_gradient, identity_rule_implicit, stop_gradient
from timm.layers.layer_scale import LayerScale
from timm.layers.mlp import GluMlp
from torch.autograd import Function

"""
=== DINO Transformer Block[0] Structure Overview ===

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
"""

class DINOPatcher:
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

        for name, module in self.wrapper.embedding_layer.named_modules():
            key = f"embedding_layer.{name}"
            if isinstance(module, nn.BatchNorm1d):
                self.original_forwards[key] = module.forward
                module.forward = types.MethodType(dino_batchnorm1d_forward, module)
                print(f"  - Patched {key} ({module.__class__.__name__})")

        return self.wrapper 

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.original_forwards:
            return

        # We need to iterate over the whole wrapper to find the modules by name
        for name, module in self.wrapper.named_modules():
            if name in self.original_forwards:
                module.forward = self.original_forwards[name]
        


# -------------------------------------------------------------------
# Forward Pass for timm Attention
# -------------------------------------------------------------------
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
    else:  # Manual implementation
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
    else:  # Manual implementation
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
    # Project to the Gated and Value paths
    x_gate, x_value = self.fc1(x).chunk(2, dim=-1)

    # --- LRP Rule 1: Identity rule on activation ---
    gate = identity_rule_implicit(self.act, x_gate)
    # gate = safe_identity_rule_implicit(self.act, x_gate)
    # ---------------------------------------------

    # Apply dropout if needed
    gate = self.drop1(gate)

    # --- LRP Rule 2: Uniform rule on element-wise multiplication ---
    # The relevance is split equally between the gate and the value
    weighted_value = gate * x_value
    weighted_value = divide_gradient(weighted_value, 2)
    # --------------------------------------------------------------

    # Apply norm if it exists
    weighted_value = self.norm(weighted_value)

    # Final projection
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

    if self.elementwise_affine:
        x = self.weight * x + self.bias
    return x


def safe_identity_rule_implicit(fn, input):
    """
    A more numerically stable version of the identity rule.
    """
    return safe_identity_rule_implicit_fn.apply(fn, input)


class safe_identity_rule_implicit_fn(Function):
    @staticmethod
    def forward(ctx, fn, input):
        output = fn(input)
        if input.requires_grad:
            # The original library's epsilon is very small (1e-10).
            # We use a larger one and a sign check for stability.
            epsilon = 1e-9

            # Key change: Use a safe divisor
            # We add epsilon to the magnitude of the input to avoid issues near zero,
            # and then restore the original sign. This prevents division by a tiny number.
            safe_divisor = torch.sign(input) * torch.max(torch.abs(input), torch.tensor(epsilon, device=input.device))

            ctx.save_for_backward(output / safe_divisor)
        return output

    @staticmethod
    def backward(ctx, *out_relevance):
        gradient = ctx.saved_tensors[0] * out_relevance[0]
        if torch.isnan(gradient).any() or torch.isinf(gradient).any():
            print("WARNING: NaN or Inf detected in safe_identity_rule backward pass. Returning zero gradient.")
            return None, torch.zeros_like(out_relevance[0]), None
        return None, gradient, None


# -------------------------------------------------------------------
# The Master Patching Function
# -------------------------------------------------------------------
def patch_dinov2_for_lrp(model: nn.Module, attention_mode: str) -> nn.Module:
    """
    Applies LRP patches to a timm-based DinoV2 model in-place,
    handling Attention, GluMlp, LayerScale, and LayerNorm.
    """
    print("Patching DinoV2 model for LRP...")

    if attention_mode == "attn_lrp":
        attn_patch_fn = dino_attention_forward
    elif attention_mode == "cp_lrp":
        attn_patch_fn = dino_attention_forward_cp
    else:
        raise ValueError("attention_mode must be 'attn_lrp' or 'cp_lrp'")

    for i, block in enumerate(model.blocks):
        # Patch Attention
        block.attn.forward = types.MethodType(attn_patch_fn, block.attn)

        # Patch LayerNorms
        block.norm1.forward = types.MethodType(dino_layernorm_forward, block.norm1)
        block.norm2.forward = types.MethodType(dino_layernorm_forward, block.norm2)

        # Patch GluMlp
        if isinstance(block.mlp, GluMlp):
            block.mlp.forward = types.MethodType(dino_glumlp_forward, block.mlp)
        else:
            print(f"  - WARNING: Block {i} does not contain a GluMlp. Skipping MLP patch.")

        # Patch LayerScales
        if hasattr(block, "ls1") and isinstance(block.ls1, LayerScale):
            block.ls1.forward = types.MethodType(dino_layerscale_forward, block.ls1)
        if hasattr(block, "ls2") and isinstance(block.ls2, LayerScale):
            block.ls2.forward = types.MethodType(dino_layerscale_forward, block.ls2)

        print(f"  - Patched block {i}: Attention, Norms, GluMlp, LayerScales")

    # Patch the final LayerNorm after the blocks
    if hasattr(model, "norm") and isinstance(model.norm, nn.LayerNorm):
        model.norm.forward = types.MethodType(dino_layernorm_forward, model.norm)
        print("  - Patched final model.norm LayerNorm")

    print("Patching complete.")
    return model


def dino_batchnorm1d_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Custom forward pass for nn.BatchNorm1d with LRP rule (stop_gradient).
    This re-implements the BatchNorm1d forward pass to isolate the variance.
    Only implements the eval() mode behavior as it's for inference.
    """
    if not self.training:
        # Use running mean and var in eval mode
        mean = self.running_mean
        var = self.running_var

        # Reshape mean and var to be broadcastable with input
        view_shape = (1, -1) + (1,) * (x.dim() - 2)
        mean = mean.view(view_shape)
        var = var.view(view_shape)

        # --- LRP Rule: Stop gradient on variance ---
        x = (x - mean) / stop_gradient(torch.sqrt(var + self.eps))
        # ------------------------------------------

        if self.affine:
            x = self.weight.view(view_shape) * x + self.bias.view(view_shape)
        return x
    else:
        # Fallback to original forward in training mode
        # Note: LRP is typically done in eval mode.
        return torch.nn.functional.batch_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps
        )

class ConservationChecker:
    def __init__(self, module_name):
        self.module_name = module_name
        self.rin = None
        self.rout = None

    def hook(self, module, grad_input, grad_output):
        # grad_output is a tuple of gradients. We are interested in the first one.
        if grad_output[0] is not None:
            self.rin = grad_output[0].sum().item()
        
        # grad_input is also a tuple.
        if grad_input[0] is not None:
            self.rout = grad_input[0].sum().item()
        else:
            # For the very first layer, grad_input might be None
            self.rout = self.rin 

        if self.rin is not None and self.rout is not None:
            diff = self.rin - self.rout
            if not torch.isclose(torch.tensor(self.rin), torch.tensor(self.rout)):
                 print(
                    f"Conservation violation in {self.module_name}: "
                    f"R_in={self.rin:.6f}, R_out={self.rout:.6f}, Diff={diff:.6f}"
                )
