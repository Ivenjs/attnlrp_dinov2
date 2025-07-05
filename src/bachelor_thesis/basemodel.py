import logging
from typing import Any, Literal, Optional, Tuple, Union

import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.data import create_transform, resolve_model_data_config
from timm.layers.classifier import ClassifierHead, NormMlpClassifierHead
from torchvision.models import vision_transformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimmWrapper(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        embedding_size: int,
        dtype: torch.dtype = torch.float32,
        embedding_id: Literal["linear", "mlp", "linear_norm_dropout", "mlp_norm_dropout"] = "linear",
        dropout_p: float = 0.0,
        pool_mode: Literal["gem", "gap", "gem_c", "none"] = "none",
        load_pretrained: bool = False,
        pretrained_weights_path: str = None,
        img_size: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        assert pool_mode == "none" or "vit" not in backbone_name, "pool_mode is not supported for VisionTransformer."
        if img_size is not None:
            logger.info(f"Setting img_size to {img_size}")
            self.model = timm.create_model(backbone_name, pretrained=True, drop_rate=0.0, img_size=img_size)
        else:
            self.model = timm.create_model(backbone_name, pretrained=True, drop_rate=0.0)

        self.model = self.model.to(dtype=dtype)
        
        # Load pretrained weights if specified
        if load_pretrained:
            print(f"Loading pretrained weights from {pretrained_weights_path}")
            self.model.load_state_dict(torch.load(pretrained_weights_path, weights_only=True), strict=True)

        self.num_features = self.model.num_features

        self.reset_if_necessary(pool_mode)

        # added proper initialization for more stable training
        self.embedding_layer = get_embedding_layer(
            id=embedding_id, feature_dim=self.num_features, embedding_dim=embedding_size, dropout_p=dropout_p
        )
        self.pool_mode = pool_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.forward_features(x)
        x = self.model.forward_head(x, pre_logits=True)
        if x.dim() == 3:
            logger.info("Assuming VisionTransformer is used and taking the first token.")
            x = x[:, 0, :]

        x = self.embedding_layer(x)
        return x

    def reset_if_necessary(self, pool_mode: Optional[Literal["gem", "gap", "gem_c", "none"]] = None) -> None:
        if pool_mode == "none":
            pool_mode = None
        if (
            hasattr(self.model, "head")
            # NOTE(rob2u): see https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/classifier.py#L73
            and hasattr(self.model.head, "global_pool")
            and pool_mode is not None
        ):
            if isinstance(self.model.head, ClassifierHead):
                self.model.head.global_pool = get_global_pooling_layer(pool_mode, self.model.head.input_fmt)
                self.model.head.fc = nn.Identity()
                self.model.head.drop = nn.Identity()
            elif isinstance(self.model.head, NormMlpClassifierHead):
                logger.warn(
                    "Model uses NormMlpClassifierHead, for which we do not want to change the global_pooling layer."
                )
        elif pool_mode is not None and hasattr(self.model, "global_pool"):
            self.model.reset_classifier(0, "")
            self.model.global_pool = get_global_pooling_layer(pool_mode, self.num_features)
        else:
            logger.info("No pooling layer reset necessary.")


def get_global_pooling_layer(id: str, num_features: int, format: Literal["NCHW", "NHWC"] = "NCHW") -> torch.nn.Module:
    if id == "gem":
        return FormatWrapper(GeM(), format)
    elif id == "gem_c":
        return FormatWrapper(GeM_adapted(p_shape=(num_features)), format)  # TODO(rob2u): test
    elif id == "gap":
        return FormatWrapper(GAP(), format)
    else:
        return nn.Identity()


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(
        self, x: torch.Tensor, p: Union[float, torch.Tensor, nn.Parameter] = 3.0, eps: float = 1e-6
    ) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p).flatten(1)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class GeM_adapted(nn.Module):
    def __init__(
        self, p: float = 3.0, p_shape: Union[tuple[int], int] = 1, eps: float = 1e-6
    ) -> None:  # TODO(rob2u): make p_shape variable (only 1 supported currently)
        super().__init__()
        self.p = nn.Parameter(torch.ones(p_shape) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(
        self, x: torch.Tensor, p: Union[float, torch.Tensor, nn.Parameter] = 3.0, eps: float = 1e-6
    ) -> torch.Tensor:  # TODO(rob2u): find better way instead of multiplying by sign
        batch_size = x.size(0)
        x_pow_mean = x.pow(p).mean((-2, -1))
        return x_pow_mean.abs().pow(1.0 / p).view(batch_size, -1) * (x_pow_mean.view(batch_size, -1).sign()) ** p

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class GAP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class FormatWrapper(nn.Module):
    def __init__(self, pool: nn.Module, format: Literal["NCHW", "NHWC"] = "NCHW") -> None:
        super().__init__()
        assert format in ("NCHW", "NHWC")

        self.pool = pool
        self.format = format

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.format == "NHWC":
            x = x.permute(0, 2, 3, 1)
        elif self.format == "NCHW":
            x = x
        else:
            raise ValueError(f"Unknown format {self.format}")
        return self.pool(x)


def get_embedding_layer(id: str, feature_dim: int, embedding_dim: int, dropout_p: float = 0.0) -> torch.nn.Module:
    def init_linear_layer(layer):
        """Helper function to initialize linear layers"""
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

    if id == "linear":
        embedding_layer = nn.Linear(feature_dim, embedding_dim)
        init_linear_layer(embedding_layer)
        return embedding_layer
    elif id == "mlp":
        layer1 = nn.Linear(feature_dim, feature_dim)
        layer2 = nn.Linear(feature_dim, embedding_dim)
        init_linear_layer(layer1)
        init_linear_layer(layer2)
        return nn.Sequential(layer1, nn.ReLU(), layer2)
    elif "linear_norm_dropout" in id:
        linear_layer = nn.Linear(feature_dim, embedding_dim)
        init_linear_layer(linear_layer)
        return nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout_p),
            linear_layer,
            nn.BatchNorm1d(embedding_dim),
        )
    elif "mlp_norm_dropout" in id:
        layer1 = nn.Linear(feature_dim, feature_dim)
        layer2 = nn.Linear(feature_dim, embedding_dim)
        init_linear_layer(layer1)
        init_linear_layer(layer2)
        return nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout_p),
            layer1,
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout_p),
            layer2,
            nn.BatchNorm1d(embedding_dim),
        )
    else:
        return nn.Identity()


BACKBONE_NAME = "vit_giant_patch14_dinov2.lvd142m"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_clean_state_dict_for_wrapper(checkpoint, wrapper_key="model_wrapper.", model_key="model."):
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned_state_dict = {k.replace(wrapper_key, ""): v for k, v in state_dict.items()}
    return cleaned_state_dict


def load_finetuned_timm_wrapper(
    checkpoint_path: str,
    backbone_name: str,
    embedding_size: int,
    image_size: int,
    device: str = "cuda",
    model_dtype: torch.dtype = torch.float32
) -> Tuple[TimmWrapper, Any]:
    """
    Loads a finetuned TimmWrapper model from a checkpoint for inference.

    This function correctly reconstructs the model architecture using the TimmWrapper
    class before loading the saved state_dict.

    Parameters:
    - checkpoint_path (str): Path to the .pth checkpoint file.
    - backbone_name (str): The name of the timm backbone used during training (e.g., 'vit_large_patch14_dinov2.lvd142m').
    - embedding_size (int): The output dimension of the custom head used during training.
    - device (str): Device to load the model on.

    Returns:
    - tuple: (model, transforms_object)
    """
    print("--- Loading Finetuned TimmWrapper Model ---")

    checkpoint_best = torch.load(checkpoint_path, map_location=device)

    print(f"Building model architecture with backbone '{backbone_name}' and embedding size {embedding_size}...")
    model_wrapper = TimmWrapper(backbone_name=backbone_name, embedding_size=embedding_size, model_dtype=model_dtype,img_size=image_size)

    cleaned_state_dict_wrapper = extract_clean_state_dict_for_wrapper(checkpoint_best)
    msg = model_wrapper.load_state_dict(cleaned_state_dict_wrapper, strict=False)
    print(f"State dict loading message: {msg}")

    model_wrapper.to(device)
    model_wrapper.eval()

    # 5. Get the correct preprocessing transforms for the timm backbone.
    data_config = resolve_model_data_config(model_wrapper.model)
    transforms = create_transform(**data_config, is_training=False)
    print("Associated model transforms created successfully.")

    print("--- Model loading complete ---")
    return model_wrapper, transforms


def load_model(model_path: str, img_size: int) -> TimmWrapper:
    """Load a pre-trained model"""
    torch.backends.cudnn.benchmark = True  # Enable CUDNN benchmarking

    # Initialize Timm Wrapper Model
    """
    embedding_size = 518 is almost certainly wrong. 
    For dinov2_vitg14, the embedding dimension is 1536. 
    The wrapper is likely misconfigured and might either fail or produce incorrect results.
    """
    embedding_size = 518
    dropout_p = 0.0
    embedding_id = "linear"
    pool_mode = "none"

    embedding_model = TimmWrapper(
        backbone_name=BACKBONE_NAME,
        embedding_size=embedding_size,
        embedding_id=embedding_id,
        dropout_p=dropout_p,
        pool_mode=pool_mode,
        img_size=img_size,
    ).to(DEVICE)

    # Load the model weights
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    # remove the prefix "module." from the keys of the state_dict
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    embedding_model.load_state_dict(state_dict)
    embedding_model = embedding_model.to(memory_format=torch.channels_last)
    embedding_model.eval()

    return embedding_model


def get_vit_imagenet(device="cuda"):
    """
    Load a pre-trained Vision Transformer (ViT) model with ImageNet weights.

    Parameters:
    device (str): Device to load the model on ('cuda' or 'cpu')

    Returns:
    tuple: (model, weights) - The ViT model and its pre-trained weights
    """
    weights = vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
    model = vision_transformer.vit_b_16(weights=weights)
    model.eval()
    model.to(device)

    # Deactivate gradients on parameters to save memory
    for param in model.parameters():
        param.requires_grad = False

    return model, weights


def load_dino_with_transforms(
    checkpoint_path, model_name="dinov2_vitg14", device="cuda"
) -> tuple[torch.nn.Module, Any]:
    """
    Load a finetuned DINOv2 model and its corresponding preprocessing transforms.

    Parameters:
    - checkpoint_path (str): Path to the .pth or .pt checkpoint file.
    - model_name (str): The name of the base DINOv2 model. This function works for
                        'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', and 'dinov2_vitg14'.
    - device (str): Device to load the model on ('cuda' or 'cpu').

    Returns:
    - tuple: (model, transforms_object)
    """
    print(f"Loading base DINOv2 model: {model_name}...")
    model = torch.hub.load("facebookresearch/dinov2", model_name)

    # ... (rest of the weight loading logic is identical) ...
    print(f"Loading finetuned weights from: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "teacher" in checkpoint:
        state_dict = checkpoint["teacher"]
    else:
        state_dict = checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"State dict loading message: {msg}")
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False

    # This resolution is optimal for ALL standard DINOv2 models, as they all use a 14x14 patch size.
    # 518 is a multiple of 14 (518 = 37 * 14).
    img_size = 518

    custom_transforms = T.Compose(
        [
            T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class DINOv2Transforms:
        def __call__(self, img):
            return custom_transforms(img)

    print("Finetuned DINOv2 model and transforms loaded successfully.")
    return model, DINOv2Transforms()
