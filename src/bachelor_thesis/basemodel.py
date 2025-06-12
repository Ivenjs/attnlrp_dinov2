import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers.classifier import ClassifierHead, NormMlpClassifierHead
from timm.data import create_transform, resolve_data_config
from typing import Any, Literal, Optional, Union
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimmWrapper(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        embedding_size: int,
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
        
        # Load pretrained weights if specified
        if load_pretrained:
            print(f"Loading pretrained weights from {pretrained_weights_path}")
            self.model.load_state_dict(torch.load(pretrained_weights_path,  weights_only=True), strict=True)
        
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
        return nn.Sequential(
            layer1,
            nn.ReLU(),
            layer2
        )
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
            
BACKBONE_NAME = 'vit_giant_patch14_dinov2.lvd142m'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path: str, img_size: int) -> TimmWrapper:
    """Load a pre-trained model"""
    torch.backends.cudnn.benchmark = True  # Enable CUDNN benchmarking

    # Initialize Timm Wrapper Model
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
        img_size=img_size
    ).to(DEVICE)    
    
    # Load the model weights
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    # remove the prefix "module." from the keys of the state_dict
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    embedding_model.load_state_dict(state_dict)
    embedding_model = embedding_model.to(memory_format=torch.channels_last)
    embedding_model.eval()
    
    return embedding_model

def load_timm_model_with_transforms(model_path: str, img_size: int) -> tuple[TimmWrapper, Any]:
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA not available, switching to CPU.")
        device = "cpu"
        
    torch.backends.cudnn.benchmark = True

    # --- 1. Initialize the Model Structure ---
    # These parameters should match the model you saved in model_path
    embedding_size = 518
    
    # We initialize the model structure on the CPU first.
    model = TimmWrapper(
        backbone_name=BACKBONE_NAME,
        embedding_size=embedding_size,
        img_size=img_size
    )

    # --- 2. Load the Trained Weights ---
    # Load state dict, removing "module." prefix if it was saved from DataParallel/DDP
    state_dict = torch.load(model_path, map_location='cpu')
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # --- 3. Finalize Model for Inference ---
    model.to(device) # Move the fully-loaded model to the target device
    model.eval()
    
    # This is a key step for saving memory during inference
    for param in model.parameters():
        param.requires_grad = False

    # --- 4. Get the Preprocessing Transforms ---
    # 'timm' provides a convenient way to get the correct transforms for a given model.
    # We use the base model (model.model) to get its pretrained configuration.
    data_config = resolve_data_config({}, model=model.model)
    transforms = create_transform(**data_config, is_training=False)

    logger.info(f"Model loaded on {device}. Image size: {img_size}x{img_size}")
    logger.info(f"Preprocessing transforms created: {transforms}")

    return model, transforms


if __name__ == "__main__":
    print("This is a module for loading and using a Timm model with specific configurations.")