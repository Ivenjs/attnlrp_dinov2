from timm.models.vision_transformer import Block
print(Block.__init__.__code__.co_consts)
type(Block(...).mlp)