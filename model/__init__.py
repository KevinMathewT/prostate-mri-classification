from model.resnet.resnet18 import MRI_ResNet18_3D
from model.resnet.resnet50 import MRI_ResNet50_3D
from model.resnet.seresnext50 import MRI_SEResNext50_3D
from model.resnet.seresnext101 import MRI_SEResNext101_3D
from model.vit.vit import MRI_ViT_3D

MODELS = {
    "MRI_ResNet18_3D": MRI_ResNet18_3D,
    "MRI_ResNet50_3D": MRI_ResNet50_3D,
    "MRI_SEResNext50_3D": MRI_SEResNext50_3D,
    "MRI_SEResNext101_3D": MRI_SEResNext101_3D,
    "MRI_ViT_3D": MRI_ViT_3D,
}

def get_model(config):
    model_name = config["train"]["model"]["name"]
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in available models: {MODELS.keys()}")
    model = MODELS[model_name](config)

    # freeze unused layers
    # cross_attn layers are not used in ViT
    for n, p in model.named_parameters():
        if "cross_attn" in n:
            p.requires_grad = False

    return model