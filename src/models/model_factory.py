from .mlp import MLP

# from .cnn import CNN  # we will add later
# from .resnet import ResNet18  # we will add later

def get_model(model_name: str):
    if model_name.lower() == "mlp":
        return MLP()
    elif model_name.lower() == "cnn":
        return CNN()
    elif model_name.lower() == "resnet":
        return ResNet18()
    else:
        raise ValueError(f"Unknown model name: {model_name}")