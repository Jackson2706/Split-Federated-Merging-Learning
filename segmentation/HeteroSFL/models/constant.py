from .ISIC_ResNet50 import (ISICClientModelResNet50, ISICMergedModelResNet50,
                            ISICServerModelResNet50)

model_dataset_map = {
    "resnet50": {
        "isic-2018": [
            ISICClientModelResNet50,
            ISICServerModelResNet50,
            ISICMergedModelResNet50,
        ]
    },
}


def get_model(model, dataset):
    try:
        return model_dataset_map[model][dataset]
    except KeyError:
        exit("Error: unrecognized model or dataset")
