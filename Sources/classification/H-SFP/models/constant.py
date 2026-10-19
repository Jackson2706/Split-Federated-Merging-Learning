# pyrefly: ignore [missing-import]
from .alexnet_cifar import (AlexnetClientModel, AlexnetEdgeModel,
                            ALexnetCloudHead)
from .cnn_cifar import ClientModel, CloudModel, EdgeModel
from .resnet50_ham10000 import (HAM10000ClientModelResNet50,
                                    HAM10000CloudModelResNet50,
                                    HAM10000EdgeModelResNet50)
from .vgg_ham10000 import VGGClient_Ours, VGGCloud_Ours, VGGEedge_Ours

model_dataset_map = {
    "resnet50": {
        "ham10000": [
            HAM10000ClientModelResNet50,
            HAM10000EdgeModelResNet50,
            HAM10000CloudModelResNet50,
        ],
        "imagenet": [
            HAM10000ClientModelResNet50,
            HAM10000EdgeModelResNet50,
            HAM10000CloudModelResNet50,
        ],
    
        "cifar100": [ClientModel, EdgeModel, CloudModel],
    },
    "alexnet": {
        "cifar10": [AlexnetClientModel, AlexnetEdgeModel, ALexnetCloudHead]
    },
    "vgg": {"ham10000": [VGGClient_Ours, VGGEedge_Ours, VGGCloud_Ours]},
}


def get_model(model, dataset):
    try:
        return model_dataset_map[model][dataset]
    except:
        exit("Error: unrecognized model")
