from .Alexnet_cifar import (
    AlexnetClientModel, AlexnetEdgeModel, ALexnetCloudHead)
from .CNN_Cifar import ClientModel, CloudModel, EdgeModel
from .CNN_ham10000_ResNet50 import (
    HAM10000ClientModelResNet50,
    HAM10000CloudModelResNet50,
    HAM10000EdgeModelResNet50,
)
from .VGG_HAM10000 import VGGClient_Ours, VGGCloud_Ours, VGGEedge_Ours

model_dataset_map = {
    "resnet50": {
        "ham10000": [
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
