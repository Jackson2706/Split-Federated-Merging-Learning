from .Alexnet_cifar import (AlexNetClient_Ours, AlexNetCloud_Ours,
                            AlexNetEdge_Ours)
from .CNN_Cifar import ClientModel, CloudModel, EdgeModel
from .CNN_ham10000_ResNet50 import (HAM10000ClientModelResNet50,
                                    HAM10000CloudModelResNet50,
                                    HAM10000EdgeModelResNet50)
from .ISIC_ResNet50 import (ISICClientModelResNet50,
                            ISICCloudModelResNet50,
                            ISICEdgeModelResNet50)
from .VGG_HAM10000 import VGGClient_Ours, VGGCloud_Ours, VGGEedge_Ours

model_dataset_map = {
    "resnet50": {
        "ham10000": [
            HAM10000ClientModelResNet50,
            HAM10000EdgeModelResNet50,
            HAM10000CloudModelResNet50,
        ],
        "isic-2018": [
            ISICClientModelResNet50,
            ISICEdgeModelResNet50,
            ISICCloudModelResNet50,
        ],
        "cifar100": [ClientModel, EdgeModel, CloudModel],
    },
    "alexnet": {
        "cifar10": [AlexNetClient_Ours, AlexNetEdge_Ours, AlexNetCloud_Ours]
    },
    "vgg": {"ham10000": [VGGClient_Ours, VGGEedge_Ours, VGGCloud_Ours]},
}


def get_model(model, dataset):
    try:
        return model_dataset_map[model][dataset]
    except:
        exit("Error: unrecognized model")
