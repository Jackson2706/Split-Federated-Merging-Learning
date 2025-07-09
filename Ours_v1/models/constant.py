from .Alexnet_cifar import (AlexNetClient_Ours, AlexNetCloud_Ours,
                            AlexNetEdge_Ours)
from .CNN_Cifar_ResNet50 import ClientModel, CloudModel, EdgeModel
from .CNN_ham10000_ResNet50 import (HAM10000ClientModelResNet50,
                                    HAM10000CloudModelResNet50,
                                    HAM10000EdgeModelResNet50)

model_dataset_map = {
    "resnet50": {
        "ham10000": [
            HAM10000ClientModelResNet50,
            HAM10000EdgeModelResNet50,
            HAM10000CloudModelResNet50,
        ],
        "cifar": [ClientModel, EdgeModel, CloudModel],
    },
    "alexnet": {
        "cifar": [AlexNetClient_Ours, AlexNetEdge_Ours, AlexNetCloud_Ours]
    },
}


def get_model(model, dataset):
    try:
        return model_dataset_map[model][dataset]
    except:
        exit("Error: unrecognized model")
