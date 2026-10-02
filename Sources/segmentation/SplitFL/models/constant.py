from .Alexnet_Cifar import (AlexNetClient_SplitFed, AlexNetMergedModel,
                            AlexNetServer_SplitFed)
from .CNN_Cifar import CifarClientModel, CifarServerModel, MergedModel
from .CNN_HAM10000_ResNet50 import (HAM10000ClientModelResNet50,
                                    HAM10000MergedModelResNet50,
                                    HAM10000ServerModelResNet50)
from .ISIC_ResNet50 import (ISICClientModelResNet50,
                            ISICMergedModelResNet50,
                            ISICServerModelResNet50)
from .VGG_HAM10000 import (HAM10000MergedModelVGG, VGGClient_SplitFed,
                           VGGServer_SplitFed)

model_dataset_map = {
    "resnet50": {
        "ham10000": [
            HAM10000ClientModelResNet50,
            HAM10000ServerModelResNet50,
            HAM10000MergedModelResNet50,
        ],
        "isic-2018": [
            ISICClientModelResNet50,
            ISICServerModelResNet50,
            ISICMergedModelResNet50,
        ],
        "cifar100": [CifarClientModel, CifarServerModel, MergedModel],
    },
    "alexnet": {
        "cifar10": [
            AlexNetClient_SplitFed,
            AlexNetServer_SplitFed,
            AlexNetMergedModel,
        ]
    },
    "vgg": {
        "ham1000": [
            VGGClient_SplitFed,
            VGGServer_SplitFed,
            HAM10000MergedModelVGG,
        ]
    },
}


def get_model(model, dataset):
    try:
        return model_dataset_map[model][dataset]
    except KeyError:
        exit("Error: unrecognized model or dataset")
