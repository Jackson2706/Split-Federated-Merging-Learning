from .Alexnet_Cifar import AlexNetCIFAR10
from .CNN_HAM10000 import CNNHAM10000
from .CNNCifar import CNNCifar

model_dataset_map = {
    "resnet50": {
        "cifar": CNNCifar,
        'ham10000': CNNHAM10000
    },
    "alexnet": {
        "cifar": AlexNetCIFAR10
    }
}


def get_model(model, dataset):
    try:
        return model_dataset_map[model][dataset]
    except:
        exit("Error: unrecognized model")