from .Alexnet_Cifar import AlexNetCIFAR10
from .CNN_HAM10000 import CNNHAM10000
from .CNNCifar import CNNCifar
from .VGG_HAM10000 import VGGHAM10000
model_dataset_map = {
    "resnet50": {
        "cifar100": CNNCifar,
        'ham10000': CNNHAM10000
    },
    "alexnet": {
        "cifar10": AlexNetCIFAR10
    },
    "vgg": {
        "ham10000": VGGHAM10000
    }
}


def get_model(model, dataset):
    try:
        return model_dataset_map[model][dataset]
    except KeyError:
        exit("Error: unrecognized model")