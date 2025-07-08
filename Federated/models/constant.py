from .CNNCifar import CNNCifar
from .CNNFashion_Mnist import CNNFashion_Mnist
from .CNNMnist import CNNMnist
from .MLP import MLP
from .CNN_HAM10000 import CNNHAM10000
model_dataset_map = {
    "resnet":{
        "cifar": CNNCifar,
        'ham10000': CNNHAM10000
    },
    "mlp": MLP,
}


def get_model(model, dataset):
    try:
        return model_dataset_map[model][dataset]
    except:
        exit("Error: unrecognized model")