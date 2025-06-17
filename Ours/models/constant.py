from .CNNCifar import ClientModel, CloudModel, EdgeModel
from .CNNFashion_Mnist import CNNFashion_Mnist
from .CNNMnist import CNNMnist
from .MLP import MLP

model_dataset_map = {
    "cnn":{
        "mnist": CNNMnist,
        "fmnist": CNNFashion_Mnist,
        "cifar": [ClientModel, EdgeModel, CloudModel]
    },
    "mlp": MLP
}


def get_model(model, dataset):
    try:
        return model_dataset_map[model][dataset]
    except:
        exit("Error: unrecognized model")