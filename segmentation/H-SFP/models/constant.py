from .ISIC_ResNet50 import (ISICClientModel, ISICCloudDecoder, ISICCloudModel,
                            ISICEdgeModel)

model_dataset_map = {
    "resnet50": {
        "isic-2018": [ISICClientModel, ISICEdgeModel, ISICCloudModel],
    },
}

# Decoder map for full-pipeline inference
decoder_map = {
    "resnet50": {
        "isic-2018": ISICCloudDecoder,
    },
}


def get_model(model, dataset):
    try:
        return model_dataset_map[model][dataset]
    except KeyError:
        exit("Error: unrecognized model or dataset")


def get_decoder(model, dataset):
    try:
        return decoder_map[model][dataset]
    except KeyError:
        exit("Error: unrecognized model or dataset for decoder")
